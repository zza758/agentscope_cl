import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.memory.retrieval_utils import coverage_aware_select, metadata_score
from src.runtime.history_guard import is_legal_history_record
from src.runtime.task_context import TaskContext

from .base_memory import BaseMemoryManager
from .memory_record import MemoryRecord


def tokenize_text(text: str) -> List[str]:
    """
    轻量 tokenizer：
    - 英文/数字/下划线：按连续 token 切分
    - 中文：按单字切分
    目的不是做高质量中文分词，而是提供一个稳定、零额外依赖的 keyword 检索基线。
    对 HotpotQA / MuSiQue 这类英文 benchmark 已足够用于 smoke test。
    """
    text = (text or "").strip()
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return [t.strip().lower() for t in tokens if t.strip()]


class KeywordMemoryManager(BaseMemoryManager):
    def __init__(
        self,
        storage_path: str,
        default_top_k: int = 3,
        persistent: bool = True,
        deduplicate: bool = True,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_top_k = default_top_k
        self.persistent = persistent
        self.deduplicate = deduplicate
        self._memory_bank: List[Dict[str, Any]] = []
        self._memory_keys = set()
        if self.persistent:
            self._load_memories()

    def _build_key(
        self,
        experiment_id: str,
        task_id: str,
        query: str,
        stream_id: str = "",
    ) -> str:
        return f"{experiment_id}::{stream_id}::{task_id}::{query.strip()}"

    def _load_memories(self) -> None:
        self._memory_bank = []
        self._memory_keys = set()
        if not self.storage_path.exists():
            return

        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                if "memory_summary" not in record:
                    fallback_summary = record.get("content") or record.get("answer") or ""
                    record["memory_summary"] = fallback_summary
                if "answer_raw" not in record:
                    record["answer_raw"] = record.get("answer", record.get("content", ""))
                if "strategy_note" not in record:
                    record["strategy_note"] = record.get(
                        "experience", "可作为后续相似任务的参考经验。"
                    )
                if "content" not in record:
                    record["content"] = record.get("memory_summary", "")

                record.setdefault("stream_id", None)
                record.setdefault("task_type", None)
                record.setdefault("entity", None)
                record.setdefault("support_task_ids", [])
                record.setdefault("source_dataset", None)
                record.setdefault("source_sample_id", None)
                record.setdefault("meta", {})
                record.setdefault("memory_quality", "unknown")
                record.setdefault("contains_placeholder", False)
                record.setdefault("contains_unknown", False)
                record.setdefault("gate_passed", True)
                record.setdefault("gate_reason", "")

                self._memory_bank.append(record)
                self._memory_keys.add(
                    self._build_key(
                        record.get("experiment_id", ""),
                        record.get("task_id", ""),
                        record.get("query", ""),
                        record.get("stream_id", "") or "",
                    )
                )

    def _append_memory_to_file(self, record: Dict[str, Any]) -> None:
        if not self.persistent:
            return
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve_memory(
        self,
        query: str,
        task_context: TaskContext,
        top_k: Optional[int] = None,
        task_type: str = None,
        task_entity: str = None,
    ) -> List[str]:
        items = self.retrieve_memory_with_scores(
            query=query,
            task_context=task_context,
            top_k=top_k,
            task_type=task_type,
            task_entity=task_entity,
        )
        return [item["content"] for item in items]

    def retrieve_memory_with_scores(
        self,
        query: str,
        task_context: TaskContext,
        top_k: Optional[int] = None,
        task_type: str = None,
        task_entity: str = None,
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.default_top_k

        query_terms = set(tokenize_text(query))
        scored = []
        for mem in self._memory_bank:
            if not is_legal_history_record(mem, task_context):
                continue

            content = mem.get("memory_summary", mem.get("content", ""))
            mem_terms = set(tokenize_text(content))
            overlap = len(query_terms & mem_terms)
            base_score = float(overlap)
            meta_score = metadata_score(
                task_type=task_type,
                mem_task_type=mem.get("task_type"),
                task_entity=task_entity,
                mem_entity=mem.get("entity"),
            )
            final_score = base_score + meta_score
            if base_score <= 0 and meta_score <= 0:
                continue

            scored.append(self._build_item_from_memory_record(mem=mem, score=final_score))

        scored.sort(key=lambda x: x["score"], reverse=True)
        deduped = self._deduplicate_scored_items(scored)
        deduped.sort(key=lambda x: x["score"], reverse=True)
        return coverage_aware_select(
            candidates=deduped,
            top_k=top_k,
            task_entity=task_entity,
            score_key="score",
        )

    def write_memory(self, record: MemoryRecord) -> None:
        if getattr(record, "gate_passed", True) is False:
            return
        if getattr(record, "memory_quality", "unknown") == "reject":
            return

        key = self._build_key(
            record.experiment_id,
            record.task_id,
            record.query,
            getattr(record, "stream_id", "") or "",
        )
        if self.deduplicate and key in self._memory_keys:
            return

        payload = record.to_dict()
        payload["content"] = record.to_retrieval_text()
        self._memory_bank.append(payload)
        self._memory_keys.add(key)
        self._append_memory_to_file(payload)

    @staticmethod
    def _normalize_dedup_text(text: str) -> str:
        if not text:
            return ""
        normalized = " ".join(text.strip().split())
        return normalized.lower()

    @staticmethod
    def _deduplicate_scored_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup_map: Dict[str, Dict[str, Any]] = {}
        for item in items:
            text = item.get("memory_summary") or item.get("content", "")
            norm_text = KeywordMemoryManager._normalize_dedup_text(text)
            if not norm_text:
                continue
            if norm_text not in dedup_map:
                dedup_map[norm_text] = item
                continue

            old_item = dedup_map[norm_text]
            old_score = float(old_item.get("score", 0.0))
            new_score = float(item.get("score", 0.0))
            old_order = old_item.get("task_order", 10**9)
            new_order = item.get("task_order", 10**9)
            if (new_score > old_score) or (new_score == old_score and new_order < old_order):
                dedup_map[norm_text] = item
        return list(dedup_map.values())

    def _build_item_from_memory_record(
        self,
        mem: Dict[str, Any],
        score: float,
        support_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        content = mem.get("memory_summary", mem.get("content", ""))
        return {
            "experiment_id": mem.get("experiment_id"),
            "stream_id": mem.get("stream_id"),
            "task_id": mem.get("task_id"),
            "task_order": mem.get("task_order"),
            "query": mem.get("query"),
            "content": content,
            "score": float(score),
            "base_score": float(score),
            "meta_score": 0.0,
            "created_at": mem.get("created_at"),
            "answer_raw": mem.get("answer_raw", ""),
            "memory_summary": mem.get("memory_summary", content),
            "strategy_note": mem.get("strategy_note", ""),
            "task_type": mem.get("task_type"),
            "entity": mem.get("entity"),
            "support_task_ids": mem.get("support_task_ids", []),
            "source_dataset": mem.get("source_dataset"),
            "source_sample_id": mem.get("source_sample_id"),
            "meta": mem.get("meta", {}),
            "memory_quality": mem.get("memory_quality", "unknown"),
            "contains_placeholder": mem.get("contains_placeholder", False),
            "contains_unknown": mem.get("contains_unknown", False),
            "gate_passed": mem.get("gate_passed", True),
            "gate_reason": mem.get("gate_reason", ""),
            "is_support_memory": support_rank is not None,
            "support_rank": support_rank,
        }

    def get_memories_by_task_ids(
        self,
        task_ids: List[str],
        task_context: TaskContext,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not task_ids:
            return []

        task_id_order = {task_id: idx for idx, task_id in enumerate(task_ids)}
        matched: List[Dict[str, Any]] = []
        for mem in self._memory_bank:
            mem_task_id = mem.get("task_id")
            if mem_task_id not in task_id_order:
                continue
            if not is_legal_history_record(mem, task_context):
                continue

            support_rank = task_id_order[mem_task_id]
            score = 1000.0 - float(support_rank)
            matched.append(
                self._build_item_from_memory_record(mem=mem, score=score, support_rank=support_rank)
            )

        matched.sort(
            key=lambda x: (
                x.get("support_rank", 10**9),
                x.get("task_order", 10**9),
                str(x.get("task_id", "")),
            )
        )
        deduped = self._deduplicate_scored_items(matched)
        if limit is not None:
            deduped = deduped[:limit]
        return deduped
