import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.runtime.task_context import TaskContext
from .base_memory import BaseMemoryManager
from .memory_record import MemoryRecord
from src.runtime.history_guard import is_legal_history_record
from src.memory.retrieval_utils import metadata_score, coverage_aware_select


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


def parse_iso_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


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

                # 兼容旧格式
                if "memory_summary" not in record:
                    fallback_summary = record.get("content") or record.get("answer") or ""
                    record["memory_summary"] = fallback_summary

                if "answer_raw" not in record:
                    record["answer_raw"] = record.get("answer", record.get("content", ""))

                if "strategy_note" not in record:
                    record["strategy_note"] = record.get(
                        "experience",
                        "可作为后续相似任务的参考经验。",
                    )

                if "content" not in record:
                    record["content"] = record.get("memory_summary", "")

                # structured / benchmark 默认字段
                record.setdefault("stream_id", None)
                record.setdefault("task_type", None)
                record.setdefault("entity", None)
                record.setdefault("support_task_ids", [])
                record.setdefault("source_dataset", None)
                record.setdefault("source_sample_id", None)
                record.setdefault("meta", {})

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

            # 完全无内容信号且无元数据信号，不保留
            if base_score <= 0 and meta_score <= 0:
                continue

            scored.append(
                {
                    "experiment_id": mem.get("experiment_id"),
                    "stream_id": mem.get("stream_id"),
                    "task_id": mem.get("task_id"),
                    "task_order": mem.get("task_order"),
                    "query": mem.get("query"),
                    "content": content,
                    "score": final_score,
                    "base_score": base_score,
                    "meta_score": meta_score,
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
                }
            )

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

            if (new_score > old_score) or (
                new_score == old_score and new_order < old_order
            ):
                dedup_map[norm_text] = item

        return list(dedup_map.values())
