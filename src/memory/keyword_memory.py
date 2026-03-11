import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import jieba

from src.runtime.task_context import TaskContext
from .base_memory import BaseMemoryManager
from .memory_record import MemoryRecord


def tokenize_zh(text: str) -> List[str]:
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


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

    def _build_key(self, experiment_id: str, task_id: str, query: str) -> str:
        return f"{experiment_id}::{task_id}::{query.strip()}"

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
                self._memory_bank.append(record)
                self._memory_keys.add(
                    self._build_key(
                        record.get("experiment_id", ""),
                        record.get("task_id", ""),
                        record.get("query", ""),
                    )
                )

    def _append_memory_to_file(self, record: Dict[str, Any]) -> None:
        if not self.persistent:
            return

        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _is_history_memory(mem: Dict[str, Any], task_context: TaskContext) -> bool:
        if mem.get("experiment_id") != task_context.experiment_id:
            return False

        mem_order = mem.get("task_order")
        if mem_order is None:
            return False

        if mem_order >= task_context.task_order:
            return False

        mem_created_at = parse_iso_time(mem.get("created_at"))
        if mem_created_at is None:
            return False

        if mem_created_at >= task_context.task_start_time:
            return False

        return True

    def retrieve_memory(
        self,
        query: str,
        task_context: TaskContext,
        top_k: Optional[int] = None,
    ) -> List[str]:
        items = self.retrieve_memory_with_scores(
            query=query,
            task_context=task_context,
            top_k=top_k,
        )
        return [item["content"] for item in items]

    def retrieve_memory_with_scores(
        self,
        query: str,
        task_context: TaskContext,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.default_top_k

        query_terms = set(tokenize_zh(query))
        scored = []

        for mem in self._memory_bank:
            if not self._is_history_memory(mem, task_context=task_context):
                continue

            content = mem.get("memory_summary", mem.get("content", ""))
            overlap = len(query_terms & set(tokenize_zh(content)))
            if overlap > 0:
                scored.append(
                    {
                        "experiment_id": mem.get("experiment_id"),
                        "task_id": mem.get("task_id"),
                        "task_order": mem.get("task_order"),
                        "query": mem.get("query"),
                        "answer": mem.get("answer"),
                        "experience": mem.get("experience"),
                        "content": content,
                        "score": float(overlap),
                        "created_at": mem.get("created_at"),
                        "answer_raw": mem.get("answer_raw", ""),
                        "memory_summary": mem.get("memory_summary", content),
                        "strategy_note": mem.get("strategy_note", ""),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def write_memory(self, record: MemoryRecord) -> None:
        key = self._build_key(record.experiment_id, record.task_id, record.query)
        if self.deduplicate and key in self._memory_keys:
            return

        payload = record.to_dict()
        payload["content"] = record.to_retrieval_text()

        self._memory_bank.append(payload)
        self._memory_keys.add(key)
        self._append_memory_to_file(payload)
