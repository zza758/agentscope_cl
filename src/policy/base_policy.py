from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseMemoryPolicy(ABC):
    @abstractmethod
    def select_memories(
        self,
        query: str,
        task_context,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def should_write_memory(
        self,
        query: str,
        task_context,
        final_answer: str,
        memory_summary: str,
        strategy_note: str,
    ) -> bool:
        return True

    def build_write_record(
        self,
        query: str,
        task_context,
        final_answer: str,
        memory_summary: str,
        strategy_note: str,
    ) -> Optional[Dict[str, str]]:
        return {
            "answer_raw": final_answer,
            "memory_summary": memory_summary,
            "strategy_note": strategy_note,
        }

    def on_task_end(
            self,
            query: str,
            task_context,
            selected_memories: List[Dict[str, Any]],
            final_answer: str,
            memory_summary: str,
            strategy_note: str,
            memory_written: bool = True,
            latency_ms=None,
            task_id=None,
            task_order=None,
            support_task_ids=None,
    ) -> None:
        pass
