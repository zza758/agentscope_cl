from typing import List, Dict, Any

from src.policy.base_policy import BaseMemoryPolicy


class RuleBasedMemoryPolicy(BaseMemoryPolicy):
    def __init__(self, max_select_k: int = 3, min_summary_len: int = 10):
        self.max_select_k = max_select_k
        self.min_summary_len = min_summary_len

    def select_memories(self, query: str, task_context, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return candidates[: self.max_select_k]

    def should_write_memory(
        self,
        query: str,
        task_context,
        final_answer: str,
        memory_summary: str,
        strategy_note: str,
    ) -> bool:
        if not memory_summary:
            return False
        if len(memory_summary.strip()) < self.min_summary_len:
            return False
        return True

    def on_task_end(
            self,
            query: str,
            task_context,
            selected_memories,
            final_answer: str,
            memory_summary: str,
            strategy_note: str,
            memory_written: bool = True,
            latency_ms=None,
            task_id=None,
            task_order=None,
            support_task_ids=None,
    ) -> None:
        return None
