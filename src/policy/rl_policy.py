from typing import Any, Dict, List

from src.policy.base_policy import BaseMemoryPolicy


class RLMemoryPolicy(BaseMemoryPolicy):
    def __init__(
        self,
        max_select_k: int = 3,
        min_summary_len: int = 10,
        score_threshold: float = 0.0,
    ):
        self.max_select_k = max_select_k
        self.min_summary_len = min_summary_len
        self.score_threshold = score_threshold

    def _get_candidate_score(self, item: Dict[str, Any]) -> float:
        if item.get("contrastive_score") is not None:
            return float(item["contrastive_score"])
        if item.get("score") is not None:
            return float(item["score"])
        return 0.0

    def select_memories(self, query: str, task_context, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [x for x in candidates if self._get_candidate_score(x) >= self.score_threshold]
        if not filtered:
            filtered = candidates
        return filtered[: self.max_select_k]

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
