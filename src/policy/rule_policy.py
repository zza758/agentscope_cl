from typing import List, Dict, Any

from src.policy.base_policy import BaseMemoryPolicy


class RuleBasedMemoryPolicy(BaseMemoryPolicy):
    def __init__(self, max_select_k: int = 3, min_summary_len: int = 10):
        self.max_select_k = max_select_k
        self.min_summary_len = min_summary_len

    def _safe_float(self, x, default=0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _tokenize(self, text: str):
        if not text:
            return set()
        text = text.lower().strip()
        tokens = set()

        buf = []
        for ch in text:
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.add("".join(buf))
                    buf = []
                if "\u4e00" <= ch <= "\u9fff":
                    tokens.add(ch)
        if buf:
            tokens.add("".join(buf))

        return tokens

    def _overlap_score(self, query: str, content: str) -> float:
        q = self._tokenize(query)
        c = self._tokenize(content)
        if not q or not c:
            return 0.0
        return len(q & c) / max(len(q), 1)

    def select_memories(
        self,
        query: str,
        task_context,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        chosen = []
        chosen_ids = set()

        def add_item(item):
            key = item.get("task_id") or item.get("content")
            if key in chosen_ids:
                return
            chosen.append(item)
            chosen_ids.add(key)

        # 1) 保留 base_score 最高的一条
        best_base = max(candidates, key=lambda x: self._safe_float(x.get("score"), 0.0))
        add_item(best_base)

        # 2) 如果有 contrastive_score，保留 rerank 最高的一条
        rerank_candidates = [x for x in candidates if x.get("contrastive_score") is not None]
        if rerank_candidates:
            best_rerank = max(
                rerank_candidates,
                key=lambda x: self._safe_float(x.get("contrastive_score"), 0.0),
            )
            add_item(best_rerank)

        # 3) 剩余按 query-content overlap 排序补齐
        remaining = [x for x in candidates if (x.get("task_id") or x.get("content")) not in chosen_ids]
        remaining.sort(
            key=lambda x: (
                self._overlap_score(query, x.get("content", "")),
                self._safe_float(x.get("contrastive_score"), -1.0),
                self._safe_float(x.get("score"), 0.0),
            ),
            reverse=True,
        )

        for item in remaining:
            if len(chosen) >= self.max_select_k:
                break
            add_item(item)

        return chosen[: self.max_select_k]

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
