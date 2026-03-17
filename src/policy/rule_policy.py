from typing import List, Dict, Any, Set

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

    def _split_entities(self, entity: str) -> List[str]:
        if not entity:
            return []
        return [x for x in entity.split("_") if x]

    def _entity_overlap(self, query_entity: str, mem_entity: str) -> int:
        q = set(self._split_entities(query_entity))
        m = set(self._split_entities(mem_entity))
        return len(q & m)

    def _entity_gain(self, query_entity: str, covered_entities: Set[str], mem_entity: str) -> int:
        target = set(self._split_entities(query_entity))
        mem = set(self._split_entities(mem_entity))
        if not target or not mem:
            return 0
        return len((target - covered_entities) & mem)

    def _same_task_type(self, query_task_type: str, mem_task_type: str) -> float:
        if not query_task_type or not mem_task_type:
            return 0.0
        return 1.0 if query_task_type == mem_task_type else 0.0

    def select_memories(
        self,
        query: str,
        task_context,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_task_type = getattr(task_context, "task_type", None)
        query_entity = getattr(task_context, "task_entity", None)

        chosen = []
        chosen_ids = set()
        covered_entities = set()

        def add_item(item):
            key = item.get("task_id") or item.get("content")
            if key in chosen_ids:
                return False
            chosen.append(item)
            chosen_ids.add(key)
            covered_entities.update(self._split_entities(item.get("entity", "")))
            return True

        # 1) 先保一个 base_score 最强的
        best_base = max(candidates, key=lambda x: self._safe_float(x.get("score"), 0.0))
        add_item(best_base)

        # 2) 再保一个 contrastive_score 最强的
        rerank_candidates = [x for x in candidates if x.get("contrastive_score") is not None]
        if rerank_candidates:
            best_rerank = max(
                rerank_candidates,
                key=lambda x: self._safe_float(x.get("contrastive_score"), 0.0),
            )
            add_item(best_rerank)

        # 3) 剩余候选按 coverage-aware 逻辑补齐
        remaining = [
            x for x in candidates
            if (x.get("task_id") or x.get("content")) not in chosen_ids
        ]

        remaining.sort(
            key=lambda x: (
                self._entity_gain(query_entity, covered_entities, x.get("entity", "")),
                self._same_task_type(query_task_type, x.get("task_type")),
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
