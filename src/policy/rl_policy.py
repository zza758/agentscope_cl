import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from src.policy.bandit_model import LinUCBModel
from src.policy.base_policy import BaseMemoryPolicy
from src.policy.reward import compute_memory_selection_reward


class RLMemoryPolicy(BaseMemoryPolicy):
    """
    contextual bandit（LinUCB）版本。
    对齐 retrieval v2：
    - 利用 task_type / entity metadata
    - 利用 coverage-aware greedy selection
    - 不做硬编码实体名匹配
    """

    FEATURE_DIM = 11

    def __init__(
        self,
        max_select_k: int = 3,
        min_summary_len: int = 10,
        alpha: float = 0.5,
        model_path: str = "outputs/rl_policy/linucb_state.json",
        log_path: str = "outputs/rl_policy/decision_log.jsonl",
        online_update: bool = False,
        write_reward: float = 0.2,
        hit_reward: float = 1.0,
        miss_penalty: float = -0.2,
    ):
        self.max_select_k = max_select_k
        self.min_summary_len = min_summary_len
        self.model_path = model_path
        self.log_path = log_path
        self.online_update = online_update
        self.write_reward = write_reward
        self.hit_reward = hit_reward
        self.miss_penalty = miss_penalty

        p = Path(model_path)
        if p.exists():
            self.model = LinUCBModel.load(model_path)
        else:
            self.model = LinUCBModel(dim=self.FEATURE_DIM, alpha=alpha)

    def _safe_float(self, x, default=0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        return text.strip().lower()

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        tokens: List[str] = []
        tokens.extend(re.findall(r"[a-z]+|\d+", text))
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))
        return tokens

    def _overlap_ratio(self, query: str, content: str) -> float:
        q_tokens = set(self._tokenize(query))
        c_tokens = set(self._tokenize(content))
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / max(len(q_tokens), 1)

    def _jaccard(self, query: str, content: str) -> float:
        q_tokens = set(self._tokenize(query))
        c_tokens = set(self._tokenize(content))
        if not q_tokens or not c_tokens:
            return 0.0
        union = q_tokens | c_tokens
        inter = q_tokens & c_tokens
        return len(inter) / max(len(union), 1)

    def _split_entities(self, entity: Optional[str]) -> List[str]:
        if not entity:
            return []
        return [x for x in entity.split("_") if x]

    def _entity_overlap(self, query_entity: Optional[str], mem_entity: Optional[str]) -> int:
        q = set(self._split_entities(query_entity))
        m = set(self._split_entities(mem_entity))
        return len(q & m)

    def _entity_gain(
        self,
        query_entity: Optional[str],
        covered_entities: Set[str],
        mem_entity: Optional[str],
    ) -> int:
        target = set(self._split_entities(query_entity))
        mem = set(self._split_entities(mem_entity))
        if not target or not mem:
            return 0
        return len((target - covered_entities) & mem)

    def _same_task_type(self, query_task_type: Optional[str], mem_task_type: Optional[str]) -> float:
        if not query_task_type or not mem_task_type:
            return 0.0
        return 1.0 if query_task_type == mem_task_type else 0.0

    def _feature_vector(
        self,
        item: Dict[str, Any],
        rank_idx: int,
        task_context,
        query: str,
        covered_entities: Optional[Set[str]] = None,
    ) -> np.ndarray:
        base_score = self._safe_float(item.get("score"), 0.0)
        rerank_score = self._safe_float(item.get("contrastive_score"), 0.0)
        base_minus_rerank = base_score - rerank_score

        mem_task_order = item.get("task_order")
        if mem_task_order is None:
            order_gap = 0.0
        else:
            order_gap = float(max(task_context.task_order - int(mem_task_order), 0))

        content = (item.get("content") or "").strip()
        content_len = float(min(len(content), 500)) / 500.0

        has_rerank = 1.0 if item.get("contrastive_score") is not None else 0.0
        rank_feature = 1.0 / float(rank_idx + 1)

        overlap_ratio = self._overlap_ratio(query, content)
        jaccard = self._jaccard(query, content)

        query_task_type = getattr(task_context, "task_type", None)
        query_entity = getattr(task_context, "task_entity", None)

        mem_task_type = item.get("task_type")
        mem_entity = item.get("entity")

        same_task_type = self._same_task_type(query_task_type, mem_task_type)
        entity_overlap = float(self._entity_overlap(query_entity, mem_entity))

        if covered_entities is None:
            covered_entities = set()
        entity_gain = float(self._entity_gain(query_entity, covered_entities, mem_entity))

        return np.array([
            base_score,
            rerank_score,
            base_minus_rerank,
            rank_feature,
            order_gap / 20.0,
            content_len,
            has_rerank,
            overlap_ratio,
            jaccard,
            same_task_type,
            entity_overlap + entity_gain,
        ], dtype=np.float64)

    def select_memories(
        self,
        query: str,
        task_context,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        chosen = []
        remaining = [dict(x) for x in candidates]
        covered_entities: Set[str] = set()

        for _ in range(min(self.max_select_k, len(remaining))):
            scored = []
            for idx, item in enumerate(remaining):
                x = self._feature_vector(
                    item=item,
                    rank_idx=idx,
                    task_context=task_context,
                    query=query,
                    covered_entities=covered_entities,
                )
                bandit_score = self.model.score(x)

                # 轻量 coverage-aware greedy bonus
                gain = self._entity_gain(
                    getattr(task_context, "task_entity", None),
                    covered_entities,
                    item.get("entity"),
                )
                final_score = bandit_score + 0.1 * gain

                enriched = dict(item)
                enriched["policy_score"] = final_score
                enriched["_policy_feature"] = x.tolist()
                scored.append(enriched)

            scored.sort(key=lambda x: x["policy_score"], reverse=True)
            best = scored[0]
            chosen.append(best)

            covered_entities.update(self._split_entities(best.get("entity")))
            best_key = best.get("task_id") or best.get("content")
            remaining = [
                x for x in remaining
                if (x.get("task_id") or x.get("content")) != best_key
            ]

        return chosen

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

    def _compute_reward(
        self,
        selected_memories,
        memory_written: bool,
        support_task_ids=None,
    ) -> float:
        return compute_memory_selection_reward(
            selected_memories=selected_memories,
            support_task_ids=support_task_ids,
            memory_written=memory_written,
            write_reward=self.write_reward,
            hit_reward=self.hit_reward,
            miss_penalty=self.miss_penalty,
        )

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
    ) -> dict[str, Union[float, str, int]]:
        reward = self._compute_reward(
            selected_memories=selected_memories,
            memory_written=memory_written,
            support_task_ids=support_task_ids,
        )

        if self.online_update:
            for item in selected_memories:
                feat = item.get("_policy_feature")
                if feat is None:
                    continue
                self.model.update(np.array(feat, dtype=np.float64), reward)
            self.model.save(self.model_path)

        log_record = {
            "task_id": task_id or getattr(task_context, "task_id", None),
            "task_order": task_order or getattr(task_context, "task_order", None),
            "task_type": getattr(task_context, "task_type", None),
            "task_entity": getattr(task_context, "task_entity", None),
            "query": query,
            "memory_written": memory_written,
            "latency_ms": latency_ms,
            "reward": reward,
            "selected_count": len(selected_memories),
            "selected_memories": [
                {
                    "task_id": x.get("task_id"),
                    "task_type": x.get("task_type"),
                    "entity": x.get("entity"),
                    "score": x.get("score"),
                    "contrastive_score": x.get("contrastive_score"),
                    "policy_score": x.get("policy_score"),
                    "policy_feature": x.get("_policy_feature"),
                }
                for x in selected_memories
            ],
            "memory_summary": memory_summary,
            "strategy_note": strategy_note,
            "support_task_ids": support_task_ids or [],
        }

        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        return {
            "reward": reward,
            "policy_name": "linucb",
            "selected_count": len(selected_memories),
        }
