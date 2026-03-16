import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.policy.bandit_model import LinUCBModel
from src.policy.base_policy import BaseMemoryPolicy
from src.policy.reward import compute_memory_selection_reward


class RLMemoryPolicy(BaseMemoryPolicy):
    """
    contextual bandit（LinUCB）版本。
    不依赖 GPU，可先在线收集和离线更新。
    这里不再使用任何写死实体/任务类型规则，只使用通用特征。
    """

    FEATURE_DIM = 9

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
        """
        通用轻量分词：
        - 英文词
        - 数字
        - 中文单字
        """
        text = self._normalize_text(text)
        if not text:
            return []

        tokens: List[str] = []

        # 英文词 / 数字
        tokens.extend(re.findall(r"[a-z]+|\d+", text))

        # 中文单字
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))

        return tokens

    def _overlap_ratio(self, query: str, content: str) -> float:
        q_tokens = self._tokenize(query)
        c_tokens = self._tokenize(content)
        if not q_tokens or not c_tokens:
            return 0.0

        q_set = set(q_tokens)
        c_set = set(c_tokens)
        overlap = len(q_set & c_set)
        return overlap / max(len(q_set), 1)

    def _jaccard(self, query: str, content: str) -> float:
        q_tokens = set(self._tokenize(query))
        c_tokens = set(self._tokenize(content))
        if not q_tokens or not c_tokens:
            return 0.0
        union = q_tokens | c_tokens
        inter = q_tokens & c_tokens
        return len(inter) / max(len(union), 1)

    def _feature_vector(
        self,
        item: Dict[str, Any],
        rank_idx: int,
        task_context,
        query: str,
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
        ], dtype=np.float64)

    def select_memories(
        self,
        query: str,
        task_context,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scored = []
        for idx, item in enumerate(candidates):
            x = self._feature_vector(item, idx, task_context, query)
            bandit_score = self.model.score(x)

            enriched = dict(item)
            enriched["policy_score"] = bandit_score
            enriched["_policy_feature"] = x.tolist()
            scored.append(enriched)

        scored.sort(key=lambda x: x["policy_score"], reverse=True)
        return scored[: self.max_select_k]

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
    ) -> None:
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
            "query": query,
            "memory_written": memory_written,
            "latency_ms": latency_ms,
            "reward": reward,
            "selected_count": len(selected_memories),
            "selected_memories": [
                {
                    "task_id": x.get("task_id"),
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
