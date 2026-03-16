from typing import Any, Dict, List, Optional, Set


def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def extract_used_task_ids(selected_memories: List[Dict[str, Any]]) -> Set[str]:
    ids = set()
    for item in selected_memories:
        task_id = item.get("task_id")
        if task_id:
            ids.add(task_id)
    return ids


def compute_memory_selection_reward(
    selected_memories: List[Dict[str, Any]],
    support_task_ids: Optional[List[str]] = None,
    memory_written: bool = True,
    write_reward: float = 0.2,
    hit_reward: float = 1.0,
    miss_penalty: float = -0.2,
    redundancy_penalty: float = -0.05,
) -> float:
    reward = 0.0

    support_set = set(support_task_ids or [])
    used_set = extract_used_task_ids(selected_memories)

    if support_set:
        hit_count = len(support_set & used_set)
        miss_count = len(support_set - used_set)
        reward += hit_reward * hit_count
        reward += miss_penalty * miss_count

    seen = set()
    for item in selected_memories:
        key = item.get("task_id") or item.get("content")
        if key in seen:
            reward += redundancy_penalty
        seen.add(key)

        if item.get("contrastive_score") is not None:
            reward += 0.2 * max(safe_float(item.get("contrastive_score")), 0.0)
        elif item.get("score") is not None:
            reward += 0.05 * max(safe_float(item.get("score")), 0.0)

    if memory_written:
        reward += write_reward

    return reward
