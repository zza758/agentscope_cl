from typing import Dict, Any, List, Optional, Set


def split_entities(entity: Optional[str]) -> List[str]:
    if not entity:
        return []
    return [x for x in entity.split("_") if x]


def entity_overlap(task_entity: Optional[str], mem_entity: Optional[str]) -> int:
    q = set(split_entities(task_entity))
    m = set(split_entities(mem_entity))
    return len(q & m)


def entity_gain(
    task_entity: Optional[str],
    covered_entities: Set[str],
    mem_entity: Optional[str],
) -> int:
    target = set(split_entities(task_entity))
    mem = set(split_entities(mem_entity))
    if not target or not mem:
        return 0
    return len((target - covered_entities) & mem)


def same_task_type(task_type: Optional[str], mem_task_type: Optional[str]) -> float:
    if not task_type or not mem_task_type:
        return 0.0
    return 1.0 if task_type == mem_task_type else 0.0


def metadata_score(
    task_type: Optional[str],
    mem_task_type: Optional[str],
    task_entity: Optional[str],
    mem_entity: Optional[str],
    type_weight: float = 0.15,
    entity_weight: float = 0.15,
) -> float:
    return (
        type_weight * same_task_type(task_type, mem_task_type)
        + entity_weight * float(entity_overlap(task_entity, mem_entity))
    )


def coverage_aware_select(
    candidates: List[Dict[str, Any]],
    top_k: int,
    task_entity: Optional[str],
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """
    多实体任务：先保证每个目标实体至少有一个候选，再按 score_key 补齐。
    单实体或无实体任务：直接按 score_key 排序截断。
    """
    if not candidates or top_k <= 0:
        return []

    ranked = sorted(candidates, key=lambda x: x.get(score_key, 0.0), reverse=True)

    target_entities = split_entities(task_entity)
    if len(target_entities) <= 1:
        return ranked[:top_k]

    chosen = []
    chosen_ids = set()

    for ent in target_entities:
        bucket = [x for x in ranked if ent in split_entities(x.get("entity"))]
        if not bucket:
            continue
        best = bucket[0]
        key = best.get("task_id") or best.get("content")
        if key not in chosen_ids:
            chosen.append(best)
            chosen_ids.add(key)

    for item in ranked:
        if len(chosen) >= top_k:
            break
        key = item.get("task_id") or item.get("content")
        if key in chosen_ids:
            continue
        chosen.append(item)
        chosen_ids.add(key)

    return chosen[:top_k]
