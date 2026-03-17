import json
from pathlib import Path
from typing import Dict, List, Set


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_support_parent_map(labels) -> Dict[str, List[str]]:
    return {
        x["task_id"]: list(x.get("support_memory_task_ids", []))
        for x in labels
    }


def get_ancestors(task_id: str, parent_map: Dict[str, List[str]]) -> Set[str]:
    visited = set()
    stack = list(parent_map.get(task_id, []))

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(parent_map.get(cur, []))

    return visited


def build_ancestor_map(labels) -> Dict[str, Set[str]]:
    parent_map = build_support_parent_map(labels)
    return {task_id: get_ancestors(task_id, parent_map) for task_id in parent_map}


def infer_entity_group(entity_value: str) -> Set[str]:
    if not entity_value:
        return set()

    text = entity_value.lower()
    parts = text.replace("-", "_").split("_")
    return {p for p in parts if p}


def build_entity_map(labels) -> Dict[str, Set[str]]:
    return {
        x["task_id"]: infer_entity_group(x.get("entity", ""))
        for x in labels
    }
