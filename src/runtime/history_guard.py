from datetime import datetime
from typing import Any


def parse_iso_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def is_legal_history_record(mem: dict, task_context: Any) -> bool:
    if mem.get("experiment_id") != task_context.experiment_id:
        return False

    mem_order = mem.get("task_order")
    if mem_order is None or mem_order >= task_context.task_order:
        return False

    mem_created_at = parse_iso_time(mem.get("created_at"))
    if mem_created_at is None or mem_created_at >= task_context.task_start_time:
        return False

    return True
