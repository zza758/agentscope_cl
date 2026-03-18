from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _to_task_dict(task: Any) -> Dict[str, Any]:
    if hasattr(task, "to_dict"):
        return task.to_dict()
    if isinstance(task, dict):
        return dict(task)
    raise TypeError(f"Unsupported task type: {type(task)}")


@dataclass
class BenchmarkTaskCache:
    tasks: List[Dict[str, Any]]
    task_by_id: Dict[str, Dict[str, Any]]
    stream_to_tasks: "OrderedDict[str, List[Dict[str, Any]]]"
    stream_to_task_ids: Dict[str, List[str]]
    task_to_support_ids: Dict[str, List[str]]

    @classmethod
    def from_tasks(cls, tasks: Iterable[Any]) -> "BenchmarkTaskCache":
        normalized_tasks: List[Dict[str, Any]] = []
        for task in tasks:
            t = _to_task_dict(task)
            t.setdefault("stream_id", "__default__")
            t.setdefault("support_task_ids", [])
            t.setdefault("meta", {})
            normalized_tasks.append(t)

        normalized_tasks.sort(
            key=lambda x: (
                str(x.get("stream_id", "__default__")),
                int(x.get("task_order", 0)),
                str(x.get("task_id", "")),
            )
        )

        task_by_id: Dict[str, Dict[str, Any]] = {}
        stream_to_tasks: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        stream_to_task_ids: Dict[str, List[str]] = {}
        task_to_support_ids: Dict[str, List[str]] = {}

        for task in normalized_tasks:
            task_id = task["task_id"]
            stream_id = task.get("stream_id") or "__default__"

            task_by_id[task_id] = task
            stream_to_tasks.setdefault(stream_id, []).append(task)
            task_to_support_ids[task_id] = list(task.get("support_task_ids", []) or [])

        for stream_id, items in stream_to_tasks.items():
            items.sort(key=lambda x: (int(x.get("task_order", 0)), str(x.get("task_id", ""))))
            stream_to_task_ids[stream_id] = [x["task_id"] for x in items]

        return cls(
            tasks=normalized_tasks,
            task_by_id=task_by_id,
            stream_to_tasks=stream_to_tasks,
            stream_to_task_ids=stream_to_task_ids,
            task_to_support_ids=task_to_support_ids,
        )

    @property
    def stream_ids(self) -> List[str]:
        return list(self.stream_to_tasks.keys())

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_streams(self) -> int:
        return len(self.stream_to_tasks)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.task_by_id.get(task_id)

    def get_stream_tasks(self, stream_id: str) -> List[Dict[str, Any]]:
        return list(self.stream_to_tasks.get(stream_id, []))

    def iter_stream_items(self):
        for stream_id, items in self.stream_to_tasks.items():
            yield stream_id, items

    def flatten(self) -> List[Dict[str, Any]]:
        return list(self.tasks)
