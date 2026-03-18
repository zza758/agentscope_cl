from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.benchmarks.common.io import load_json_or_jsonl


@dataclass
class LoadedBenchmarkTask:
    task_id: str
    stream_id: str
    task_order: int
    query: str
    answer: Optional[str] = None
    task_type: Optional[str] = None
    entity: Optional[str] = None
    support_task_ids: List[str] = field(default_factory=list)
    gold_support_units: List[str] = field(default_factory=list)
    source_dataset: Optional[str] = None
    source_sample_id: Optional[str] = None
    history_ref: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_row(row: Dict[str, Any], dataset: Optional[str] = None) -> LoadedBenchmarkTask:
    stream_id = row.get("stream_id") or row.get("task_id") or "default_stream"
    source_dataset = row.get("source_dataset") or dataset
    return LoadedBenchmarkTask(
        task_id=str(row["task_id"]),
        stream_id=str(stream_id),
        task_order=int(row.get("task_order", 0)),
        query=str(row.get("query", "")),
        answer=row.get("answer"),
        task_type=row.get("task_type"),
        entity=row.get("entity"),
        support_task_ids=[str(x) for x in _ensure_list(row.get("support_task_ids"))],
        gold_support_units=[str(x) for x in _ensure_list(row.get("gold_support_units"))],
        source_dataset=source_dataset,
        source_sample_id=row.get("source_sample_id"),
        history_ref=[str(x) for x in _ensure_list(row.get("history_ref"))],
        meta=row.get("meta") or {},
    )


def load_benchmark_tasks(task_file: str, dataset: Optional[str] = None) -> List[LoadedBenchmarkTask]:
    rows = load_json_or_jsonl(task_file)
    tasks = [_normalize_row(row, dataset=dataset) for row in rows]

    stream_rank: "OrderedDict[str, int]" = OrderedDict()
    for task in tasks:
        if task.stream_id not in stream_rank:
            stream_rank[task.stream_id] = len(stream_rank)

    tasks.sort(key=lambda x: (stream_rank[x.stream_id], x.task_order, x.task_id))
    return tasks


def group_tasks_by_stream(tasks: List[LoadedBenchmarkTask]) -> "OrderedDict[str, List[LoadedBenchmarkTask]]":
    groups: "OrderedDict[str, List[LoadedBenchmarkTask]]" = OrderedDict()
    for task in tasks:
        groups.setdefault(task.stream_id, []).append(task)
    return groups


def select_benchmark_tasks(
    tasks: List[LoadedBenchmarkTask],
    task_ids: Optional[List[str]] = None,
    stream_ids: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    stream_limit: Optional[int] = None,
) -> List[LoadedBenchmarkTask]:
    selected = list(tasks)

    if stream_ids:
        stream_allow = {x.strip() for x in stream_ids if x and x.strip()}
        selected = [task for task in selected if task.stream_id in stream_allow]

    if stream_limit is not None:
        groups = group_tasks_by_stream(selected)
        keep_streams = list(groups.keys())[:stream_limit]
        keep_set = set(keep_streams)
        selected = [task for task in selected if task.stream_id in keep_set]

    if task_ids:
        task_allow = {x.strip() for x in task_ids if x and x.strip()}
        selected = [task for task in selected if task.task_id in task_allow]

    if max_tasks is not None:
        selected = selected[:max_tasks]

    return selected
