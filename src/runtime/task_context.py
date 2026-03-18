from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TaskContext:
    experiment_id: str
    task_id: str
    task_order: int
    task_run_id: Optional[int]
    task_start_time: datetime
    stream_id: Optional[str] = None
    task_type: Optional[str] = None
    task_entity: Optional[str] = None
    support_task_ids: List[str] = field(default_factory=list)
    source_dataset: Optional[str] = None
    source_sample_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
