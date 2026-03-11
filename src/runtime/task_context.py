from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TaskContext:
    experiment_id: str
    task_id: str
    task_order: int
    task_run_id: Optional[int]
    task_start_time: datetime
