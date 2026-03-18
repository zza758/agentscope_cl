from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SupportUnit:
    unit_id: str
    source_dataset: str
    source_sample_id: str

    title: Optional[str] = None
    sent_id: Optional[int] = None
    text: Optional[str] = None

    entity: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedTask:
    task_id: str
    stream_id: str
    task_order: int

    query: str
    answer: Optional[str] = None

    task_type: Optional[str] = None
    entity: Optional[str] = None

    source_dataset: Optional[str] = None
    source_sample_id: Optional[str] = None

    gold_support_units: List[str] = field(default_factory=list)
    support_task_ids: List[str] = field(default_factory=list)

    history_ref: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
