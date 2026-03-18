from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    experiment_id: str
    task_id: str
    task_order: int
    query: str
    answer_raw: str
    memory_summary: str
    strategy_note: str
    created_at: str
    stream_id: Optional[str] = None
    task_type: Optional[str] = None
    entity: Optional[str] = None
    support_task_ids: List[str] = field(default_factory=list)
    source_dataset: Optional[str] = None
    source_sample_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_retrieval_text(self) -> str:
        return (self.memory_summary or self.answer_raw).strip()

    def to_log_text(self) -> str:
        parts = [
            f"任务: {self.query}",
            f"原始答案: {self.answer_raw}",
            f"记忆摘要: {self.memory_summary}",
            f"策略备注: {self.strategy_note}",
        ]
        if self.stream_id:
            parts.append(f"stream_id: {self.stream_id}")
        if self.task_type:
            parts.append(f"task_type: {self.task_type}")
        if self.entity:
            parts.append(f"entity: {self.entity}")
        if self.support_task_ids:
            parts.append(f"support_task_ids: {self.support_task_ids}")
        return "\n".join(parts).strip()

    @staticmethod
    def now_ts() -> str:
        return datetime.now().isoformat(timespec="seconds")
