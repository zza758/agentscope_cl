from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List


@dataclass
class MemoryRecord:
    experiment_id: str
    task_id: str
    task_order: int
    query: str

    answer_raw: str
    memory_summary: str
    strategy_note: str

    task_type: Optional[str] = None
    entity: Optional[str] = None
    memory_type: str = "fact"

    support_task_ids: List[str] = field(default_factory=list)

    created_at: str = ""
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None

    confidence: Optional[float] = None
    reuse_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_retrieval_text(self) -> str:
        return (self.memory_summary or self.answer_raw).strip()

    def to_log_text(self) -> str:
        return (
            f"任务: {self.query}\n"
            f"任务类型: {self.task_type}\n"
            f"实体: {self.entity}\n"
            f"原始答案: {self.answer_raw}\n"
            f"记忆摘要: {self.memory_summary}\n"
            f"策略备注: {self.strategy_note}"
        ).strip()

    @staticmethod
    def now_ts() -> str:
        return datetime.now().isoformat(timespec="seconds")
