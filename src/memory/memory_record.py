from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_retrieval_text(self) -> str:
        return self.memory_summary.strip()

    def to_log_text(self) -> str:
        return (
            f"任务: {self.query}\n"
            f"原始答案: {self.answer_raw}\n"
            f"记忆摘要: {self.memory_summary}\n"
            f"策略备注: {self.strategy_note}"
        ).strip()

    @staticmethod
    def now_ts() -> str:
        return datetime.now().isoformat(timespec="seconds")
