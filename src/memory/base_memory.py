from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.runtime.task_context import TaskContext
from src.memory.memory_record import MemoryRecord

class BaseMemoryManager(ABC):
    @abstractmethod
    def retrieve_memory(
        self,
        query: str,
        task_context: TaskContext,
        top_k: Optional[int] = None,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_memory_with_scores(
        self,
        query: str,
        task_context: TaskContext,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def write_memory(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    def format_memories(self, memories: List[str]) -> str:
        if not memories:
            return "暂无历史经验。"
        return "\n".join(
            [f"[历史经验{idx}] {mem}" for idx, mem in enumerate(memories, start=1)]
        )
