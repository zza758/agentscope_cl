import json
from pathlib import Path
from typing import Any, Dict, Optional


class JSONLLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._task_run_seq = 0

    def close(self) -> None:
        return None

    def _append(self, filename: str, row: Dict[str, Any]) -> int:
        path = self.output_dir / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return 1

    def log_task_run(
        self,
        experiment_id: str,
        task_id: str,
        task_order: int,
        query_text: str,
        task_start_time: str,
        final_answer: Optional[str] = None,
        success_flag: int = 0,
        reward_score: Optional[float] = None,
        token_cost: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> int:
        self._task_run_seq += 1
        task_run_id = self._task_run_seq
        self._append(
            "task_runs.jsonl",
            {
                "id": task_run_id,
                "experiment_id": experiment_id,
                "task_id": task_id,
                "task_order": task_order,
                "query_text": query_text,
                "task_start_time": task_start_time,
                "final_answer": final_answer,
                "success_flag": success_flag,
                "reward_score": reward_score,
                "token_cost": token_cost,
                "latency_ms": latency_ms,
            },
        )
        return task_run_id

    def update_task_result(
        self,
        task_run_id: int,
        final_answer: str,
        success_flag: int,
        reward_score: Optional[float],
        token_cost: Optional[int],
        latency_ms: Optional[int],
    ) -> None:
        self._append(
            "task_results.jsonl",
            {
                "task_run_id": task_run_id,
                "final_answer": final_answer,
                "success_flag": success_flag,
                "reward_score": reward_score,
                "token_cost": token_cost,
                "latency_ms": latency_ms,
            },
        )

    def log_retrieval(
        self,
        task_run_id: int,
        query_text: str,
        doc_id: str,
        chunk_id: Optional[str],
        rank_pos: int,
        score: Optional[float],
        used_flag: int = 0,
    ) -> int:
        return self._append(
            "retrieval_logs.jsonl",
            {
                "task_run_id": task_run_id,
                "query_text": query_text,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "rank_pos": rank_pos,
                "score": score,
                "used_flag": used_flag,
            },
        )

    def log_memory(
        self,
        task_run_id: int,
        memory_key: str,
        operation_type: str,
        memory_content: str,
        relevance_score: Optional[float] = None,
    ) -> int:
        return self._append(
            "memory_logs.jsonl",
            {
                "task_run_id": task_run_id,
                "memory_key": memory_key,
                "operation_type": operation_type,
                "memory_content": memory_content,
                "relevance_score": relevance_score,
            },
        )

    def log_trajectory(
        self,
        task_run_id: int,
        step_no: int,
        agent_name: str,
        action_type: str,
        action_input: Any,
        action_output: Any,
    ) -> int:
        return self._append(
            "trajectory_logs.jsonl",
            {
                "task_run_id": task_run_id,
                "step_no": step_no,
                "agent_name": agent_name,
                "action_type": action_type,
                "action_input": action_input,
                "action_output": action_output,
            },
        )
