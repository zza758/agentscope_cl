import json
from typing import Any, Dict, Optional

import mysql.connector


class MySQLLogger:
    def __init__(self, db_config: Dict[str, Any]):
        self.conn = mysql.connector.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            port=db_config.get("port", 3306),
            charset=db_config.get("charset", "utf8mb4"),
        )

    def close(self) -> None:
        if self.conn.is_connected():
            self.conn.close()

    def _execute(self, sql: str, params: tuple) -> int:
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        self.conn.commit()
        last_id = cursor.lastrowid
        cursor.close()
        return last_id

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
        sql = """
        INSERT INTO task_runs (
            experiment_id, task_id, task_order, query_text, task_start_time,
            final_answer, success_flag, reward_score, token_cost, latency_ms
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        return self._execute(
            sql,
            (
                experiment_id,
                task_id,
                task_order,
                query_text,
                task_start_time,
                final_answer,
                success_flag,
                reward_score,
                token_cost,
                latency_ms,
            ),
        )

    def update_task_result(
        self,
        task_run_id: int,
        final_answer: str,
        success_flag: int,
        reward_score: Optional[float],
        token_cost: Optional[int],
        latency_ms: Optional[int],
    ) -> None:
        sql = """
        UPDATE task_runs
        SET final_answer=%s, success_flag=%s, reward_score=%s, token_cost=%s, latency_ms=%s
        WHERE id=%s
        """
        cursor = self.conn.cursor()
        cursor.execute(
            sql,
            (final_answer, success_flag, reward_score, token_cost, latency_ms, task_run_id),
        )
        self.conn.commit()
        cursor.close()

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
        sql = """
        INSERT INTO retrieval_logs (task_run_id, query_text, doc_id, chunk_id, rank_pos, score, used_flag)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        return self._execute(
            sql,
            (task_run_id, query_text, doc_id, chunk_id, rank_pos, score, used_flag),
        )

    def log_memory(
        self,
        task_run_id: int,
        memory_key: str,
        operation_type: str,
        memory_content: str,
        relevance_score: Optional[float] = None,
    ) -> int:
        sql = """
        INSERT INTO memory_logs (task_run_id, memory_key, operation_type, memory_content, relevance_score)
        VALUES (%s, %s, %s, %s, %s)
        """
        return self._execute(
            sql,
            (task_run_id, memory_key, operation_type, memory_content, relevance_score),
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
        sql = """
        INSERT INTO trajectory_logs (task_run_id, step_no, agent_name, action_type, action_input, action_output)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        return self._execute(
            sql,
            (
                task_run_id,
                step_no,
                agent_name,
                action_type,
                json.dumps(action_input, ensure_ascii=False) if not isinstance(action_input, str) else action_input,
                json.dumps(action_output, ensure_ascii=False) if not isinstance(action_output, str) else action_output,
            ),
        )