from __future__ import annotations

import json
import queue
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional


class BufferedJSONLLogger:
    """
    异步缓冲 JSONL 日志器。
    目标：
    1. 兼容当前 TaskRunner / app.py 用法；
    2. 减少同步 open/write/close 带来的碎 I/O；
    3. 保持和 JSONLLogger 类似的接口风格。
    """

    def __init__(
        self,
        log_dir: str,
        flush_every: int = 100,
        flush_interval: float = 1.0,
        queue_maxsize: int = 10000,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.flush_every = flush_every
        self.flush_interval = flush_interval

        self._queue: "queue.Queue[Optional[tuple[str, Dict[str, Any]]]]" = queue.Queue(maxsize=queue_maxsize)
        self._buffers: Dict[str, list[str]] = {}
        self._files: Dict[str, Any] = {}
        self._task_run_counter = count(1)
        self._closed = False
        self._lock = threading.Lock()

        self._worker = threading.Thread(target=self._drain_loop, daemon=True)
        self._worker.start()

    def _json_line(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str) + "\n"

    def _enqueue(self, name: str, payload: Dict[str, Any]) -> None:
        # logger 关闭后静默丢弃尾部日志，避免二次异常覆盖真正的首发错误
        if self._closed:
            return
        self._queue.put((name, payload))

    def _get_fp(self, name: str):
        if name not in self._files:
            self._files[name] = open(self.log_dir / f"{name}.jsonl", "a", encoding="utf-8")
        return self._files[name]

    def _flush_name(self, name: str) -> None:
        lines = self._buffers.get(name)
        if not lines:
            return
        fp = self._get_fp(name)
        fp.writelines(lines)
        fp.flush()
        self._buffers[name] = []

    def _flush_all(self) -> None:
        for name in list(self._buffers.keys()):
            self._flush_name(name)

    def _drain_loop(self) -> None:
        last_flush_ts = time.time()

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                item = None

            if item is None:
                now = time.time()
                if now - last_flush_ts >= self.flush_interval:
                    self._flush_all()
                    last_flush_ts = now

                if self._closed and self._queue.empty():
                    break
                continue

            name, payload = item
            self._buffers.setdefault(name, [])
            self._buffers[name].append(self._json_line(payload))

            if len(self._buffers[name]) >= self.flush_every:
                self._flush_name(name)
                last_flush_ts = time.time()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._worker.join(timeout=10)
        self._flush_all()
        for fp in self._files.values():
            try:
                fp.close()
            except Exception:
                pass

    # ========= 兼容当前主链的接口 =========

    def log_task_run(
        self,
        experiment_id: str,
        task_id: str,
        task_order: int,
        query_text: str,
        task_start_time: str,
    ) -> int:
        with self._lock:
            task_run_id = next(self._task_run_counter)

        self._enqueue(
            "task_runs",
            {
                "task_run_id": task_run_id,
                "experiment_id": experiment_id,
                "task_id": task_id,
                "task_order": task_order,
                "query_text": query_text,
                "task_start_time": task_start_time,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        return task_run_id

    def update_task_result(
        self,
        task_run_id: int,
        final_answer: str,
        success_flag: int,
        reward_score: Optional[float],
        token_cost: Optional[float],
        latency_ms: Optional[int],
    ) -> None:
        self._enqueue(
            "task_results",
            {
                "task_run_id": task_run_id,
                "final_answer": final_answer,
                "success_flag": success_flag,
                "reward_score": reward_score,
                "token_cost": token_cost,
                "latency_ms": latency_ms,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        used_flag: int = 1,
    ) -> None:
        self._enqueue(
            "retrieval_logs",
            {
                "task_run_id": task_run_id,
                "query_text": query_text,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "rank_pos": rank_pos,
                "score": score,
                "used_flag": used_flag,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def log_memory(
        self,
        task_run_id: int,
        memory_key: str,
        operation_type: str,
        memory_content: str,
        relevance_score: Optional[float],
    ) -> None:
        self._enqueue(
            "memory_logs",
            {
                "task_run_id": task_run_id,
                "memory_key": memory_key,
                "operation_type": operation_type,
                "memory_content": memory_content,
                "relevance_score": relevance_score,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    ) -> None:
        self._enqueue(
            "trajectory_logs",
            {
                "task_run_id": task_run_id,
                "step_no": step_no,
                "agent_name": agent_name,
                "action_type": action_type,
                "action_input": action_input,
                "action_output": action_output,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def log_profile(
        self,
        task_run_id: int,
        task_id: str,
        task_order: int,
        stream_id: Optional[str],
        profile: Dict[str, Any],
    ) -> None:
        self._enqueue(
            "profile_logs",
            {
                "task_run_id": task_run_id,
                "task_id": task_id,
                "task_order": task_order,
                "stream_id": stream_id,
                "profile": profile,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
