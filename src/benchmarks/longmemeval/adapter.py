from typing import Dict, List, Any

from src.benchmarks.common.schema import UnifiedTask
from src.benchmarks.common.io import load_json_or_jsonl


def load_lme(path: str) -> List[Dict]:
    return load_json_or_jsonl(path)


def get_sample_id(sample: Dict, idx: int) -> str:
    return str(sample.get("id", sample.get("sample_id", idx)))


def get_sessions(sample: Dict) -> List[Dict]:
    # 兼容不同字段名
    for key in ["sessions", "history", "conversation", "dialogue"]:
        val = sample.get(key)
        if isinstance(val, list):
            return val
    return []


def get_question(sample: Dict) -> str:
    return sample.get("question") or sample.get("query") or sample.get("final_question") or ""


def get_answer(sample: Dict) -> str:
    return sample.get("answer") or sample.get("gold_answer") or ""


def infer_task_type(sample: Dict) -> str:
    return (
        sample.get("question_type")
        or sample.get("ability_type")
        or sample.get("task_type")
        or "long_memory"
    )


def session_to_text(sess: Any) -> str:
    if isinstance(sess, str):
        return sess
    if isinstance(sess, dict):
        # 优先取 text/content
        return (
            sess.get("text")
            or sess.get("content")
            or sess.get("utterance")
            or str(sess)
        )
    return str(sess)


def session_to_timestamp(sess: Any):
    if isinstance(sess, dict):
        return sess.get("timestamp") or sess.get("time") or sess.get("date")
    return None


def session_to_entity(sess: Any):
    if isinstance(sess, dict):
        return sess.get("entity") or sess.get("topic") or ""
    return ""


def convert_lme_sample(sample: Dict, idx: int) -> List[UnifiedTask]:
    sample_id = get_sample_id(sample, idx)
    stream_id = f"lme::{sample_id}"
    tasks: List[UnifiedTask] = []

    sessions = get_sessions(sample)
    final_question = get_question(sample)
    final_answer = get_answer(sample)
    final_type = infer_task_type(sample)

    # 1. 历史写入任务
    for i, sess in enumerate(sessions, start=1):
        tasks.append(
            UnifiedTask(
                task_id=f"{stream_id}::hist::{i}",
                stream_id=stream_id,
                task_order=i,
                query=session_to_text(sess),
                answer=None,
                task_type="history_write",
                entity=session_to_entity(sess),
                source_dataset="longmemeval",
                source_sample_id=sample_id,
                gold_support_units=[],
                support_task_ids=[],
                history_ref=[],
                meta={
                    "timestamp": session_to_timestamp(sess),
                    "session_raw": sess,
                },
            )
        )

    # 2. 最终问题任务
    hist_ids = [t.task_id for t in tasks]
    tasks.append(
        UnifiedTask(
            task_id=f"{stream_id}::final",
            stream_id=stream_id,
            task_order=len(tasks) + 1,
            query=final_question,
            answer=final_answer,
            task_type=final_type,
            entity=sample.get("entity", ""),
            source_dataset="longmemeval",
            source_sample_id=sample_id,
            gold_support_units=[],
            support_task_ids=hist_ids,
            history_ref=hist_ids,
            meta={"raw_sample": sample},
        )
    )

    return tasks
