import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import math
import sys

from src.memory.embedder import DashScopeEmbedder
from src.utils.config_loader import load_config, PROJECT_ROOT


def normalize_summary_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split()).lower()


def deduplicate_memory_candidates(
        candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    按 memory_summary 文本去重。
    保留规则：
    1. task_order 更小者优先（更早历史）
    """
    dedup_map = {}

    for item in candidates:
        summary = item.get("memory_summary", "").strip()
        norm_summary = normalize_summary_text(summary)
        if not norm_summary:
            continue

        if norm_summary not in dedup_map:
            dedup_map[norm_summary] = item
            continue

        old_item = dedup_map[norm_summary]
        old_order = old_item.get("task_order", 10 ** 9)
        new_order = item.get("task_order", 10 ** 9)

        if new_order < old_order:
            dedup_map[norm_summary] = item

    return list(dedup_map.values())


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_memory_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容旧格式 memory：
    - 优先使用 memory_summary
    - 若不存在则回退到 content / answer
    """
    if "memory_summary" not in record:
        record["memory_summary"] = record.get("content") or record.get("answer") or ""

    if "answer_raw" not in record:
        record["answer_raw"] = record.get("answer", record.get("content", ""))

    if "strategy_note" not in record:
        record["strategy_note"] = record.get(
            "experience",
            "可作为后续相似任务的参考经验。",
        )

    if "content" not in record:
        record["content"] = record.get("memory_summary", "")

    return record


def build_memory_index(memory_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    以 task_id 建索引，默认同一 task_id 对应一条 memory。
    """
    index = {}
    for record in memory_records:
        record = normalize_memory_record(record)
        task_id = record.get("task_id")
        if task_id:
            index[task_id] = record
    return index


def is_legal_history_memory(
        memory_record: Dict[str, Any],
        query_task: Dict[str, Any],
) -> bool:
    """
    contrastive 构造也必须遵守合法历史原则：
    - 同实验
    - memory.task_order < query.task_order
    """
    if memory_record.get("experiment_id") != query_task.get("experiment_id"):
        return False

    mem_order = memory_record.get("task_order")
    query_order = query_task.get("task_order")

    if mem_order is None or query_order is None:
        return False

    return mem_order < query_order


def select_hard_negative_candidate(
    query: str,
    legal_candidates: List[Dict[str, Any]],
    positive_task_ids: set,
    positive_summaries: set,
    embedder,
) -> Optional[Dict[str, Any]]:
    """
    从合法历史候选中挑选 hard negative：
    - 非正样本
    - 与 query 的相似度高
    """
    negatives = []
    for r in legal_candidates:
        task_id = r.get("task_id")
        summary = r.get("memory_summary", "").strip()
        norm_summary = normalize_summary_text(summary)

        if not summary:
            continue
        if task_id in positive_task_ids:
            continue
        if norm_summary in positive_summaries:
            continue

        negatives.append(r)

    if not negatives:
        return None

    query_vec = embedder.embed_query(query)

    scored_negatives = []
    for item in negatives:
        memory_text = item.get("memory_summary", "").strip()
        memory_vec = embedder.embed_query(memory_text)
        score = cosine_similarity(query_vec, memory_vec)

        scored_negatives.append((score, item))

    scored_negatives.sort(key=lambda x: x[0], reverse=True)

    return scored_negatives[0][1] if scored_negatives else None


def build_memory_contrastive_samples(
        memory_records: List[Dict[str, Any]],
        task_labels: List[Dict[str, Any]],
        embedder,
        seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    基于结构化 memory + 任务标注文件构造 contrastive 样本。

    正样本：
    - 来自 task_labels 里 support_memory_task_ids 指定的历史 task

    负样本：
    - 来自同实验、当前任务之前、但不属于正样本的合法历史 memory

    训练文本：
    - 统一使用 memory_summary
    """
    seed_rng = random.Random(seed)
    samples = []

    if not memory_records or not task_labels:
        return samples

    memory_records = [normalize_memory_record(r) for r in memory_records]
    memory_index = build_memory_index(memory_records)

    for task in task_labels:
        query = task.get("query", "").strip()
        task_id = task.get("task_id")
        support_ids = task.get("support_memory_task_ids", [])

        if not query or not task_id:
            continue

        positive_task_ids = set(support_ids)

        positive_summaries = set()
        for pos_task_id in support_ids:
            pos_record_tmp = memory_index.get(pos_task_id)
            if not pos_record_tmp:
                continue
            pos_summary_tmp = pos_record_tmp.get("memory_summary", "").strip()
            if pos_summary_tmp:
                positive_summaries.add(normalize_summary_text(pos_summary_tmp))

        # 当前任务开始前、同实验内的合法历史候选
        legal_candidates = [
            r for r in memory_records
            if is_legal_history_memory(r, task)
        ]
        legal_candidates = deduplicate_memory_candidates(legal_candidates)

        if not legal_candidates:
            continue

        positive_task_ids = set(support_ids)

        # 对当前任务的每个 positive 都构造一个三元组
        for pos_task_id in support_ids:
            pos_record = memory_index.get(pos_task_id)
            if not pos_record:
                continue

            if not is_legal_history_memory(pos_record, task):
                # 防止标注写错，把未来 memory 或跨实验 memory 当成正样本
                continue

            positive_memory_summary = pos_record.get("memory_summary", "").strip()
            if not positive_memory_summary:
                continue

            neg_record = select_hard_negative_candidate(
                query=query,
                legal_candidates=legal_candidates,
                positive_task_ids=positive_task_ids,
                positive_summaries=positive_summaries,
                embedder=embedder,
            )

            if not neg_record:
                continue

            negative_memory_summary = neg_record.get("memory_summary", "").strip()
            if not negative_memory_summary:
                continue

            sample = {
                "experiment_id": task.get("experiment_id"),
                "query_task_id": task_id,
                "query_task_order": task.get("task_order"),
                "query": query,
                "positive_task_id": pos_task_id,
                "positive_memory_summary": positive_memory_summary,
                "negative_task_id": neg_record.get("task_id"),
                "negative_memory_summary": negative_memory_summary,
                "source": "structured_memory",
                "positive_source": "task_label_support_memory",
                "negative_type": "hard_legal_history",
                "candidate_pool_size": len(legal_candidates),
            }
            samples.append(sample)

    return samples


def build_meta(
        samples: List[Dict[str, Any]],
        memory_count: int,
        task_label_count: int,
) -> Dict[str, Any]:
    return {
        "num_samples": len(samples),
        "num_memory_records": memory_count,
        "num_task_labels": task_label_count,
        "fields": [
            "experiment_id",
            "query_task_id",
            "query_task_order",
            "query",
            "positive_task_id",
            "positive_memory_summary",
            "negative_task_id",
            "negative_memory_summary",
            "source",
            "positive_source",
            "negative_type",
            "candidate_pool_size",
        ],
        "description": (
            "Contrastive dataset built from structured memory_summary "
            "under legal-history constraints."
        ),
    }


def main():
    project_root = Path(__file__).resolve().parents[2]

    memory_path = project_root / "data" / "memory" / "memory_bank.jsonl"
    task_label_path = project_root / "data" / "tasks" / "contrastive_task_labels.jsonl"

    output_path = project_root / "data" / "contrastive" / "memory_contrastive_samples.jsonl"
    meta_path = project_root / "data" / "contrastive" / "memory_contrastive_meta.json"

    memory_records = load_jsonl(memory_path)
    task_labels = load_jsonl(task_label_path)

    config = load_config()
    embedding_cfg = config.get("embedding", {})
    model_cfg = config.get("model", {})

    embedder = DashScopeEmbedder(
        api_key=model_cfg["dashscope_api_key"],
        model_name=embedding_cfg.get("model_name", "text-embedding-v4"),
        normalize=embedding_cfg.get("normalize", True),
    )

    samples = build_memory_contrastive_samples(
        memory_records=memory_records,
        task_labels=task_labels,
        embedder=embedder,
        seed=42,
    )

    save_jsonl(output_path, samples)

    meta = build_meta(
        samples=samples,
        memory_count=len(memory_records),
        task_label_count=len(task_labels),
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已读取 memory 条数: {len(memory_records)}")
    print(f"已读取任务标注条数: {len(task_labels)}")
    print(f"已构造 contrastive 样本数: {len(samples)}")
    print(f"样本保存路径: {output_path}")
    print(f"元信息保存路径: {meta_path}")


if __name__ == "__main__":
    main()
