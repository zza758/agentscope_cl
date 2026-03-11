import json
import random
from pathlib import Path
from typing import List, Dict, Any


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


def build_memory_contrastive_samples(
    memory_records: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    基于 memory_bank.jsonl 构造最基础的对比学习样本。

    思路：
    - 每条 memory 记录都可作为某个 query 的 positive_memory
    - negative_memory 从其他 task 的 memory 中随机采样
    - 只构造“跨 task”负样本，避免把同一 task 的历史经验误当负样本
    """
    random.seed(seed)

    samples = []

    if len(memory_records) < 2:
        return samples

    for record in memory_records:
        task_id = record.get("task_id")
        query = record.get("query")
        positive_memory = record.get("content")

        if not task_id or not query or not positive_memory:
            continue

        candidate_negatives = [
            r for r in memory_records
            if r.get("task_id") != task_id and r.get("content")
        ]

        if not candidate_negatives:
            continue

        negative_record = random.choice(candidate_negatives)
        negative_memory = negative_record["content"]

        sample = {
            "task_id": task_id,
            "query": query,
            "positive_memory": positive_memory,
            "negative_memory": negative_memory,
            "source": "memory_bank",
            "negative_type": "random",
        }
        samples.append(sample)

    return samples


def build_meta(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_samples": len(samples),
        "fields": [
            "task_id",
            "query",
            "positive_memory",
            "negative_memory",
            "source",
            "negative_type",
        ],
        "description": "Memory contrastive dataset for continual learning reranking.",
    }


def main():
    project_root = Path(__file__).resolve().parents[2]

    memory_path = project_root / "data" / "memory" / "memory_bank.jsonl"
    output_path = project_root / "data" / "contrastive" / "memory_contrastive_samples.jsonl"
    meta_path = project_root / "data" / "contrastive" / "memory_contrastive_meta.json"

    memory_records = load_jsonl(memory_path)
    samples = build_memory_contrastive_samples(memory_records)

    save_jsonl(output_path, samples)

    meta = build_meta(samples)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已读取 memory 条数: {len(memory_records)}")
    print(f"已构造 contrastive 样本数: {len(samples)}")
    print(f"样本保存路径: {output_path}")
    print(f"元信息保存路径: {meta_path}")


if __name__ == "__main__":
    main()
