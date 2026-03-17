import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_label_map(labels):
    return {x["task_id"]: x for x in labels}


def normalize_record(record: Dict[str, Any], label_map: Dict[str, Any]) -> Dict[str, Any]:
    task_id = record.get("task_id")
    label = label_map.get(task_id, {})

    # 补齐核心字段
    if "memory_summary" not in record or not record.get("memory_summary"):
        record["memory_summary"] = (
            record.get("content")
            or record.get("answer")
            or record.get("answer_raw")
            or ""
        )

    if "answer_raw" not in record or not record.get("answer_raw"):
        record["answer_raw"] = (
            record.get("answer")
            or record.get("content")
            or record.get("memory_summary")
            or ""
        )

    if "strategy_note" not in record or not record.get("strategy_note"):
        record["strategy_note"] = record.get(
            "experience",
            "可作为后续相似任务的参考经验。",
        )

    if "content" not in record or not record.get("content"):
        record["content"] = record.get("memory_summary", "")

    # 用 labels 补 task metadata
    if "task_type" not in record or record.get("task_type") is None:
        record["task_type"] = label.get("task_type")

    if "entity" not in record or record.get("entity") is None:
        record["entity"] = label.get("entity")

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-path", type=str, required=True)
    parser.add_argument("--labels-file", type=str, required=True)
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    memory_path = Path(args.memory_path)
    labels_file = Path(args.labels_file)

    if not memory_path.exists():
        raise FileNotFoundError(f"memory file not found: {memory_path}")
    if not labels_file.exists():
        raise FileNotFoundError(f"labels file not found: {labels_file}")

    rows = load_jsonl(memory_path)
    labels = load_jsonl(labels_file)
    label_map = build_label_map(labels)

    if args.backup:
        backup_path = memory_path.with_suffix(memory_path.suffix + ".bak")
        shutil.copy2(memory_path, backup_path)
        print(f"backup saved to: {backup_path}")

    normalized = [normalize_record(dict(r), label_map) for r in rows]

    with open(memory_path, "w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"migrated_records={len(normalized)}")
    print(f"memory_path={memory_path}")


if __name__ == "__main__":
    main()
