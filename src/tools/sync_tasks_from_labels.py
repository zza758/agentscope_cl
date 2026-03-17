import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_labels_to_tasks(labels):
    tasks = []
    for item in labels:
        tasks.append({
            "task_id": item["task_id"],
            "task_order": item["task_order"],
            "query": item["query"],
            "task_type": item.get("task_type"),
            "entity": item.get("entity"),
            "support_memory_task_ids": item.get("support_memory_task_ids", []),
        })
    tasks.sort(key=lambda x: x["task_order"])
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-file", type=str, required=True)
    parser.add_argument("--tasks-file", type=str, required=True)
    args = parser.parse_args()

    labels_path = Path(args.labels_file)
    tasks_path = Path(args.tasks_file)

    labels = load_jsonl(labels_path)
    tasks = convert_labels_to_tasks(labels)
    save_jsonl(tasks_path, tasks)

    print(f"已从 {labels_path} 同步生成 {len(tasks)} 条任务 -> {tasks_path}")


if __name__ == "__main__":
    main()
