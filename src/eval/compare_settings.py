import argparse
import json
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=str, required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.summary_path))

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["setting_name"]] = r

    for task_id in sorted(by_task.keys()):
        task_rows = by_task[task_id]
        print(f"\n===== {task_id} =====")
        for setting_name, r in sorted(task_rows.items()):
            print({
                "setting": setting_name,
                "used_memory_count": r.get("used_memory_count"),
                "support_hit_count": r.get("support_hit_count"),
                "support_expected_count": r.get("support_expected_count"),
                "used_memory_task_ids": r.get("used_memory_task_ids"),
            })


if __name__ == "__main__":
    main()
