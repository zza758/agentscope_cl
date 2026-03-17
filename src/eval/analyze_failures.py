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
    parser.add_argument("--setting", type=str, required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.summary_path))
    rows = [r for r in rows if r.get("setting_name") == args.setting]

    miss = [
        r for r in rows
        if r.get("support_hit_count", 0) < r.get("support_expected_count", 0)
    ]

    print(f"setting={args.setting}")
    print(f"total={len(rows)}")
    print(f"miss_count={len(miss)}")

    for r in miss:
        print("\n-----")
        print("task_id =", r.get("task_id"))
        print("query =", r.get("query"))
        print("used_memory_task_ids =", r.get("used_memory_task_ids"))
        print("support_expected_task_ids =", r.get("support_expected_task_ids"))
        print("support_hit_count =", r.get("support_hit_count"))
        print("support_expected_count =", r.get("support_expected_count"))


if __name__ == "__main__":
    main()
