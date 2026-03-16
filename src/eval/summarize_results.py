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
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    rows = []
    for file in sorted(input_dir.glob("*.jsonl")):
        records = load_jsonl(file)
        for r in records:
            rows.append({
                "task_id": r.get("task_id"),
                "setting_name": r.get("setting_name"),
                "query": r.get("query"),
                "final_answer": r.get("final_answer"),
                "memory_summary": r.get("memory_summary"),
                "strategy_note": r.get("strategy_note"),
                "used_memory_count": len(r.get("used_memories", [])),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已汇总 {len(rows)} 条结果到: {output_path}")


if __name__ == "__main__":
    main()
