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
    parser.add_argument("--log-path", type=str, required=True)
    args = parser.parse_args()

    logs = load_jsonl(Path(args.log_path))
    if not logs:
        print("no logs found")
        return

    total = len(logs)
    avg_reward = sum(float(x.get("reward", 0.0)) for x in logs) / total
    avg_selected = sum(int(x.get("selected_count", 0)) for x in logs) / total
    write_rate = sum(1 for x in logs if x.get("memory_written")) / total

    print(f"total_logs={total}")
    print(f"avg_reward={avg_reward:.4f}")
    print(f"avg_selected_count={avg_selected:.4f}")
    print(f"memory_write_rate={write_rate:.4f}")


if __name__ == "__main__":
    main()
