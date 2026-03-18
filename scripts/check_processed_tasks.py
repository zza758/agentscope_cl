import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    count = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            print("=" * 80)
            print(f"task_id={row.get('task_id')}")
            print(f"stream_id={row.get('stream_id')}")
            print(f"task_order={row.get('task_order')}")
            print(f"task_type={row.get('task_type')}")
            print(f"entity={row.get('entity')}")
            print(f"query={row.get('query')[:200]}")
            print(f"support_task_ids={row.get('support_task_ids')}")
            print(f"gold_support_units={row.get('gold_support_units')}")
            count += 1
            if count >= args.limit:
                break


if __name__ == "__main__":
    main()
