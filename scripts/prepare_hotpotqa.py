import argparse

from src.benchmarks.common.io import dump_jsonl
from src.benchmarks.hotpotqa.adapter import load_hotpot_json
from src.benchmarks.hotpotqa.stream_builder import build_hotpot_streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--task-output", type=str, required=True)
    parser.add_argument("--support-output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = load_hotpot_json(args.input)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    tasks, units = build_hotpot_streams(samples)

    dump_jsonl(tasks, args.task_output)
    dump_jsonl(units, args.support_output)

    print(f"[HotpotQA] samples={len(samples)} tasks={len(tasks)} support_units={len(units)}")
    print(f"[HotpotQA] task_output={args.task_output}")
    print(f"[HotpotQA] support_output={args.support_output}")


if __name__ == "__main__":
    main()
