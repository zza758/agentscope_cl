import argparse

from src.benchmarks.common.io import dump_jsonl
from src.benchmarks.longmemeval.adapter import load_lme
from src.benchmarks.longmemeval.stream_builder import build_longmemeval_streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--task-output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = load_lme(args.input)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    tasks = build_longmemeval_streams(samples)
    dump_jsonl(tasks, args.task_output)

    print(f"[LongMemEval] samples={len(samples)} tasks={len(tasks)}")
    print(f"[LongMemEval] task_output={args.task_output}")


if __name__ == "__main__":
    main()
