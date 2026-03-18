import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.benchmarks.common.io import dump_jsonl
from src.benchmarks.musique.adapter import load_musique
from src.benchmarks.musique.stream_builder import build_musique_streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--task-output", type=str, required=True)
    parser.add_argument("--support-output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = load_musique(args.input)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    tasks, units = build_musique_streams(samples)

    dump_jsonl(tasks, args.task_output)
    dump_jsonl(units, args.support_output)

    print(f"[MuSiQue] samples={len(samples)} tasks={len(tasks)} support_units={len(units)}")
    print(f"[MuSiQue] task_output={args.task_output}")
    print(f"[MuSiQue] support_output={args.support_output}")


if __name__ == "__main__":
    main()
