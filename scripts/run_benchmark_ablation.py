#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_MODULE = "src.app"
ANALYZE_SCRIPT = PROJECT_ROOT / "scripts" / "analyze_benchmark_run.py"


def run_and_stream(cmd: List[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()
        return process.returncode


def build_rounds() -> List[Dict]:
    return [
        {
            "name": "retrieval_only",
            "extra_args": [
                "--policy-backend", "none",
                "--disable-contrastive-rerank",
            ],
        },
        {
            "name": "contrastive",
            "extra_args": [
                "--policy-backend", "none",
                "--use-contrastive-rerank",
            ],
        },
        {
            "name": "rule_policy",
            "extra_args": [
                "--policy-backend", "rule",
                "--disable-contrastive-rerank",
            ],
        },
        {
            "name": "rl_policy",
            "extra_args": [
                "--policy-backend", "rl",
                "--disable-contrastive-rerank",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["musique", "hotpotqa"], required=True)
    parser.add_argument("--task-file", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-prefix", type=str, required=True)
    parser.add_argument("--log-root", type=str, default="outputs/benchmark_logs")
    parser.add_argument("--report-root", type=str, default="outputs/benchmark_reports")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-limit", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-concurrent-streams", type=int, default=None)
    args = parser.parse_args()

    # 数据集默认稳定配置
    if args.dataset == "musique":
        stream_limit = args.stream_limit if args.stream_limit is not None else 5
        max_tasks = args.max_tasks
        max_concurrent_streams = args.max_concurrent_streams if args.max_concurrent_streams is not None else 4
    else:
        stream_limit = args.stream_limit
        max_tasks = args.max_tasks if args.max_tasks is not None else 20
        max_concurrent_streams = args.max_concurrent_streams if args.max_concurrent_streams is not None else 1

    rounds = build_rounds()
    report_root = PROJECT_ROOT / args.report_root
    report_root.mkdir(parents=True, exist_ok=True)

    combined_summary = {
        "dataset": args.dataset,
        "task_file": args.task_file,
        "experiment_prefix": args.experiment_prefix,
        "rounds": [],
    }

    for round_cfg in rounds:
        round_name = round_cfg["name"]
        experiment_id = f"{args.experiment_prefix}_{round_name}"
        exp_dir = PROJECT_ROOT / args.log_root / experiment_id
        run_log = report_root / f"{experiment_id}.log"

        cmd = [
            sys.executable,
            "-m",
            APP_MODULE,
            "--mode", "benchmark",
            "--dataset", args.dataset,
            "--task-file", args.task_file,
            "--experiment-id", experiment_id,
            "--logger-backend", "buffered_jsonl",
            "--disable-retrieval-logging",
            "--profile",
            "--max-concurrent-streams", str(max_concurrent_streams),
        ]

        if args.config:
            cmd += ["--config", args.config]

        if stream_limit is not None:
            cmd += ["--stream-limit", str(stream_limit)]

        if max_tasks is not None:
            cmd += ["--max-tasks", str(max_tasks)]

        cmd += round_cfg["extra_args"]

        print("=" * 100)
        print(f"[Round] {round_name}")
        print(" ".join(cmd))

        if args.skip_existing and exp_dir.exists():
            print(f"[skip] existing experiment dir: {exp_dir}")
        else:
            if args.dry_run:
                continue

            rc = run_and_stream(cmd, run_log)
            if rc != 0:
                print(f"[round failed] {round_name} rc={rc}")
                if args.stop_on_fail:
                    break

        analyze_cmd = [
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--experiment-id", experiment_id,
            "--log-root", args.log_root,
            "--save-json",
        ]
        if not args.dry_run:
            analyze_rc = subprocess.call(analyze_cmd, cwd=PROJECT_ROOT)
            if analyze_rc != 0:
                print(f"[analyze failed] {round_name} rc={analyze_rc}")

        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            combined_summary["rounds"].append(
                {
                    "round": round_name,
                    "experiment_id": experiment_id,
                    "summary": summary,
                }
            )

    combined_file = report_root / f"{args.experiment_prefix}_combined_summary.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {combined_file}")


if __name__ == "__main__":
    main()
