#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List


UNKNOWN_PATTERNS = [
    "未知",
    "未提供",
    "无法确定",
    "不清楚",
    "需进一步查询",
    "信息不足",
    "unknown",
    "not provided",
    "cannot determine",
    "unclear",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * p))
    return float(arr[idx])


def summarize_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "p95": round(percentile(values, 0.95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def contains_unknown(text: str) -> bool:
    text = (text or "").lower()
    return any(p.lower() in text for p in UNKNOWN_PATTERNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--log-root", type=str, default="outputs/benchmark_logs")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.log_root) / args.experiment_id
    task_runs_file = exp_dir / "task_runs.jsonl"
    task_results_file = exp_dir / "task_results.jsonl"
    profile_file = exp_dir / "profile_logs.jsonl"
    memory_logs_file = exp_dir / "memory_logs.jsonl"

    task_runs = load_jsonl(task_runs_file)
    task_results = load_jsonl(task_results_file)
    profile_logs = load_jsonl(profile_file)
    memory_logs = load_jsonl(memory_logs_file)

    run_map = {x["task_run_id"]: x for x in task_runs}
    result_map = {x["task_run_id"]: x for x in task_results}

    missing_result_runs = []
    for task_run_id, run in run_map.items():
        if task_run_id not in result_map:
            missing_result_runs.append(
                {
                    "task_run_id": task_run_id,
                    "task_id": run.get("task_id"),
                    "task_order": run.get("task_order"),
                    "query_text": run.get("query_text"),
                }
            )

    retrieve_ms = []
    compose_ms = []
    agent_ms = []
    memory_write_ms = []
    total_ms = []
    raw_support_candidates = []
    filtered_support_candidates = []

    for row in profile_logs:
        p = row.get("profile", {})
        retrieve_ms.append(float(p.get("retrieve_ms", 0.0)))
        compose_ms.append(float(p.get("compose_ms", 0.0)))
        agent_ms.append(float(p.get("agent_ms", 0.0)))
        memory_write_ms.append(float(p.get("memory_write_ms", 0.0)))
        total_ms.append(float(p.get("total_ms", 0.0)))
        raw_support_candidates.append(float(p.get("raw_support_candidates", 0.0)))
        filtered_support_candidates.append(float(p.get("filtered_support_candidates", 0.0)))

    task_rows = []
    for result in task_results:
        task_run_id = result["task_run_id"]
        run = run_map.get(task_run_id, {})
        task_id = run.get("task_id", "")
        final_answer = result.get("final_answer", "")
        task_rows.append(
            {
                "task_run_id": task_run_id,
                "task_id": task_id,
                "task_order": run.get("task_order"),
                "query_text": run.get("query_text"),
                "latency_ms": result.get("latency_ms"),
                "final_answer": final_answer,
                "is_final_task": "::final" in task_id,
                "contains_unknown": contains_unknown(final_answer),
            }
        )

    has_explicit_final_tasks = any(x["is_final_task"] for x in task_rows)
    if has_explicit_final_tasks:
        eval_rows = [x for x in task_rows if x["is_final_task"]]
    else:
        eval_rows = list(task_rows)

    unknown_count = sum(1 for x in eval_rows if x["contains_unknown"])
    unknown_ratio = round(unknown_count / len(eval_rows), 4) if eval_rows else 0.0

    support_retrieve_count = 0
    support_retrieve_raw_count = 0
    support_retrieve_task_runs = set()
    support_retrieve_raw_task_runs = set()

    for row in memory_logs:
        op = row.get("operation_type")
        if op == "support_retrieve":
            support_retrieve_count += 1
            support_retrieve_task_runs.add(row.get("task_run_id"))
        elif op == "support_retrieve_raw":
            support_retrieve_raw_count += 1
            support_retrieve_raw_task_runs.add(row.get("task_run_id"))

    summary = {
        "experiment_id": args.experiment_id,
        "exp_dir": str(exp_dir),
        "counts": {
            "task_runs": len(task_runs),
            "task_results": len(task_results),
            "profile_logs": len(profile_logs),
            "memory_logs": len(memory_logs),
            "missing_result_runs": len(missing_result_runs),
        },
        "profiles": {
            "retrieve_ms": summarize_metric(retrieve_ms),
            "compose_ms": summarize_metric(compose_ms),
            "agent_ms": summarize_metric(agent_ms),
            "memory_write_ms": summarize_metric(memory_write_ms),
            "total_ms": summarize_metric(total_ms),
            "raw_support_candidates": summarize_metric(raw_support_candidates),
            "filtered_support_candidates": summarize_metric(filtered_support_candidates),
        },
        "quality": {
            "evaluated_rows": len(eval_rows),
            "unknown_count": unknown_count,
            "unknown_ratio": unknown_ratio,
        },
        "support_memory": {
            "support_retrieve_count": support_retrieve_count,
            "support_retrieve_task_runs": len(support_retrieve_task_runs),
            "support_retrieve_raw_count": support_retrieve_raw_count,
            "support_retrieve_raw_task_runs": len(support_retrieve_raw_task_runs),
        },
        "missing_result_examples": missing_result_runs[:5],
        "final_task_examples": eval_rows[:5],
    }

    print("=" * 80)
    print(f"experiment_id: {args.experiment_id}")
    print(f"exp_dir: {exp_dir}")
    print("- counts")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print("- profiles")
    print(json.dumps(summary["profiles"], ensure_ascii=False, indent=2))
    print("- quality")
    print(json.dumps(summary["quality"], ensure_ascii=False, indent=2))
    print("- support_memory")
    print(json.dumps(summary["support_memory"], ensure_ascii=False, indent=2))

    if missing_result_runs:
        print("- missing_result_examples")
        print(json.dumps(summary["missing_result_examples"], ensure_ascii=False, indent=2))

    if summary["final_task_examples"]:
        print("- final_task_examples")
        print(json.dumps(summary["final_task_examples"], ensure_ascii=False, indent=2))

    if args.save_json:
        out_file = exp_dir / "summary.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_file}")


if __name__ == "__main__":
    main()
