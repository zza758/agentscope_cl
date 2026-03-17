import argparse
import json
from pathlib import Path

from src.eval.support_graph import (
    load_jsonl,
    build_support_parent_map,
    build_ancestor_map,
    build_entity_map,
)


def build_label_map(labels):
    return {x["task_id"]: x for x in labels}


def extract_used_task_ids(used_memories):
    ids = set()
    for m in used_memories:
        if isinstance(m, dict) and m.get("task_id"):
            ids.add(m["task_id"])
    return ids


def compute_entity_coverage(task_id, used_ids, label_map, entity_map):
    expected_entities = entity_map.get(task_id, set())
    if not expected_entities:
        return 0, 0

    covered = set()
    for used_id in used_ids:
        covered |= entity_map.get(used_id, set())

    hit = len(expected_entities & covered)
    total = len(expected_entities)
    return hit, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--labels-file", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    labels = load_jsonl(Path(args.labels_file))
    label_map = build_label_map(labels)
    parent_map = build_support_parent_map(labels)
    ancestor_map = build_ancestor_map(labels)
    entity_map = build_entity_map(labels)

    rows = []
    for file in sorted(input_dir.glob("*.jsonl")):
        records = load_jsonl(file)
        for r in records:
            task_id = r.get("task_id")
            used_memories = r.get("used_memories", [])
            used_ids = extract_used_task_ids(used_memories)

            direct_support = set(parent_map.get(task_id, []))
            all_ancestors = set(ancestor_map.get(task_id, set()))

            direct_hit = len(direct_support & used_ids)
            direct_total = len(direct_support)

            ancestor_hit = len(all_ancestors & used_ids)
            ancestor_total = len(all_ancestors)

            entity_hit, entity_total = compute_entity_coverage(
                task_id=task_id,
                used_ids=used_ids,
                label_map=label_map,
                entity_map=entity_map,
            )

            label = label_map.get(task_id, {})
            rows.append({
                "task_id": task_id,
                "setting_name": r.get("setting_name"),
                "task_type": label.get("task_type"),
                "entity": label.get("entity"),
                "query": r.get("query"),
                "final_answer": r.get("final_answer"),
                "memory_summary": r.get("memory_summary"),
                "strategy_note": r.get("strategy_note"),
                "used_memory_count": len(used_memories),
                "used_memory_task_ids": sorted(list(used_ids)),

                "support_expected_task_ids": sorted(list(direct_support)),
                "support_hit_count": direct_hit,
                "support_expected_count": direct_total,

                "ancestor_expected_task_ids": sorted(list(all_ancestors)),
                "ancestor_hit_count": ancestor_hit,
                "ancestor_expected_count": ancestor_total,

                "entity_hit_count": entity_hit,
                "entity_expected_count": entity_total,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已汇总 {len(rows)} 条结果到: {output_path}")

    # 简单总览
    from collections import defaultdict
    agg = defaultdict(lambda: {
        "n": 0,
        "used": 0,
        "direct_hit": 0,
        "direct_total": 0,
        "ancestor_hit": 0,
        "ancestor_total": 0,
        "entity_hit": 0,
        "entity_total": 0,
    })

    for r in rows:
        s = r["setting_name"]
        agg[s]["n"] += 1
        agg[s]["used"] += r["used_memory_count"]
        agg[s]["direct_hit"] += r["support_hit_count"]
        agg[s]["direct_total"] += r["support_expected_count"]
        agg[s]["ancestor_hit"] += r["ancestor_hit_count"]
        agg[s]["ancestor_total"] += r["ancestor_expected_count"]
        agg[s]["entity_hit"] += r["entity_hit_count"]
        agg[s]["entity_total"] += r["entity_expected_count"]

    print("aggregate_by_setting_v2:")
    for k, v in agg.items():
        print(
            k,
            {
                "n": v["n"],
                "avg_used_memory": round(v["used"] / v["n"], 4) if v["n"] else 0,
                "direct_support_hit_rate": round(v["direct_hit"] / v["direct_total"], 4) if v["direct_total"] else 0,
                "ancestor_support_hit_rate": round(v["ancestor_hit"] / v["ancestor_total"], 4) if v["ancestor_total"] else 0,
                "entity_coverage_rate": round(v["entity_hit"] / v["entity_total"], 4) if v["entity_total"] else 0,
            }
        )


if __name__ == "__main__":
    main()
