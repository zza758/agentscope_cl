from collections import defaultdict
from typing import Dict, List

from src.benchmarks.hotpotqa.adapter import convert_sample_to_task


def build_stream_groups(samples: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for s in samples:
        titles = [t for t, _ in s.get("supporting_facts", [])]
        key = titles[0] if titles else "misc"
        groups[key].append(s)
    return groups


def sort_group(group: List[Dict]) -> List[Dict]:
    level_rank = {"easy": 0, "medium": 1, "hard": 2}
    return sorted(group, key=lambda x: level_rank.get(x.get("level"), 99))


def build_hotpot_streams(samples: List[Dict]):
    groups = build_stream_groups(samples)
    all_tasks = []
    all_units = []

    for gid, group in groups.items():
        ordered = sort_group(group)
        prev_tasks = []

        for idx, sample in enumerate(ordered, start=1):
            stream_id = f"hotpot_stream::{gid}"
            task, support_units = convert_sample_to_task(sample, stream_id, idx)

            prev_ids = []
            current_titles = set([u.title for u in support_units if u.title])

            for prev_task in prev_tasks:
                prev_titles = set(prev_task.meta.get("context_titles", []))
                if current_titles & prev_titles:
                    prev_ids.append(prev_task.task_id)

            task.support_task_ids = prev_ids
            all_tasks.append(task)
            all_units.extend(support_units)
            prev_tasks.append(task)

    return all_tasks, all_units
