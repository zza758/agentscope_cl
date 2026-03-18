from typing import Dict, List

from src.benchmarks.musique.adapter import convert_musique_sample_to_stream


def build_musique_streams(samples: List[Dict]):
    all_tasks = []
    all_units = []

    for idx, sample in enumerate(samples):
        tasks, units = convert_musique_sample_to_stream(sample, idx)
        all_tasks.extend(tasks)
        all_units.extend(units)

    return all_tasks, all_units
