from typing import Dict, List

from src.benchmarks.longmemeval.adapter import convert_lme_sample


def build_longmemeval_streams(samples: List[Dict]):
    all_tasks = []
    for idx, sample in enumerate(samples):
        all_tasks.extend(convert_lme_sample(sample, idx))
    return all_tasks
