from typing import Dict, List, Tuple

from src.benchmarks.common.schema import UnifiedTask, SupportUnit
from src.benchmarks.common.io import load_json_or_jsonl


def load_hotpot_json(path: str) -> List[Dict]:
    return load_json_or_jsonl(path)


def build_context_map(context: List[List]) -> Dict[str, List[str]]:
    return {title: sents for title, sents in context}


def extract_support_units(sample: Dict) -> List[SupportUnit]:
    sample_id = sample["_id"]
    context_map = build_context_map(sample["context"])
    units = []

    for title, sent_id in sample.get("supporting_facts", []):
        sents = context_map.get(title, [])
        text = sents[sent_id] if 0 <= sent_id < len(sents) else ""
        unit_id = f"hotpot::{sample_id}::{title}::{sent_id}"
        units.append(
            SupportUnit(
                unit_id=unit_id,
                source_dataset="hotpotqa",
                source_sample_id=sample_id,
                title=title,
                sent_id=sent_id,
                text=text,
                entity=title,
                meta={},
            )
        )
    return units


def infer_entity(sample: Dict) -> str:
    titles = [t for t, _ in sample.get("supporting_facts", [])]
    if titles:
        if sample.get("type") == "comparison" and len(titles) >= 2:
            return " || ".join(titles[:2])
        return titles[0]
    return ""


def convert_sample_to_task(sample: Dict, stream_id: str, task_order: int) -> Tuple[UnifiedTask, List[SupportUnit]]:
    support_units = extract_support_units(sample)
    task = UnifiedTask(
        task_id=f"hotpot::{sample['_id']}",
        stream_id=stream_id,
        task_order=task_order,
        query=sample["question"],
        answer=sample.get("answer"),
        task_type=sample.get("type"),
        entity=infer_entity(sample),
        source_dataset="hotpotqa",
        source_sample_id=sample["_id"],
        gold_support_units=[u.unit_id for u in support_units],
        support_task_ids=[],
        history_ref=[],
        meta={
            "level": sample.get("level"),
            "context_titles": [x[0] for x in sample.get("context", [])],
        },
    )
    return task, support_units
