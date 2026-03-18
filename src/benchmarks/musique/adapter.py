from typing import Dict, List, Any, Set

from src.benchmarks.common.schema import UnifiedTask, SupportUnit
from src.benchmarks.common.io import load_json_or_jsonl


def load_musique(path: str) -> List[Dict]:
    return load_json_or_jsonl(path)


def get_sample_id(sample: Dict, idx: int) -> str:
    return str(sample.get("id", sample.get("question_id", idx)))


def get_question(sample: Dict) -> str:
    return sample.get("question") or sample.get("query") or ""


def get_answer(sample: Dict) -> str:
    return sample.get("answer") or sample.get("answer_text") or ""


def get_paragraphs(sample: Dict) -> List[Dict]:
    return sample.get("paragraphs", [])


def get_question_decomposition(sample: Dict) -> List[Dict]:
    return sample.get("question_decomposition", [])


def paragraph_title(p: Dict, fallback_idx: int) -> str:
    return p.get("title") or f"para_{fallback_idx}"


def paragraph_text(p: Dict) -> str:
    return p.get("paragraph_text") or p.get("text") or ""


def paragraph_global_idx(p: Dict, fallback_idx: int) -> int:
    return p.get("idx", fallback_idx)


def build_paragraph_map(sample: Dict) -> Dict[int, Dict]:
    para_map = {}
    for i, p in enumerate(get_paragraphs(sample)):
        para_idx = paragraph_global_idx(p, i)
        para_map[para_idx] = p
    return para_map


def extract_support_units_for_indices(sample: Dict, para_indices: Set[int], sample_id: str) -> List[SupportUnit]:
    para_map = build_paragraph_map(sample)
    units = []

    for para_idx in sorted(para_indices):
        p = para_map.get(para_idx)
        if not p:
            continue

        title = paragraph_title(p, para_idx)
        text = paragraph_text(p)
        unit_id = f"musique::{sample_id}::{title}::{para_idx}"

        units.append(
            SupportUnit(
                unit_id=unit_id,
                source_dataset="musique",
                source_sample_id=sample_id,
                title=title,
                sent_id=para_idx,
                text=text,
                entity=title,
                meta={
                    "paragraph_index": para_idx,
                    "is_supporting": True,
                },
            )
        )

    return units


def infer_entity_from_indices(sample: Dict, para_indices: Set[int]) -> str:
    para_map = build_paragraph_map(sample)
    titles = []
    for para_idx in sorted(para_indices):
        p = para_map.get(para_idx)
        if not p:
            continue
        title = paragraph_title(p, para_idx)
        if title and title not in titles:
            titles.append(title)
    return " || ".join(titles[:2]) if titles else ""


def convert_musique_sample_to_stream(sample: Dict, idx: int):
    sample_id = get_sample_id(sample, idx)
    stream_id = f"musique_stream::{sample_id}"

    tasks: List[UnifiedTask] = []
    all_support_units: List[SupportUnit] = []

    decomposition = get_question_decomposition(sample)

    # 1) 先生成 decomposition 子任务
    for step_i, step in enumerate(decomposition, start=1):
        para_indices = set()
        val = step.get("paragraph_support_idx", [])
        if isinstance(val, int):
            para_indices.add(val)
        elif isinstance(val, list):
            for x in val:
                if isinstance(x, int):
                    para_indices.add(x)

        support_units = extract_support_units_for_indices(sample, para_indices, sample_id)
        all_support_units.extend(support_units)

        task = UnifiedTask(
            task_id=f"musique::{sample_id}::subq::{step_i}",
            stream_id=stream_id,
            task_order=step_i,
            query=step.get("question", ""),
            answer=step.get("answer", ""),
            task_type="decomposition_qa",
            entity=infer_entity_from_indices(sample, para_indices),
            source_dataset="musique",
            source_sample_id=sample_id,
            gold_support_units=[u.unit_id for u in support_units],
            support_task_ids=[],
            history_ref=[],
            meta={
                "step_id": step.get("id", step_i),
                "support_titles": [u.title for u in support_units if u.title],
            },
        )
        tasks.append(task)

    # 2) 最后生成 final compositional task
    final_support_indices = set()
    for step in decomposition:
        val = step.get("paragraph_support_idx", [])
        if isinstance(val, int):
            final_support_indices.add(val)
        elif isinstance(val, list):
            for x in val:
                if isinstance(x, int):
                    final_support_indices.add(x)

    final_support_units = extract_support_units_for_indices(sample, final_support_indices, sample_id)
    # 去重追加
    existing_ids = {u.unit_id for u in all_support_units}
    for u in final_support_units:
        if u.unit_id not in existing_ids:
            all_support_units.append(u)

    final_task = UnifiedTask(
        task_id=f"musique::{sample_id}::final",
        stream_id=stream_id,
        task_order=len(tasks) + 1,
        query=get_question(sample),
        answer=get_answer(sample),
        task_type="compositional_multihop",
        entity=infer_entity_from_indices(sample, final_support_indices),
        source_dataset="musique",
        source_sample_id=sample_id,
        gold_support_units=[u.unit_id for u in final_support_units],
        support_task_ids=[t.task_id for t in tasks],
        history_ref=[t.task_id for t in tasks],
        meta={
            "num_paragraphs": len(get_paragraphs(sample)),
            "raw_sample_keys": list(sample.keys()),
        },
    )
    tasks.append(final_task)

    return tasks, all_support_units
