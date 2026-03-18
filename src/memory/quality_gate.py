import re
from typing import Any, Dict, List, Optional


DEFAULT_UNKNOWN_PATTERNS = [
    "未知",
    "未提供",
    "无法确定",
    "不清楚",
    "信息不足",
    "need more information",
    "not provided",
    "unknown",
    "cannot determine",
]

DEFAULT_PLACEHOLDER_PATTERNS = [
    r"#\d+",
    r"\bplaceholder\b",
]

DEFAULT_WEAK_PATTERNS = [
    "需要进一步查询",
    "需进一步查询",
    "根据历史经验",
]


def _text_contains_any(text: str, patterns: List[str], regex: bool = False) -> bool:
    if not text:
        return False
    if regex:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)
    lowered = text.lower()
    return any((p or "").lower() in lowered for p in patterns)


def _min_summary_len_for_task(task_type: Optional[str], cfg: Dict[str, Any]) -> int:
    by_task_type = cfg.get("min_summary_len_by_task_type", {}) or {}
    if task_type and task_type in by_task_type:
        return int(by_task_type[task_type])
    return int(cfg.get("min_summary_len_default", 8))


def assess_memory_quality(
    query: str,
    final_answer: str,
    memory_summary: str,
    task_type: Optional[str],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", True))
    summary = (memory_summary or "").strip()
    answer = (final_answer or "").strip()

    unknown_patterns = cfg.get("unknown_patterns") or DEFAULT_UNKNOWN_PATTERNS
    placeholder_patterns = cfg.get("placeholder_patterns") or DEFAULT_PLACEHOLDER_PATTERNS
    weak_patterns = cfg.get("weak_patterns") or DEFAULT_WEAK_PATTERNS

    contains_unknown = _text_contains_any(summary, unknown_patterns) or _text_contains_any(
        answer, unknown_patterns
    )
    contains_placeholder = _text_contains_any(summary, placeholder_patterns, regex=True) or _text_contains_any(
        answer, placeholder_patterns, regex=True
    )
    contains_weak = _text_contains_any(summary, weak_patterns)
    min_summary_len = _min_summary_len_for_task(task_type, cfg)

    if not enabled:
        return {
            "memory_quality": "unknown",
            "contains_placeholder": contains_placeholder,
            "contains_unknown": contains_unknown,
            "gate_passed": True,
            "gate_reason": "quality_gate_disabled",
        }

    if not summary:
        return {
            "memory_quality": "reject",
            "contains_placeholder": contains_placeholder,
            "contains_unknown": contains_unknown,
            "gate_passed": False,
            "gate_reason": "empty_summary",
        }

    if cfg.get("reject_unknown", True) and contains_unknown:
        return {
            "memory_quality": "reject",
            "contains_placeholder": contains_placeholder,
            "contains_unknown": contains_unknown,
            "gate_passed": False,
            "gate_reason": "contains_unknown",
        }

    if cfg.get("reject_placeholder", True) and contains_placeholder:
        return {
            "memory_quality": "reject",
            "contains_placeholder": contains_placeholder,
            "contains_unknown": contains_unknown,
            "gate_passed": False,
            "gate_reason": "contains_placeholder",
        }

    if contains_weak and len(summary) < max(min_summary_len, 12):
        return {
            "memory_quality": "reject",
            "contains_placeholder": contains_placeholder,
            "contains_unknown": contains_unknown,
            "gate_passed": False,
            "gate_reason": "weak_summary",
        }

    if len(summary) < min_summary_len:
        quality = "low" if task_type != "decomposition_qa" else "medium"
        return {
            "memory_quality": quality,
            "contains_placeholder": contains_placeholder,
            "contains_unknown": contains_unknown,
            "gate_passed": True,
            "gate_reason": "short_but_allowed",
        }

    quality = "high" if len(summary) >= max(min_summary_len * 2, 24) else "medium"
    return {
        "memory_quality": quality,
        "contains_placeholder": contains_placeholder,
        "contains_unknown": contains_unknown,
        "gate_passed": True,
        "gate_reason": "accepted",
    }
