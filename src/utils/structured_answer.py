import re
from typing import Dict


FINAL_MARKER = "【最终答案】"
SUMMARY_MARKER = "【记忆摘要】"
STRATEGY_MARKER = "【策略备注】"


def parse_structured_answer(text: str) -> Dict[str, str]:
    if not text:
        return {
            "final_answer": "",
            "memory_summary": "",
            "strategy_note": "",
        }

    final_answer = ""
    memory_summary = ""
    strategy_note = ""

    final_pattern = re.compile(
        rf"{re.escape(FINAL_MARKER)}(.*?)(?={re.escape(SUMMARY_MARKER)}|{re.escape(STRATEGY_MARKER)}|$)",
        re.S,
    )
    summary_pattern = re.compile(
        rf"{re.escape(SUMMARY_MARKER)}(.*?)(?={re.escape(STRATEGY_MARKER)}|$)",
        re.S,
    )
    strategy_pattern = re.compile(
        rf"{re.escape(STRATEGY_MARKER)}(.*)$",
        re.S,
    )

    m1 = final_pattern.search(text)
    m2 = summary_pattern.search(text)
    m3 = strategy_pattern.search(text)

    if m1:
        final_answer = m1.group(1).strip()
    if m2:
        memory_summary = m2.group(1).strip()
    if m3:
        strategy_note = m3.group(1).strip()

    # 回退策略：如果模型没按格式输出，就全部当作 final_answer
    if not final_answer:
        final_answer = text.strip()
    if not memory_summary:
        memory_summary = final_answer
    if not strategy_note:
        strategy_note = "可作为后续相似任务的参考经验。"

    return {
        "final_answer": final_answer,
        "memory_summary": memory_summary,
        "strategy_note": strategy_note,
    }
