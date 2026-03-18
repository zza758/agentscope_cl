import json
from pathlib import Path
from typing import Any, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Any]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_or_jsonl(path: str) -> List[Any]:
    """
    更鲁棒的加载器：
    1. 先尝试按标准 JSON 读取
    2. 如果失败，再回退按 JSONL 读取
    """
    p = Path(path)

    # 如果后缀是 jsonl，优先按 jsonl
    if p.suffix == ".jsonl":
        return load_jsonl(path)

    # 先尝试按标准 json
    try:
        data = load_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            return [data]
        raise ValueError(f"Unsupported json structure: {type(data)}")
    except json.JSONDecodeError:
        # 回退：按 jsonl 再试一次
        return load_jsonl(path)


def dump_jsonl(items: List[Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            if hasattr(item, "to_dict"):
                row = item.to_dict()
            else:
                row = item
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
