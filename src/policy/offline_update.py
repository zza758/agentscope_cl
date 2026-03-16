import argparse
import json
from pathlib import Path

import numpy as np

from src.policy.bandit_model import LinUCBModel


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if model_path.exists():
        model = LinUCBModel.load(str(model_path))
    else:
        model = LinUCBModel(dim=args.dim, alpha=args.alpha)

    logs = load_jsonl(Path(args.log_path))
    for row in logs:
        reward = float(row.get("reward", 0.0))
        for item in row.get("selected_memories", []):
            feat = item.get("policy_feature")
            if feat is None:
                continue
            x = np.array(feat, dtype=np.float64)
            model.update(x, reward)

    model.save(str(model_path))
    print(f"updated from {len(logs)} logs -> {model_path}")


if __name__ == "__main__":
    main()
