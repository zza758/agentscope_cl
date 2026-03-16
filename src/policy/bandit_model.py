import json
import math
from pathlib import Path
from typing import List

import numpy as np


class LinUCBModel:
    def __init__(self, dim: int, alpha: float = 0.5):
        self.dim = dim
        self.alpha = alpha
        self.A = np.eye(dim, dtype=np.float64)
        self.b = np.zeros((dim,), dtype=np.float64)

    def score(self, x: np.ndarray) -> float:
        x = x.astype(np.float64)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        exploit = float(theta @ x)
        explore = float(self.alpha * math.sqrt(x @ A_inv @ x))
        return exploit + explore

    def update(self, x: np.ndarray, reward: float):
        x = x.astype(np.float64)
        self.A += np.outer(x, x)
        self.b += reward * x

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "alpha": self.alpha,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(dim=payload["dim"], alpha=payload["alpha"])
        model.A = np.array(payload["A"], dtype=np.float64)
        model.b = np.array(payload["b"], dtype=np.float64)
        return model
