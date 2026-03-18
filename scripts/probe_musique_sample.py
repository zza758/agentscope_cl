import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.benchmarks.common.io import load_json_or_jsonl

path = "/root/autodl-tmp/data/benchmarks/musique/musique_full_v1.0_dev.jsonl"
samples = load_json_or_jsonl(path)
sample = samples[0]

print("TOP KEYS:", list(sample.keys()))
print("=" * 80)

for k, v in sample.items():
    if isinstance(v, list):
        print(f"{k}: list, len={len(v)}")
        if v:
            print(" first item type:", type(v[0]))
            if isinstance(v[0], dict):
                print(" first item keys:", list(v[0].keys()))
    else:
        print(f"{k}: {type(v).__name__}")
