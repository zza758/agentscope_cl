def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory-path",
        type=str,
        default="data/memory/memory_bank.jsonl",
    )
    parser.add_argument(
        "--task-label-path",
        type=str,
        default="data/tasks/baseline_12/labels.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/contrastive/baseline_12_hard/memory_contrastive_samples.jsonl",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/contrastive/baseline_12_hard/memory_contrastive_meta.json",
    )
    parser.add_argument(
        "--negative-type",
        type=str,
        default="hard_legal_history",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser
