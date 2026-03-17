import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tasks-file", type=str, required=True)
    parser.add_argument("--labels-file", type=str, required=True)
    parser.add_argument("--settings", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "tasks_file": args.tasks_file,
        "labels_file": args.labels_file,
        "settings": [x.strip() for x in args.settings.split(",") if x.strip()],
        "notes": args.notes,
    }

    path = output_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"manifest written to: {path}")


if __name__ == "__main__":
    main()
