#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from validate_teacher_distilled_v1 import DEFAULT_RAW, load_rows, normalize_text

DEFAULT_OUT = Path(__file__).with_name("teacher_distilled_v1.jsonl")
DEFAULT_META = Path(__file__).with_name("teacher_distilled_v1_metadata.json")


def messages_to_conversation(messages):
    return [
        {"role": message["role"], "content": normalize_text(message["content"])}
        for message in messages
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build teacher_distilled_v1.jsonl from raw rows")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--include-medium", action="store_true", help="include rows marked medium quality")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if len(rows) < args.min_rows:
        raise SystemExit(f"{args.input} has only {len(rows)} rows; require at least {args.min_rows}")

    kept = []
    dropped = []
    for row in rows:
        if row["quality"] == "high" or args.include_medium:
            kept.append(row)
        else:
            dropped.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in kept:
            json.dump(messages_to_conversation(row["messages"]), f, ensure_ascii=True)
            f.write("\n")

    category_counts = Counter(row["category"] for row in kept)
    source_counts = Counter(row["source"] for row in kept)
    teacher_counts = Counter(row["teacher_model"] for row in kept)
    quality_counts = Counter(row["quality"] for row in rows)
    meta = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "rows_in_raw": len(rows),
        "rows_emitted": len(kept),
        "rows_dropped": len(dropped),
        "include_medium": args.include_medium,
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_model_counts": dict(sorted(teacher_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
    }
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
