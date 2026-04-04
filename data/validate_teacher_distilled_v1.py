#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

DEFAULT_RAW = Path(__file__).with_name("teacher_distilled_v1_raw.jsonl")
ALLOWED_CATEGORIES = {
    "math",
    "science",
    "puzzle",
    "logic",
    "reasoning",
    "general_reasoning",
    "grounded_qa",
    "comparison",
    "data_analysis",
    "information_seeking",
    "safety",
    "format_following",
    "planning",
}
ALLOWED_QUALITIES = {"high", "medium"}
MAX_USER_WORDS = 180
MAX_ASSISTANT_WORDS = 220
MAX_TURNS = 4
LONG_COT_MARKERS = (
    "let's think step by step",
    "let us think step by step",
    "step-by-step",
    "chain of thought",
    "reasoning:",
    "solution:",
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def validate_messages(messages: List[Dict[str, str]], path: Path, lineno: int) -> None:
    if not isinstance(messages, list) or len(messages) < 2:
        raise SystemExit(f"{path}:{lineno}: messages must be a list with at least 2 entries")
    if len(messages) > MAX_TURNS:
        raise SystemExit(f"{path}:{lineno}: too many turns ({len(messages)} > {MAX_TURNS})")
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SystemExit(f"{path}:{lineno}: message {idx} is not an object")
        role = message.get("role")
        content = message.get("content")
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if role != expected_role:
            raise SystemExit(f"{path}:{lineno}: message {idx} role {role!r} != {expected_role!r}")
        if not isinstance(content, str) or not normalize_text(content):
            raise SystemExit(f"{path}:{lineno}: message {idx} has empty content")
        if "```" in content:
            raise SystemExit(f"{path}:{lineno}: fenced code blocks are not allowed")
        if idx % 2 == 0 and word_count(content) > MAX_USER_WORDS:
            raise SystemExit(f"{path}:{lineno}: user message {idx} exceeds {MAX_USER_WORDS} words")
        if idx % 2 == 1:
            lowered = content.lower()
            if word_count(content) > MAX_ASSISTANT_WORDS:
                raise SystemExit(f"{path}:{lineno}: assistant message {idx} exceeds {MAX_ASSISTANT_WORDS} words")
            if any(marker in lowered for marker in LONG_COT_MARKERS):
                raise SystemExit(f"{path}:{lineno}: assistant message {idx} looks like long-CoT spillover")


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"raw teacher-distilled file not found: {path}")

    rows: List[Dict[str, object]] = []
    seen_ids = set()
    seen_first_user = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: each line must be a JSON object")

        row_id = row.get("id")
        source = row.get("source")
        teacher_model = row.get("teacher_model")
        category = row.get("category")
        quality = row.get("quality")
        messages = row.get("messages")

        if not isinstance(row_id, str) or not row_id.strip():
            raise SystemExit(f"{path}:{lineno}: missing non-empty id")
        if row_id in seen_ids:
            raise SystemExit(f"{path}:{lineno}: duplicate id {row_id!r}")
        seen_ids.add(row_id)

        if not isinstance(source, str) or not source.strip():
            raise SystemExit(f"{path}:{lineno}: missing non-empty source")
        if not isinstance(teacher_model, str) or not teacher_model.strip():
            raise SystemExit(f"{path}:{lineno}: missing non-empty teacher_model")
        if category not in ALLOWED_CATEGORIES:
            raise SystemExit(f"{path}:{lineno}: category {category!r} not in {sorted(ALLOWED_CATEGORIES)}")
        if quality not in ALLOWED_QUALITIES:
            raise SystemExit(f"{path}:{lineno}: quality {quality!r} not in {sorted(ALLOWED_QUALITIES)}")

        validate_messages(messages, path, lineno)
        normalized_first_user = normalize_text(messages[0]["content"]).lower()
        if normalized_first_user in seen_first_user:
            raise SystemExit(f"{path}:{lineno}: duplicate first user turn detected")
        seen_first_user.add(normalized_first_user)
        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    category_counts = Counter(row["category"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    teacher_counts = Counter(row["teacher_model"] for row in rows)
    quality_counts = Counter(row["quality"] for row in rows)
    assistant_words = [word_count(row["messages"][-1]["content"]) for row in rows]
    return {
        "rows": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_model_counts": dict(sorted(teacher_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "avg_assistant_words": round(sum(assistant_words) / len(assistant_words), 1) if assistant_words else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw teacher_distilled_v1 rows")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print summary as JSON")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if len(rows) < args.min_rows:
        raise SystemExit(f"{args.input} has only {len(rows)} rows; require at least {args.min_rows}")
    summary = summarize(rows)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"validated_rows={summary['rows']}")
        print(f"avg_assistant_words={summary['avg_assistant_words']}")
        print(f"category_counts={summary['category_counts']}")
        print(f"source_counts={summary['source_counts']}")
        print(f"teacher_model_counts={summary['teacher_model_counts']}")
        print(f"quality_counts={summary['quality_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
