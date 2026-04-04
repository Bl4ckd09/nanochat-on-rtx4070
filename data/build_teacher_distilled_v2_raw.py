#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import build_teacher_distilled_v1_raw as base

DEFAULT_OUT = Path(__file__).with_name("teacher_distilled_v2_raw.jsonl")
DEFAULT_META = Path(__file__).with_name("teacher_distilled_v2_raw_metadata.json")
DEFAULT_MANUAL_JSONL = Path(__file__).with_name("manual_reasoning_chat_v1.jsonl")

DEFAULT_OPENTHOUGHTS_QUOTAS = {
    "science": 50,
    "puzzle": 40,
    "logic": 25,
    "grounded_qa": 15,
}
DEFAULT_MAGPIE_QUOTAS = {
    "reasoning": 50,
    "data-analysis": 30,
    "information-seeking": 20,
}
DEFAULT_OPENR1_TARGET = 35
DEFAULT_MANUAL_TARGET = 15
DEFAULT_MANUAL_QUOTAS = {
    "math": 4,
    "logic": 3,
    "science": 2,
    "grounded_qa": 3,
    "information_seeking": 2,
    "reasoning": 1,
}


def classify_manual_anchor(user: str) -> str:
    lowered = base.normalize_user_prompt(user).lower()
    if any(token in lowered for token in ["budget", "compare models", "what two details", "what should i figure out first", "before comparing models"]):
        return "information_seeking"
    if any(token in lowered for token in ["study order", "what should i do first", "what makes the most sense", "stronger evidence"]):
        return "reasoning"
    if "phishing" in lowered:
        return ""
    if "exactly two bullets" in lowered:
        return ""
    category = base.classify_prompt(user)
    if category == "grounded_qa" and any(token in lowered for token in ["evidence", "compare", "which is stronger"]):
        return "reasoning"
    return category


def collect_manual_anchors(seed: int, target: int, path: Path) -> tuple[list[base.RawRow], dict[str, int]]:
    if not path.exists() or target <= 0:
        return [], {"accepted": 0, "scanned": 0}

    quotas = dict(DEFAULT_MANUAL_QUOTAS)
    total_quota = sum(quotas.values())
    if total_quota < target:
        quotas["reasoning"] += target - total_quota
    elif total_quota > target:
        quotas["reasoning"] = max(0, quotas["reasoning"] - (total_quota - target))

    heaps: dict[str, list[tuple[int, str, base.RawRow]]] = {category: [] for category in quotas}
    seen = set()
    scanned = 0
    accepted = {category: 0 for category in quotas}

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        scanned += 1
        messages = json.loads(line)
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            continue
        user = base.normalize_user_prompt(messages[0].get("content", ""))
        assistant = base.normalize_teacher_answer(messages[1].get("content", ""))
        if not base.is_valid_pair(user, assistant):
            continue
        category = classify_manual_anchor(user)
        if category not in quotas or quotas[category] <= 0:
            continue
        key = user.lower()
        if key in seen:
            continue
        seen.add(key)
        accepted[category] += 1
        row = base.RawRow(
            row_id=f"manual-anchor-{seed}-{lineno:04d}",
            source="manual_reasoning_v1",
            teacher_model="manual_anchor",
            category=category,
            quality="high",
            tags=["manual_anchor", "short_answer", "filtered"],
            reward=0.0,
            user=user,
            assistant=assistant,
        )
        score = base.selection_score(seed, f"manual:{category}:{user}", user, assistant)
        base.push_topk(heaps[category], quotas[category], score, row)

    rows: list[base.RawRow] = []
    for heap in heaps.values():
        rows.extend(row for _, _, row in sorted(heap, key=lambda item: -item[0]))
    return rows, {**accepted, "scanned": scanned}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build teacher_distilled_v2 raw rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--manual-anchor-jsonl", type=Path, default=DEFAULT_MANUAL_JSONL)
    parser.add_argument("--manual-target", type=int, default=DEFAULT_MANUAL_TARGET)
    parser.add_argument("--ot-science", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["science"])
    parser.add_argument("--ot-puzzle", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["puzzle"])
    parser.add_argument("--ot-logic", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["logic"])
    parser.add_argument("--ot-grounded-qa", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["grounded_qa"])
    parser.add_argument("--magpie-reasoning", type=int, default=DEFAULT_MAGPIE_QUOTAS["reasoning"])
    parser.add_argument("--magpie-data-analysis", type=int, default=DEFAULT_MAGPIE_QUOTAS["data-analysis"])
    parser.add_argument("--magpie-information-seeking", type=int, default=DEFAULT_MAGPIE_QUOTAS["information-seeking"])
    parser.add_argument("--openr1-target", type=int, default=DEFAULT_OPENR1_TARGET)
    parser.add_argument("--magpie-min-reward", type=float, default=0.17)
    parser.add_argument("--max-assistant-chars", type=int, default=850)
    parser.add_argument("--max-assistant-words", type=int, default=130)
    parser.add_argument("--magpie-scan-limit", type=int, default=50000)
    parser.add_argument("--openthoughts-scan-limit", type=int, default=114000)
    parser.add_argument("--openr1-scan-limit", type=int, default=50000)
    args = parser.parse_args()

    base.MIN_REWARD = args.magpie_min_reward
    base.MAX_ASSISTANT_CHARS = args.max_assistant_chars
    base.MAX_ASSISTANT_WORDS = args.max_assistant_words
    base.MAGPIE_SCAN_LIMIT = args.magpie_scan_limit
    base.OPENTHOUGHTS_SCAN_LIMIT = args.openthoughts_scan_limit
    base.OPENR1_SCAN_LIMIT = args.openr1_scan_limit

    openthoughts_quotas = {
        "science": args.ot_science,
        "puzzle": args.ot_puzzle,
        "logic": args.ot_logic,
        "grounded_qa": args.ot_grounded_qa,
    }
    magpie_quotas = {
        "reasoning": args.magpie_reasoning,
        "data-analysis": args.magpie_data_analysis,
        "information-seeking": args.magpie_information_seeking,
    }

    openthoughts_rows, openthoughts_stats = base.collect_openthoughts(args.seed, openthoughts_quotas)
    openr1_rows, openr1_stats = base.collect_openr1(args.seed, args.openr1_target)
    magpie_rows, magpie_stats = base.collect_magpie(args.seed, magpie_quotas)
    manual_rows, manual_stats = collect_manual_anchors(args.seed, args.manual_target, args.manual_anchor_jsonl)

    rows: list[base.RawRow] = []
    seen = set()
    source_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for row in openthoughts_rows + openr1_rows + magpie_rows + manual_rows:
        key = row.user.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        source_counts[row.source] = source_counts.get(row.source, 0) + 1
        category_counts[row.category] = category_counts.get(row.category, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(base.raw_record(row), ensure_ascii=True) + "\n")

    meta = {
        "seed": args.seed,
        "output": str(args.output),
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "openthoughts_stats": openthoughts_stats,
        "openr1_stats": openr1_stats,
        "magpie_stats": magpie_stats,
        "manual_stats": manual_stats,
        "openthoughts_quotas": openthoughts_quotas,
        "magpie_quotas": magpie_quotas,
        "openr1_target": args.openr1_target,
        "manual_target": args.manual_target,
        "filters": {
            "magpie_min_reward": args.magpie_min_reward,
            "max_assistant_chars": args.max_assistant_chars,
            "max_assistant_words": args.max_assistant_words,
            "mcq_filtered": True,
            "short_answer_normalized": True,
            "code_filtered": True,
        },
    }
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
