#!/usr/bin/env python3
import argparse
import hashlib
import heapq
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

DEFAULT_OUT = Path(__file__).with_name("teacher_reasoning_v2.jsonl")
DEFAULT_META = Path(__file__).with_name("teacher_reasoning_v2_metadata.json")

DEFAULT_OPENTHOUGHTS_QUOTAS = {
    "math": 80,
    "science": 60,
    "puzzle": 50,
    "general-reasoning": 30,
}
DEFAULT_MAGPIE_QUOTAS = {
    "reasoning": 50,
    "data-analysis": 40,
    "information-seeking": 30,
}
DEFAULT_OPENR1_TARGET = 180
MIN_USER_CHARS = 40
MAX_USER_CHARS = 900
MIN_ASSISTANT_CHARS = 40
MAX_ASSISTANT_CHARS = 2200
MIN_REWARD = 0.14
MAGPIE_SCAN_LIMIT = 25000
OPENTHOUGHTS_SCAN_LIMIT = 114000
OPENR1_SCAN_LIMIT = 25000


@dataclass
class Candidate:
    key: str
    source: str
    source_meta: Dict[str, object]
    messages: List[Dict[str, str]]


CODE_PATTERNS = [
    "python",
    "stdin",
    "stdout",
    "write a program",
    "write a function",
    "codeforces",
    "leetcode",
    "class solution",
    "compile",
    "javascript",
    "java",
    "c++",
    "cpp",
    "sql",
    "debug this code",
    "executable",
    "implement",
]

OPENTHOUGHTS_CATEGORY_KEYWORDS = {
    "science": [
        "physics",
        "chemistry",
        "biology",
        "cell",
        "genetics",
        "molecule",
        "atom",
        "force",
        "energy",
        "thermodynamics",
        "electric",
        "magnetic",
        "planet",
        "orbit",
        "star",
        "geology",
    ],
    "puzzle": [
        "logic puzzle",
        "riddle",
        "liar",
        "truth-teller",
        "knight",
        "arrangement",
        "seating",
        "probability",
        "game",
        "strategy",
        "puzzle",
        "combinatorics",
    ],
    "math": [
        "prove",
        "triangle",
        "polynomial",
        "equation",
        "inequality",
        "integral",
        "derivative",
        "function",
        "matrix",
        "number theory",
        "geometry",
        "algebra",
        "find the maximum",
        "evaluate",
        "real numbers",
        "\\boxed",
    ],
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_score(seed: int, payload: str) -> int:
    digest = hashlib.sha1(f"{seed}:{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def is_valid_pair(user: str, assistant: str) -> bool:
    user = normalize_text(user)
    assistant = normalize_text(assistant)
    if not user or not assistant:
        return False
    if len(user) < MIN_USER_CHARS or len(user) > MAX_USER_CHARS:
        return False
    if len(assistant) < MIN_ASSISTANT_CHARS or len(assistant) > MAX_ASSISTANT_CHARS:
        return False
    lowered = user.lower()
    if any(token in lowered for token in CODE_PATTERNS):
        return False
    return True


def push_topk(heap: List[Tuple[int, Candidate]], k: int, score: int, cand: Candidate) -> None:
    item = (-score, cand)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def clean_solution_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
    text = text.replace("<|begin_of_thought|>", "").replace("<|end_of_thought|>", "")
    return normalize_text(text)


def extract_solution_from_openthoughts(text: str) -> str:
    text = normalize_text(text)
    if "<|end_of_thought|>" in text:
        text = text.split("<|end_of_thought|>", 1)[1]
    text = clean_solution_text(text)
    return text


def classify_openthoughts(user: str) -> str:
    lowered = normalize_text(user).lower()
    for category, keywords in OPENTHOUGHTS_CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "general-reasoning"


def collect_magpie(seed: int, magpie_quotas: Dict[str, int]) -> Tuple[List[Candidate], Dict[str, int]]:
    ds = load_dataset(
        "argilla/magpie-ultra-v1.0",
        "top_300k_shorter_conversations",
        split="train",
        streaming=True,
    )
    heaps: Dict[str, List[Tuple[int, Candidate]]] = {cat: [] for cat in magpie_quotas}
    seen = set()
    scanned = 0
    accepted = {cat: 0 for cat in magpie_quotas}

    for ex in ds:
        scanned += 1
        if scanned > MAGPIE_SCAN_LIMIT:
            break
        category = ex.get("category")
        if category not in magpie_quotas:
            continue
        reward = float(ex.get("reward_model_score") or 0.0)
        if reward < MIN_REWARD:
            continue
        messages = ex.get("messages") or []
        if len(messages) < 2:
            continue
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            continue
        user = normalize_text(messages[0].get("content", ""))
        assistant = normalize_text(messages[1].get("content", ""))
        if not is_valid_pair(user, assistant):
            continue
        dedupe_key = user.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        accepted[category] += 1
        cand = Candidate(
            key=dedupe_key,
            source="magpie-ultra",
            source_meta={
                "category": category,
                "reward_model_score": reward,
            },
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        push_topk(heaps[category], magpie_quotas[category], stable_score(seed, f"magpie:{category}:{user}"), cand)

    selected: List[Candidate] = []
    for heap in heaps.values():
        selected.extend(c for _, c in sorted(heap, key=lambda x: -x[0]))
    return selected, {**accepted, "scanned": scanned}


def collect_openthoughts(seed: int, quotas: Dict[str, int]) -> Tuple[List[Candidate], Dict[str, int]]:
    ds = load_dataset(
        "open-thoughts/OpenThoughts-114k",
        split="train",
        streaming=True,
    )
    heaps: Dict[str, List[Tuple[int, Candidate]]] = {cat: [] for cat in quotas}
    seen = set()
    scanned = 0
    accepted = {cat: 0 for cat in quotas}

    for ex in ds:
        scanned += 1
        if scanned > OPENTHOUGHTS_SCAN_LIMIT:
            break
        conv = ex.get("conversations") or []
        if len(conv) < 2:
            continue
        if conv[0].get("from") != "user" or conv[1].get("from") != "assistant":
            continue
        user = normalize_text(conv[0].get("value", ""))
        assistant = extract_solution_from_openthoughts(conv[1].get("value", ""))
        if not is_valid_pair(user, assistant):
            continue
        category = classify_openthoughts(user)
        if category not in quotas:
            continue
        dedupe_key = user.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        accepted[category] += 1
        cand = Candidate(
            key=dedupe_key,
            source="open-thoughts-114k",
            source_meta={"category": category},
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        push_topk(heaps[category], quotas[category], stable_score(seed, f"open-thoughts:{category}:{user}"), cand)

    selected: List[Candidate] = []
    for heap in heaps.values():
        selected.extend(c for _, c in sorted(heap, key=lambda x: -x[0]))
    return selected, {**accepted, "scanned": scanned}


def collect_openr1(seed: int, target: int) -> Tuple[List[Candidate], Dict[str, int]]:
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        split="train",
        streaming=True,
    )
    heap: List[Tuple[int, Candidate]] = []
    seen = set()
    scanned = 0
    accepted = 0

    for ex in ds:
        scanned += 1
        if scanned > OPENR1_SCAN_LIMIT:
            break
        user = normalize_text(ex.get("problem", ""))
        assistant = clean_solution_text(ex.get("solution", ""))
        if not is_valid_pair(user, assistant):
            continue
        dedupe_key = user.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        accepted += 1
        cand = Candidate(
            key=dedupe_key,
            source="open-r1-math-220k",
            source_meta={"category": "math"},
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        push_topk(heap, target, stable_score(seed, f"open-r1:{user}"), cand)

    selected = [c for _, c in sorted(heap, key=lambda x: -x[0])]
    return selected, {"accepted": accepted, "scanned": scanned}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--ot-math", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["math"])
    parser.add_argument("--ot-science", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["science"])
    parser.add_argument("--ot-puzzle", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["puzzle"])
    parser.add_argument("--ot-general", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["general-reasoning"])
    parser.add_argument("--magpie-reasoning", type=int, default=DEFAULT_MAGPIE_QUOTAS["reasoning"])
    parser.add_argument("--magpie-data-analysis", type=int, default=DEFAULT_MAGPIE_QUOTAS["data-analysis"])
    parser.add_argument("--magpie-information-seeking", type=int, default=DEFAULT_MAGPIE_QUOTAS["information-seeking"])
    parser.add_argument("--openr1-target", type=int, default=DEFAULT_OPENR1_TARGET)
    args = parser.parse_args()

    openthoughts_quotas = {
        "math": args.ot_math,
        "science": args.ot_science,
        "puzzle": args.ot_puzzle,
        "general-reasoning": args.ot_general,
    }
    magpie_quotas = {
        "reasoning": args.magpie_reasoning,
        "data-analysis": args.magpie_data_analysis,
        "information-seeking": args.magpie_information_seeking,
    }

    openthoughts, ot_stats = collect_openthoughts(args.seed, openthoughts_quotas)
    openr1, openr1_stats = collect_openr1(args.seed, args.openr1_target)
    magpie, magpie_stats = collect_magpie(args.seed, magpie_quotas)

    rows: List[Candidate] = []
    seen = set()
    source_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    for cand in openthoughts + openr1 + magpie:
        if cand.key in seen:
            continue
        seen.add(cand.key)
        rows.append(cand)
        source_counts[cand.source] = source_counts.get(cand.source, 0) + 1
        category = str(cand.source_meta.get("category", ""))
        category_counts[category] = category_counts.get(category, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for cand in rows:
            f.write(json.dumps(cand.messages, ensure_ascii=True) + "\n")

    meta = {
        "seed": args.seed,
        "output": str(args.output),
        "rows": len(rows),
        "source_counts": source_counts,
        "category_counts": category_counts,
        "openthoughts_stats": ot_stats,
        "openr1_stats": openr1_stats,
        "magpie_stats": magpie_stats,
        "openthoughts_quotas": openthoughts_quotas,
        "magpie_quotas": magpie_quotas,
        "openr1_target": args.openr1_target,
        "sources": {
            "open-thoughts-114k": "open-thoughts/OpenThoughts-114k",
            "open-r1-math-220k": "open-r1/OpenR1-Math-220k",
            "magpie-ultra": "argilla/magpie-ultra-v1.0 top_300k_shorter_conversations",
        },
    }
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
