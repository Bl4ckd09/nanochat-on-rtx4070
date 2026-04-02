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

DEFAULT_OUT = Path(__file__).with_name("teacher_reasoning_v1.jsonl")
DEFAULT_META = Path(__file__).with_name("teacher_reasoning_v1_metadata.json")

MAGPIE_QUOTAS = {
    "reasoning": 110,
    "data-analysis": 90,
    "information-seeking": 50,
    "math": 50,
}
ALLOWED_MAGPIE_CATEGORIES = set(MAGPIE_QUOTAS)
MIN_USER_CHARS = 40
MAX_USER_CHARS = 800
MIN_ASSISTANT_CHARS = 40
MAX_ASSISTANT_CHARS = 1600
MIN_REWARD = 0.14
ORCA_TARGET = 300
ORCA_SCAN_LIMIT = 25000
MAGPIE_SCAN_LIMIT = 25000


@dataclass
class Candidate:
    key: str
    source: str
    source_meta: Dict[str, object]
    messages: List[Dict[str, str]]


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
    banned_user = [
        "roleplay",
        "write a poem",
        "creative story",
        "fan fiction",
        "write a song",
        "sql query",
        "python program",
        "javascript",
        "c++",
        "debug this code",
    ]
    lowered = user.lower()
    if any(token in lowered for token in banned_user):
        return False
    return True


def first_turn_pair(messages: List[Dict[str, str]]) -> Optional[Tuple[str, str]]:
    if len(messages) < 2:
        return None
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        return None
    user = normalize_text(messages[0].get("content", ""))
    assistant = normalize_text(messages[1].get("content", ""))
    if not is_valid_pair(user, assistant):
        return None
    return user, assistant


def push_topk(heap: List[Tuple[int, Candidate]], k: int, score: int, cand: Candidate) -> None:
    item = (-score, cand)
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        if item > heap[0]:
            heapq.heapreplace(heap, item)


def collect_magpie(seed: int) -> Tuple[List[Candidate], Dict[str, int]]:
    ds = load_dataset(
        "argilla/magpie-ultra-v1.0",
        "top_300k_shorter_conversations",
        split="train",
        streaming=True,
    )
    heaps: Dict[str, List[Tuple[int, Candidate]]] = {cat: [] for cat in MAGPIE_QUOTAS}
    seen = set()
    scanned = 0
    accepted = {cat: 0 for cat in MAGPIE_QUOTAS}

    for ex in ds:
        scanned += 1
        if scanned > MAGPIE_SCAN_LIMIT:
            break
        category = ex.get("category")
        if category not in ALLOWED_MAGPIE_CATEGORIES:
            continue
        reward = float(ex.get("reward_model_score") or 0.0)
        if reward < MIN_REWARD:
            continue
        pair = first_turn_pair(ex.get("messages") or [])
        if pair is None:
            continue
        user, assistant = pair
        dedupe_key = normalize_text(user).lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        accepted[category] += 1
        cand = Candidate(
            key=dedupe_key,
            source="magpie-ultra",
            source_meta={
                "category": category,
                "difficulty": ex.get("difficulty"),
                "quality": ex.get("quality"),
                "reward_model_score": reward,
                "conversation_tokens": ex.get("conversation_tokens"),
            },
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        score = stable_score(seed, f"magpie:{category}:{user}")
        push_topk(heaps[category], MAGPIE_QUOTAS[category], score, cand)

    selected = []
    for _, heap in heaps.items():
        ranked = sorted(heap, key=lambda x: -x[0])
        selected.extend(c for _, c in ranked)
    return selected, {**accepted, "scanned": scanned}


def collect_orca(seed: int) -> Tuple[List[Candidate], Dict[str, int]]:
    ds = load_dataset(
        "microsoft/orca-math-word-problems-200k",
        split="train",
        streaming=True,
    )
    heap: List[Tuple[int, Candidate]] = []
    seen = set()
    scanned = 0
    accepted = 0

    for ex in ds:
        scanned += 1
        if scanned > ORCA_SCAN_LIMIT:
            break
        user = normalize_text(ex.get("question", ""))
        assistant = normalize_text(ex.get("answer", ""))
        if not is_valid_pair(user, assistant):
            continue
        dedupe_key = user.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        accepted += 1
        cand = Candidate(
            key=dedupe_key,
            source="orca-math",
            source_meta={},
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        score = stable_score(seed, f"orca:{user}")
        push_topk(heap, ORCA_TARGET, score, cand)

    selected = [c for _, c in sorted(heap, key=lambda x: -x[0])]
    return selected, {"accepted": accepted, "scanned": scanned}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    magpie, magpie_stats = collect_magpie(args.seed)
    orca, orca_stats = collect_orca(args.seed)

    all_rows = []
    seen = set()
    source_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    for cand in magpie + orca:
        key = cand.key
        if key in seen:
            continue
        seen.add(key)
        all_rows.append(cand)
        source_counts[cand.source] = source_counts.get(cand.source, 0) + 1
        category = str(cand.source_meta.get("category", "")) if cand.source == "magpie-ultra" else "math"
        category_counts[category] = category_counts.get(category, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for cand in all_rows:
            f.write(json.dumps(cand.messages, ensure_ascii=True) + "\n")

    meta = {
        "seed": args.seed,
        "output": str(args.output),
        "rows": len(all_rows),
        "source_counts": source_counts,
        "category_counts": category_counts,
        "magpie_stats": magpie_stats,
        "orca_stats": orca_stats,
        "magpie_quotas": MAGPIE_QUOTAS,
        "orca_target": ORCA_TARGET,
        "magpie_source": {
            "dataset": "argilla/magpie-ultra-v1.0",
            "config": "top_300k_shorter_conversations",
        },
        "orca_source": {
            "dataset": "microsoft/orca-math-word-problems-200k",
        },
    }
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    rc = main()
    # Hugging Face streaming sometimes leaves background state that aborts during interpreter finalization.
    # Exit the process directly after all files are flushed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
