#!/usr/bin/env python3
import argparse
import hashlib
import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

DEFAULT_OUT = Path(__file__).with_name("teacher_distilled_v1_raw.jsonl")
DEFAULT_META = Path(__file__).with_name("teacher_distilled_v1_raw_metadata.json")

DEFAULT_OPENTHOUGHTS_QUOTAS = {
    "science": 60,
    "puzzle": 50,
    "logic": 30,
    "grounded_qa": 20,
}
DEFAULT_MAGPIE_QUOTAS = {
    "reasoning": 45,
    "data-analysis": 35,
    "information-seeking": 20,
}
DEFAULT_OPENR1_TARGET = 40

MIN_USER_CHARS = 35
MAX_USER_CHARS = 650
MIN_ASSISTANT_CHARS = 35
MAX_ASSISTANT_CHARS = 700
MAX_ASSISTANT_WORDS = 110
MIN_REWARD = 0.24
MAGPIE_SCAN_LIMIT = 40000
OPENTHOUGHTS_SCAN_LIMIT = 114000
OPENR1_SCAN_LIMIT = 50000

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
    "implement",
    "time complexity",
    "space complexity",
]

MCQ_PATTERNS = [
    "which of the following",
    "select one answer",
    "choose the correct answer",
    "multiple choice",
    "\na.",
    "\nb.",
    "\nc.",
    "\nd.",
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]

LONG_COT_MARKERS = [
    "let's think step by step",
    "let us think step by step",
    "step-by-step",
    "chain of thought",
    "reasoning:",
    "solution:",
]

USER_PREFIX_PATTERNS = [
    r"^return your final response within\s+\\boxed\{\}\.?\s*",
    r"^give your final answer in\s+\\boxed\{\}\.?\s*",
    r"^put your final answer in\s+\\boxed\{\}\.?\s*",
]

GENERIC_PREFIX_PATTERNS = [
    r"^let'?s think step by step[:,\s-]*",
    r"^let'?s solve this(?: carefully)?[:,\s-]*",
    r"^we need to[:,\s-]*",
    r"^here(?:'s| is) (?:a )?(?:concise )?solution[:,\s-]*",
    r"^step-by-step explanation[:,\s-]*",
    r"^solution[:,\s-]*",
    r"^reasoning[:,\s-]*",
    r"^step\s*\d+[:.)\-\s]*",
    r"^\d+\.\s*",
    r"^(?:first|second|third|finally)[,:\s-]+",
]

ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{([^{}]{1,120})\}"),
    re.compile(r"(?:final answer|answer)\s*[:\-]\s*([^\n.]{1,160})", re.IGNORECASE),
    re.compile(r"the answer is\s+([^\n.]{1,160})", re.IGNORECASE),
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
        "ecosystem",
        "climate",
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
        "deduce",
        "deduction",
    ],
    "logic": [
        "if all",
        "if some",
        "must be true",
        "cannot be true",
        "conclusion follows",
        "logical consequence",
        "syllogism",
        "deduce",
    ],
}


@dataclass
class RawRow:
    row_id: str
    source: str
    teacher_model: str
    category: str
    quality: str
    tags: List[str]
    reward: float
    user: str
    assistant: str


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_user_prompt(text: str) -> str:
    text = normalize_text(text)
    lowered = text.lower()
    for pat in USER_PREFIX_PATTERNS:
        if re.match(pat, lowered):
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()
            lowered = text.lower()
    return normalize_text(text)


def stable_score(seed: int, payload: str) -> int:
    digest = hashlib.sha1(f"{seed}:{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def strip_markup(text: str) -> str:
    text = text.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
    text = text.replace("<|begin_of_thought|>", "").replace("<|end_of_thought|>", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
    return normalize_text(text)


def sentence_chunks(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [normalize_text(p) for p in parts if normalize_text(p)]


def extract_answer_hint(text: str) -> str:
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            value = normalize_text(match.group(1)).strip(" .$")
            if value and re.search(r"[A-Za-z0-9]", value):
                return value
    return ""


def classify_prompt(user: str) -> str:
    lowered = normalize_user_prompt(user).lower()
    for category, keywords in OPENTHOUGHTS_CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    if any(token in lowered for token in ["table", "chart", "average", "median", "mean", "trend", "dataset", "graph"]):
        return "data_analysis"
    if any(token in lowered for token in ["how many", "what percent", "total", "ratio", "probability", "equation", "solve for"]):
        return "math"
    return "grounded_qa"


def normalize_teacher_answer(text: str) -> str:
    text = strip_markup(text)
    if not text or "```" in text:
        return ""
    lowered_text = text.lower()
    if any(marker in lowered_text for marker in LONG_COT_MARKERS):
        return ""

    answer_hint = extract_answer_hint(text)
    chunks: List[str] = []
    total_chars = 0
    total_words = 0
    for raw in sentence_chunks(text):
        lowered = raw.lower()
        for pat in GENERIC_PREFIX_PATTERNS:
            if re.match(pat, lowered):
                raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()
                lowered = raw.lower()
        if not raw:
            continue
        if total_chars + len(raw) > 360:
            break
        words = len(raw.split())
        if total_words + words > MAX_ASSISTANT_WORDS:
            break
        chunks.append(raw)
        total_chars += len(raw)
        total_words += words
        if len(chunks) >= 3:
            break

    if answer_hint and answer_hint.lower() in {"a", "b", "c", "d"}:
        return ""

    if answer_hint:
        explanation_chunks = [chunk for chunk in chunks if answer_hint.lower() not in chunk.lower()]
        explanation = " ".join(explanation_chunks[:2]).strip()
        if explanation:
            return normalize_text(f"Answer: {answer_hint}. Why: {explanation}")
        return normalize_text(f"Answer: {answer_hint}.")

    return normalize_text(" ".join(chunks[:3]))


def is_valid_pair(user: str, assistant: str) -> bool:
    if not user or not assistant:
        return False
    if len(user) < MIN_USER_CHARS or len(user) > MAX_USER_CHARS:
        return False
    if len(assistant) < MIN_ASSISTANT_CHARS or len(assistant) > MAX_ASSISTANT_CHARS:
        return False
    if len(assistant.split()) > MAX_ASSISTANT_WORDS:
        return False
    if assistant.count("\n") > 2:
        return False
    lowered_user = user.lower()
    lowered_assistant = assistant.lower()
    if any(token in lowered_user for token in CODE_PATTERNS):
        return False
    if any(token in lowered_user for token in MCQ_PATTERNS):
        return False
    if any(marker in lowered_assistant for marker in LONG_COT_MARKERS):
        return False
    if re.fullmatch(r"(answer:\s*)?[a-d][.]?", lowered_assistant.strip()):
        return False
    return True


def brevity_rank(user: str, assistant: str) -> int:
    return max(0, 650 - len(user)) * 1_000 + max(0, 700 - len(assistant)) * 100 + max(0, 110 - len(assistant.split()))


def selection_score(seed: int, payload: str, user: str, assistant: str, reward: float = 0.0) -> int:
    return int(reward * 1000) * 1_000_000 + brevity_rank(user, assistant) * 1000 + stable_score(seed, payload) % 1000


def push_topk(heap: List[Tuple[int, str, RawRow]], k: int, score: int, row: RawRow) -> None:
    item = (score, row.row_id, row)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def collect_magpie(seed: int, quotas: Dict[str, int]) -> Tuple[List[RawRow], Dict[str, int]]:
    ds = load_dataset(
        "argilla/magpie-ultra-v1.0",
        "top_300k_shorter_conversations",
        split="train",
        streaming=True,
    )
    heaps: Dict[str, List[Tuple[int, str, RawRow]]] = {category: [] for category in quotas}
    seen = set()
    scanned = 0
    accepted = {category: 0 for category in quotas}

    for example in ds:
        scanned += 1
        if scanned > MAGPIE_SCAN_LIMIT:
            break
        category = example.get("category")
        if category not in quotas:
            continue
        reward = float(example.get("reward_model_score") or 0.0)
        if reward < MIN_REWARD:
            continue
        messages = example.get("messages") or []
        if len(messages) < 2:
            continue
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            continue
        user = normalize_user_prompt(messages[0].get("content", ""))
        assistant = normalize_teacher_answer(messages[1].get("content", ""))
        if not is_valid_pair(user, assistant):
            continue
        key = user.lower()
        if key in seen:
            continue
        seen.add(key)
        accepted[category] += 1
        row = RawRow(
            row_id=f"magpie-{category}-{stable_score(seed, user) % 10_000_000:07d}",
            source="magpie-ultra",
            teacher_model="llama-3.1-405b-instruct",
            category=category.replace("-", "_"),
            quality="high",
            tags=["teacher_generated", "short_answer", "filtered"],
            reward=reward,
            user=user,
            assistant=assistant,
        )
        score = selection_score(seed, f"magpie:{category}:{user}", user, assistant, reward)
        push_topk(heaps[category], quotas[category], score, row)

    rows: List[RawRow] = []
    for heap in heaps.values():
        rows.extend(row for _, _, row in sorted(heap, key=lambda item: -item[0]))
    return rows, {**accepted, "scanned": scanned}


def collect_openthoughts(seed: int, quotas: Dict[str, int]) -> Tuple[List[RawRow], Dict[str, int]]:
    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
    heaps: Dict[str, List[Tuple[int, str, RawRow]]] = {category: [] for category in quotas}
    seen = set()
    scanned = 0
    accepted = {category: 0 for category in quotas}

    for example in ds:
        scanned += 1
        if scanned > OPENTHOUGHTS_SCAN_LIMIT:
            break
        conversations = example.get("conversations") or []
        if len(conversations) < 2:
            continue
        if conversations[0].get("from") != "user" or conversations[1].get("from") != "assistant":
            continue
        user = normalize_user_prompt(conversations[0].get("value", ""))
        assistant = normalize_teacher_answer(conversations[1].get("value", ""))
        if not is_valid_pair(user, assistant):
            continue
        category = classify_prompt(user)
        if category not in quotas:
            continue
        key = user.lower()
        if key in seen:
            continue
        seen.add(key)
        accepted[category] += 1
        row = RawRow(
            row_id=f"openthoughts-{category}-{stable_score(seed, user) % 10_000_000:07d}",
            source="open-thoughts-114k",
            teacher_model="open-thoughts-source",
            category=category,
            quality="high",
            tags=["teacher_generated", "short_answer", "filtered"],
            reward=0.0,
            user=user,
            assistant=assistant,
        )
        score = selection_score(seed, f"openthoughts:{category}:{user}", user, assistant)
        push_topk(heaps[category], quotas[category], score, row)

    rows: List[RawRow] = []
    for heap in heaps.values():
        rows.extend(row for _, _, row in sorted(heap, key=lambda item: -item[0]))
    return rows, {**accepted, "scanned": scanned}


def collect_openr1(seed: int, target: int) -> Tuple[List[RawRow], Dict[str, int]]:
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    heap: List[Tuple[int, str, RawRow]] = []
    seen = set()
    scanned = 0
    accepted = 0

    for example in ds:
        scanned += 1
        if scanned > OPENR1_SCAN_LIMIT:
            break
        user = normalize_user_prompt(example.get("problem", ""))
        assistant = normalize_teacher_answer(example.get("solution", ""))
        if not is_valid_pair(user, assistant):
            continue
        key = user.lower()
        if key in seen:
            continue
        seen.add(key)
        accepted += 1
        row = RawRow(
            row_id=f"openr1-math-{stable_score(seed, user) % 10_000_000:07d}",
            source="open-r1-math-220k",
            teacher_model="open-r1-source",
            category="math",
            quality="high",
            tags=["teacher_generated", "short_answer", "filtered"],
            reward=0.0,
            user=user,
            assistant=assistant,
        )
        score = selection_score(seed, f"openr1:{user}", user, assistant)
        push_topk(heap, target, score, row)

    rows = [row for _, _, row in sorted(heap, key=lambda item: -item[0])]
    return rows, {"accepted": accepted, "scanned": scanned}


def raw_record(row: RawRow) -> Dict[str, object]:
    return {
        "id": row.row_id,
        "source": row.source,
        "teacher_model": row.teacher_model,
        "category": row.category,
        "quality": row.quality,
        "tags": row.tags,
        "messages": [
            {"role": "user", "content": row.user},
            {"role": "assistant", "content": row.assistant},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build teacher_distilled_v1 raw rows from open sources")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--ot-science", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["science"])
    parser.add_argument("--ot-puzzle", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["puzzle"])
    parser.add_argument("--ot-logic", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["logic"])
    parser.add_argument("--ot-grounded-qa", type=int, default=DEFAULT_OPENTHOUGHTS_QUOTAS["grounded_qa"])
    parser.add_argument("--magpie-reasoning", type=int, default=DEFAULT_MAGPIE_QUOTAS["reasoning"])
    parser.add_argument("--magpie-data-analysis", type=int, default=DEFAULT_MAGPIE_QUOTAS["data-analysis"])
    parser.add_argument("--magpie-information-seeking", type=int, default=DEFAULT_MAGPIE_QUOTAS["information-seeking"])
    parser.add_argument("--openr1-target", type=int, default=DEFAULT_OPENR1_TARGET)
    args = parser.parse_args()

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

    openthoughts_rows, openthoughts_stats = collect_openthoughts(args.seed, openthoughts_quotas)
    openr1_rows, openr1_stats = collect_openr1(args.seed, args.openr1_target)
    magpie_rows, magpie_stats = collect_magpie(args.seed, magpie_quotas)

    rows: List[RawRow] = []
    seen = set()
    source_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    for row in openthoughts_rows + openr1_rows + magpie_rows:
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
            handle.write(json.dumps(raw_record(row), ensure_ascii=True) + "\n")

    meta = {
        "seed": args.seed,
        "output": str(args.output),
        "rows": len(rows),
        "source_counts": source_counts,
        "category_counts": category_counts,
        "openthoughts_stats": openthoughts_stats,
        "openr1_stats": openr1_stats,
        "magpie_stats": magpie_stats,
        "openthoughts_quotas": openthoughts_quotas,
        "magpie_quotas": magpie_quotas,
        "openr1_target": args.openr1_target,
        "filters": {
            "min_user_chars": MIN_USER_CHARS,
            "max_user_chars": MAX_USER_CHARS,
            "min_assistant_chars": MIN_ASSISTANT_CHARS,
            "max_assistant_chars": MAX_ASSISTANT_CHARS,
            "max_assistant_words": MAX_ASSISTANT_WORDS,
            "min_reward": MIN_REWARD,
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
