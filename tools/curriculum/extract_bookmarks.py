#!/usr/bin/env python3
"""Extract ML-internals bookmark notes from the Obsidian vault into JSONL batches."""
import json, os, re, sys

VAULT = os.path.expanduser("~/Documents/Obsidian Vault/sources/twitter")
OUT = os.path.dirname(os.path.abspath(__file__))

KEYWORDS = re.compile(
    r"attention|MoE|mixture of experts|expert rout|rotary|RoPE|position embed"
    r"|KV.?cache|RLHF|GRPO|RLVR|reward model|reward hack|nanochat|nanogpt|karpathy"
    r"|verifiers|prime intellect|willccbb|tokeniz|\bmuP\b|Muon|GQA|latent attention"
    r"|\bMLA\b|sliding window|speculative dec|quantiz|distill|scaling law|pretrain"
    r"|\bSFT\b|softmax|logits?\b|eval|benchmark|transformer|LLM training|from scratch",
    re.IGNORECASE,
)

FM_KEYS = ("url", "author", "handle", "created", "likes")

def parse_note(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    if not raw.startswith("---"):
        return None
    try:
        fm_end = raw.index("\n---", 3)
    except ValueError:
        return None
    fm, body = raw[3:fm_end], raw[fm_end + 4:]
    meta = {}
    for line in fm.splitlines():
        m = re.match(r"^(\w+):\s*(.*)$", line.strip())
        if m and m.group(1) in FM_KEYS:
            meta[m.group(1)] = m.group(2).strip('"')
    # body: drop H1 header line, footer links
    lines = []
    for ln in body.splitlines():
        if ln.startswith("# ") or ln.startswith("[View on X]") or ln.startswith("See also") or ln.strip() == "---":
            continue
        lines.append(ln)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return meta, text

records = []
for root, _, files in os.walk(VAULT):
    for f in sorted(files):
        if not f.endswith(".md") or f == "Twitter Bookmarks.md":
            continue
        path = os.path.join(root, f)
        parsed = parse_note(path)
        if not parsed:
            continue
        meta, text = parsed
        if not KEYWORDS.search(text):
            continue
        records.append({
            "id": len(records),
            "file": os.path.relpath(path, VAULT),
            "url": meta.get("url", ""),
            "author": meta.get("author", ""),
            "handle": meta.get("handle", ""),
            "created": meta.get("created", ""),
            "likes": int(meta.get("likes", "0") or 0),
            "text": text[:600],
        })

print(f"matched {len(records)} notes")
with open(os.path.join(OUT, "ml_bookmarks.jsonl"), "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

NBATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
per = (len(records) + NBATCH - 1) // NBATCH
for i in range(NBATCH):
    chunk = records[i * per:(i + 1) * per]
    if not chunk:
        continue
    with open(os.path.join(OUT, f"batch_{i}.jsonl"), "w") as f:
        for r in chunk:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"batch_{i}.jsonl: {len(chunk)} notes")
