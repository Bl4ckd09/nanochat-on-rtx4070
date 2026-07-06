#!/usr/bin/env python3
"""Merge classifier outputs with bookmark metadata; emit curriculum.json + per-unit digest."""
import json, os, collections

SP = os.path.dirname(os.path.abspath(__file__))

books = {}
with open(os.path.join(SP, "ml_bookmarks.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        books[r["id"]] = r

merged = []
for i in range(6):
    with open(os.path.join(SP, f"classified_{i}.json")) as f:
        for c in json.load(f):
            b = books.get(c["id"])
            if not b or c["unit"] == "OFF" or c["score"] == 0:
                continue
            merged.append({
                "id": c["id"], "unit": c["unit"], "score": c["score"],
                "takeaway": c["takeaway"], "url": b["url"], "author": b["author"],
                "handle": b["handle"], "created": b["created"], "likes": b["likes"],
            })

by_unit = collections.defaultdict(list)
for m in merged:
    by_unit[m["unit"]].append(m)
for u in by_unit:
    by_unit[u].sort(key=lambda m: (-m["score"], -m["likes"]))

with open(os.path.join(SP, "curriculum.json"), "w") as f:
    json.dump({u: v for u, v in sorted(by_unit.items())}, f, ensure_ascii=False, indent=1)

print(f"total relevant: {len(merged)} / 1301")
for u, v in sorted(by_unit.items(), key=lambda kv: -len(kv[1])):
    s3 = sum(1 for m in v if m["score"] == 3)
    print(f"{u:12s} n={len(v):3d}  score3={s3:3d}")

# digest: top 14 per unit for curriculum writing
with open(os.path.join(SP, "digest.md"), "w") as f:
    for u, v in sorted(by_unit.items()):
        f.write(f"\n## {u} (n={len(v)})\n")
        for m in v[:14]:
            f.write(f"- [{m['score']}] {m['handle']} ({m['likes']}♥ {m['created']}): {m['takeaway']} | {m['url']}\n")
print("wrote curriculum.json + digest.md")
