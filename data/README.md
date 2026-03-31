# Curated SFT Data

This directory is for manually curated chat/SFT data that is specific to this fork.

Current planned use:

- `manual_reasoning_chat_v1.jsonl`
  - high-signal reasoning/chat conversations
  - used by the `reasoning_manual_v1` preset in `scripts/chat_sft.py`

JSONL schema:

- one conversation per line
- each line is a JSON array
- roles must alternate starting with `user`
- each message must contain:
  - `role`
  - `content`

Example:

```json
[{"role":"user","content":"What is 12 * 13?"},{"role":"assistant","content":"12 * 13 = 156."}]
```

Curation rules:

- keep examples short and high-value
- prefer reasoning, grounded QA, concise math, and disciplined refusals
- avoid filler chat
- avoid identity/personality stuffing in the generalist branch
- keep spelling/letter-count specialist data out of this file

Operational rule:

- do not launch `reasoning_manual_v1` from the starter scaffold alone
- the automation wrapper enforces a minimum row count before training
