#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL:-${REPO_DIR}/data/manual_reasoning_chat_v1.jsonl}"
MIN_MANUAL_ROWS="${MIN_MANUAL_ROWS:-100}"
CHECK_ONLY="${CHECK_ONLY:-0}"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

validate_manual_jsonl() {
  python3 - "${MANUAL_REASONING_JSONL}" "${MIN_MANUAL_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_rows = int(sys.argv[2])
if not path.exists():
    raise SystemExit(f"manual reasoning JSONL not found: {path}")

rows = 0
with path.open("r", encoding="utf-8") as f:
    for lineno, raw in enumerate(f, start=1):
        line = raw.strip()
        if not line:
            continue
        rows += 1
        messages = json.loads(line)
        if not isinstance(messages, list) or len(messages) < 2:
            raise SystemExit(f"{path}:{lineno}: expected a conversation list with at least 2 messages")
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise SystemExit(f"{path}:{lineno}: message {idx} is not an object")
            role = message.get("role")
            content = message.get("content")
            expected_role = "user" if idx % 2 == 0 else "assistant"
            if role != expected_role:
                raise SystemExit(f"{path}:{lineno}: message {idx} has role {role!r}, expected {expected_role!r}")
            if not isinstance(content, str) or not content.strip():
                raise SystemExit(f"{path}:{lineno}: message {idx} has empty content")

print(f"validated_rows={rows}")
if rows < min_rows:
    raise SystemExit(f"manual reasoning JSONL has only {rows} rows; require at least {min_rows} before launching")
PY
}

validate_manual_jsonl

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[ok] manual reasoning dataset passed validation: ${MANUAL_REASONING_JSONL}"
  exit 0
fi

MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL}" \
SEEDS="${SEEDS:-42,43}" \
ATTEMPT_ORDER="${ATTEMPT_ORDER:-s768_gc:768:7680,s640_gc:640:7680,s512_gc:512:7680}" \
QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN:-27.0}" \
QUICK_GATE_REQUIRE_PASS1_NONZERO="${QUICK_GATE_REQUIRE_PASS1_NONZERO:-1}" \
EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-500}" \
FULL_CONFIRM_MAX_PROBLEMS="${FULL_CONFIRM_MAX_PROBLEMS:-1000}" \
PROMOTE_PASS_COUNT_MIN="${PROMOTE_PASS_COUNT_MIN:-2}" \
PROMOTE_GSM8K_MIN="${PROMOTE_GSM8K_MIN:-4.60}" \
PROMOTE_MMLU_MIN="${PROMOTE_MMLU_MIN:-27.40}" \
DATASET_PRESET="reasoning_manual_v1" \
bash "${REPO_DIR}/tools/automation/run_reasoning_seed_sweep.sh" \
  "reasoning_manual_v1" \
  "reasoning_manual_v1" \
  "${BASE_MODEL_TAG:-d24_asp48_track}" \
  "${BASE_MODEL_STEP:-820230}" \
  "${NUM_ITERATIONS:-300}"
