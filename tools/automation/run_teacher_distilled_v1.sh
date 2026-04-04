#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
RAW_JSONL="${RAW_JSONL:-${REPO_DIR}/data/teacher_distilled_v1_raw.jsonl}"
MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL:-${REPO_DIR}/data/teacher_distilled_v1.jsonl}"
METADATA_JSON="${METADATA_JSON:-${REPO_DIR}/data/teacher_distilled_v1_metadata.json}"
MIN_RAW_ROWS="${MIN_RAW_ROWS:-200}"
INCLUDE_MEDIUM="${INCLUDE_MEDIUM:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
TMPDIR="${TMPDIR:-${HOME}/tmp}"

mkdir -p "${NOTES_DIR}" "${TMPDIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

python3 "${REPO_DIR}/data/validate_teacher_distilled_v1.py" --input "${RAW_JSONL}" --min-rows "${MIN_RAW_ROWS}"

build_cmd=(python3 "${REPO_DIR}/data/build_teacher_distilled_v1.py" --input "${RAW_JSONL}" --output "${MANUAL_REASONING_JSONL}" --metadata "${METADATA_JSON}" --min-rows "${MIN_RAW_ROWS}")
if [[ "${INCLUDE_MEDIUM}" == "1" ]]; then
  build_cmd+=(--include-medium)
fi
"${build_cmd[@]}"

python3 - "${MANUAL_REASONING_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = 0
for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
    line = raw.strip()
    if not line:
        continue
    rows += 1
    messages = json.loads(line)
    if not isinstance(messages, list) or len(messages) < 2:
        raise SystemExit(f"{path}:{lineno}: built row is not a valid conversation")
print(f"built_rows={rows}")
PY

if [[ "${CHECK_ONLY}" == "1" || "${BUILD_ONLY}" == "1" ]]; then
  echo "[ok] teacher_distilled_v1 scaffold passed validation/build"
  exit 0
fi

export TMPDIR
MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL}" \
SEEDS="${SEEDS:-42,43}" \
ATTEMPT_ORDER="${ATTEMPT_ORDER:-s768_gc:768:7680}" \
QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN:-27.0}" \
QUICK_GATE_REQUIRE_PASS1_NONZERO="${QUICK_GATE_REQUIRE_PASS1_NONZERO:-1}" \
EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-500}" \
FULL_CONFIRM_MAX_PROBLEMS="${FULL_CONFIRM_MAX_PROBLEMS:-1000}" \
PROMOTE_PASS_COUNT_MIN="${PROMOTE_PASS_COUNT_MIN:-2}" \
PROMOTE_GSM8K_MIN="${PROMOTE_GSM8K_MIN:-4.60}" \
PROMOTE_MMLU_MIN="${PROMOTE_MMLU_MIN:-27.40}" \
DETERMINISTIC="${DETERMINISTIC:-1}" \
INIT_LR_FRAC="${INIT_LR_FRAC:-0.05}" \
DATASET_PRESET="teacher_distilled_v1" \
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${HOME}/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt}" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}" \
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}" \
FREEZE_LAYERS="${FREEZE_LAYERS:-20}" \
bash "${REPO_DIR}/tools/automation/run_reasoning_seed_sweep.sh" \
  "teacher_distilled_v1" \
  "teacher_distilled_v1" \
  "${BASE_MODEL_TAG:-d24_asp48_track}" \
  "${BASE_MODEL_STEP:-820230}" \
  "${NUM_ITERATIONS:-300}"
