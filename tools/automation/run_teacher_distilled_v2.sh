#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
RAW_JSONL="${RAW_JSONL:-${REPO_DIR}/data/teacher_distilled_v2_raw.jsonl}"
RAW_METADATA_JSON="${RAW_METADATA_JSON:-${REPO_DIR}/data/teacher_distilled_v2_raw_metadata.json}"
MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL:-${REPO_DIR}/data/teacher_distilled_v2.jsonl}"
METADATA_JSON="${METADATA_JSON:-${REPO_DIR}/data/teacher_distilled_v2_metadata.json}"
MIN_RAW_ROWS="${MIN_RAW_ROWS:-240}"
CHECK_ONLY="${CHECK_ONLY:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"
TMPDIR="${TMPDIR:-${HOME}/tmp}"

mkdir -p "${NOTES_DIR}" "${TMPDIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

python3 "${REPO_DIR}/data/build_teacher_distilled_v2_raw.py" \
  --output "${RAW_JSONL}" \
  --metadata "${RAW_METADATA_JSON}" \
  --manual-anchor-jsonl "${MANUAL_ANCHOR_JSONL:-${REPO_DIR}/data/manual_reasoning_chat_v1.jsonl}" \
  --manual-target "${MANUAL_TARGET:-15}" \
  --ot-science "${OT_SCIENCE:-50}" \
  --ot-puzzle "${OT_PUZZLE:-40}" \
  --ot-logic "${OT_LOGIC:-25}" \
  --ot-grounded-qa "${OT_GROUNDED_QA:-15}" \
  --magpie-reasoning "${MAGPIE_REASONING:-50}" \
  --magpie-data-analysis "${MAGPIE_DATA_ANALYSIS:-30}" \
  --magpie-information-seeking "${MAGPIE_INFORMATION_SEEKING:-20}" \
  --openr1-target "${OPENR1_TARGET:-35}" \
  --magpie-min-reward "${MAGPIE_MIN_REWARD:-0.17}" \
  --max-assistant-chars "${MAX_ASSISTANT_CHARS:-850}" \
  --max-assistant-words "${MAX_ASSISTANT_WORDS:-130}"

python3 "${REPO_DIR}/data/validate_teacher_distilled_v1.py" --input "${RAW_JSONL}" --min-rows "${MIN_RAW_ROWS}"

python3 - "${RAW_METADATA_JSON}" "${MIN_RAW_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
min_rows = int(sys.argv[2])
rows = int(meta.get('rows', 0))
if rows < min_rows:
    raise SystemExit(f"raw rows {rows} < required {min_rows}")
source_counts = meta.get('source_counts', {})
category_counts = meta.get('category_counts', {})
magpie_rows = int(source_counts.get('magpie-ultra', 0))
if magpie_rows < 60:
    raise SystemExit(f"Magpie rows {magpie_rows} < required 60")
math_rows = int(category_counts.get('math', 0))
math_pct = (100.0 * math_rows / rows) if rows else 0.0
if math_pct > 20.0:
    raise SystemExit(f"math share {math_pct:.2f}% exceeds 20% cap")
for category, count in category_counts.items():
    share = 100.0 * count / rows if rows else 0.0
    if category == 'science':
        if share > 35.0:
            raise SystemExit(f"science share {share:.2f}% exceeds 35% cap")
    else:
        if share > 30.0:
            raise SystemExit(f"category {category} share {share:.2f}% exceeds 30% cap")
print(f"raw_rows={rows}")
print(f"magpie_rows={magpie_rows}")
print(f"math_pct={math_pct:.2f}")
PY

python3 "${REPO_DIR}/data/build_teacher_distilled_v1.py" --input "${RAW_JSONL}" --output "${MANUAL_REASONING_JSONL}" --metadata "${METADATA_JSON}" --min-rows "${MIN_RAW_ROWS}"

python3 - "${MANUAL_REASONING_JSONL}" "${MIN_RAW_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_rows = int(sys.argv[2])
rows = 0
for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
    line = raw.strip()
    if not line:
        continue
    rows += 1
    messages = json.loads(line)
    if not isinstance(messages, list) or len(messages) < 2:
        raise SystemExit(f"{path}:{lineno}: built row is not a valid conversation")
if rows < min_rows:
    raise SystemExit(f"built rows {rows} < required {min_rows}")
print(f"built_rows={rows}")
PY

if [[ "${CHECK_ONLY}" == "1" || "${BUILD_ONLY}" == "1" ]]; then
  echo "[ok] teacher_distilled_v2 scaffold passed validation/build"
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
DATASET_PRESET="teacher_distilled_v2" \
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${HOME}/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt}" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}" \
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}" \
FREEZE_LAYERS="${FREEZE_LAYERS:-20}" \
bash "${REPO_DIR}/tools/automation/run_reasoning_seed_sweep.sh" \
  "teacher_distilled_v2" \
  "teacher_distilled_v2" \
  "${BASE_MODEL_TAG:-d24_asp48_track}" \
  "${BASE_MODEL_STEP:-820230}" \
  "${NUM_ITERATIONS:-300}"
