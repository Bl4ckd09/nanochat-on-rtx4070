#!/usr/bin/env bash
set -euo pipefail

STAGE_A_TAG="${STAGE_A_TAG:-d24_r32_adamw_curriculum_v1_2026-03-30_0635_stageA}"
STAGE_A_STEP="${STAGE_A_STEP:-200}"
STAGE_A_MODEL_GROUP="${STAGE_A_MODEL_GROUP:-${STAGE_A_TAG}/best}"
FREEZE_LAYERS="${FREEZE_LAYERS:-20}"
SEQ_LEN="${SEQ_LEN:-768}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-7680}"
QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN:-27.0}"
PROMOTE_GSM8K_MIN="${PROMOTE_GSM8K_MIN:-4.0}"
PROMOTE_MMLU_MIN="${PROMOTE_MMLU_MIN:-27.0}"
SEED="${SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-0}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
TS="$(date +%F_%H%M)"
RUN_BASE="d24_r32_adamw_stageb_booster_v2_${TS}"
MASTER_LOG="${NOTES_DIR}/${RUN_BASE}.log"
META_FILE="${NOTES_DIR}/${RUN_BASE}_meta.env"
DECISION_FILE="${NOTES_DIR}/${RUN_BASE}_decision.md"
ANCHOR_LOG="${NOTES_DIR}/${RUN_BASE}_anchor_quick.log"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}"

log() {
  echo "$1" | tee -a "${MASTER_LOG}"
}

extract_summary_path() {
  local eval_log="$1"
  rg '^summary=' "${eval_log}" | tail -n 1 | sed 's/^summary=//'
}

read_summary_metrics() {
  local summary_file="$1"
  python3 - "${summary_file}" <<'PY'
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text() if Path(sys.argv[1]).exists() else ""

def extract(label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

print(extract("GSM8K pass@1"), extract("GSM8K pass@8"), extract("MMLU"), extract("SpellingBee"))
PY
}

full_confirm_hits_band() {
  local summary_file="$1"
  python3 - "${summary_file}" "${PROMOTE_GSM8K_MIN}" "${PROMOTE_MMLU_MIN}" <<'PY'
import math
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
gsm_min = float(sys.argv[2])
mmlu_min = float(sys.argv[3])
text = summary_path.read_text() if summary_path.exists() else ""

def extract(label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

pass8 = extract("GSM8K pass@8")
mmlu = extract("MMLU")
raise SystemExit(0 if (not math.isnan(pass8) and not math.isnan(mmlu) and pass8 >= gsm_min and mmlu >= mmlu_min) else 1)
PY
}

latest_chain_for_prefix() {
  local prefix="$1"
  local files=()
  shopt -s nullglob
  files=("${NOTES_DIR}/sft_${prefix}_"*_chain.log)
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    return 1
  fi
  ls -1t "${files[@]}" | head -n 1
}

write_decision() {
  cat > "${DECISION_FILE}" <<EOF2
# ${RUN_BASE} decision

- stage_a_model_group: ${STAGE_A_MODEL_GROUP}
- stage_a_step: ${STAGE_A_STEP}
- anchor_quick_summary: ${ANCHOR_SUMMARY:-missing}
- anchor_gsm8k_pass1: ${ANCHOR_PASS1:-nan}%
- anchor_mmlu: ${ANCHOR_MMLU:-nan}%
- soft50_chain_log: ${SOFT50_CHAIN_LOG:-missing}
- soft50_summary: ${SOFT50_SUMMARY:-missing}
- soft50_gsm8k_pass1: ${SOFT50_PASS1:-nan}%
- soft50_gsm8k_pass8: ${SOFT50_PASS8:-nan}%
- soft50_mmlu: ${SOFT50_MMLU:-nan}%
- soft50_spellingbee: ${SOFT50_SPELL:-nan}%
- soft50_rc: ${SOFT50_RC:-na}
- soft25_chain_log: ${SOFT25_CHAIN_LOG:-missing}
- soft25_summary: ${SOFT25_SUMMARY:-missing}
- soft25_gsm8k_pass1: ${SOFT25_PASS1:-nan}%
- soft25_gsm8k_pass8: ${SOFT25_PASS8:-nan}%
- soft25_mmlu: ${SOFT25_MMLU:-nan}%
- soft25_spellingbee: ${SOFT25_SPELL:-nan}%
- soft25_rc: ${SOFT25_RC:-na}
- final_status: ${FINAL_STATUS}
- promoted_summary: ${PROMOTED_SUMMARY:-none}
EOF2
  log "[decision] ${DECISION_FILE}"
}

ANCHOR_SUMMARY=""
ANCHOR_PASS1="nan"
ANCHOR_PASS8="nan"
ANCHOR_MMLU="nan"
ANCHOR_SPELL="nan"
SOFT50_CHAIN_LOG=""
SOFT50_SUMMARY=""
SOFT50_PASS1="nan"
SOFT50_PASS8="nan"
SOFT50_MMLU="nan"
SOFT50_SPELL="nan"
SOFT50_RC="na"
SOFT25_CHAIN_LOG=""
SOFT25_SUMMARY=""
SOFT25_PASS1="nan"
SOFT25_PASS8="nan"
SOFT25_MMLU="nan"
SOFT25_SPELL="nan"
SOFT25_RC="na"
PROMOTED_SUMMARY=""
FINAL_STATUS="running"
LAST_CHAIN_LOG=""
LAST_SUMMARY=""
LAST_PASS1="nan"
LAST_PASS8="nan"
LAST_MMLU="nan"
LAST_SPELL="nan"
LAST_RC="na"

run_anchor_gate() {
  log "[anchor] quick gate on stage-A best checkpoint"
  set +e
  RUN_PASS1=1 SKIP_PASS1=0 SKIP_PASS8=1 SKIP_MMLU=0 SKIP_SPELLING=1 \
  EVAL_SEED="${EVAL_SEED}" DETERMINISTIC="${DETERMINISTIC}" \
  EVAL_WANDB_RUN_PREFIX="${RUN_BASE}_anchor_quick" \
  "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${STAGE_A_MODEL_GROUP}" "${STAGE_A_STEP}" 250 \
    |& tee "${ANCHOR_LOG}" | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  ANCHOR_SUMMARY="$(extract_summary_path "${ANCHOR_LOG}" || true)"
  if [[ -n "${ANCHOR_SUMMARY}" && -f "${ANCHOR_SUMMARY}" ]]; then
    read -r ANCHOR_PASS1 ANCHOR_PASS8 ANCHOR_MMLU ANCHOR_SPELL <<< "$(read_summary_metrics "${ANCHOR_SUMMARY}")"
    log "[anchor] pass1=${ANCHOR_PASS1}% mmlu=${ANCHOR_MMLU}% summary=${ANCHOR_SUMMARY}"
  else
    log "[anchor] summary missing rc=${rc}"
  fi
  return 0
}

run_candidate() {
  local label="$1"
  local iterations="$2"
  local init_lr_frac="$3"
  local prefix="d24_r32_adamw_stageb_v2_${label}"

  LAST_CHAIN_LOG=""
  LAST_SUMMARY=""
  LAST_PASS1="nan"
  LAST_PASS8="nan"
  LAST_MMLU="nan"
  LAST_SPELL="nan"
  LAST_RC="na"

  log "[candidate] ${label}: source=sft tag=${STAGE_A_MODEL_GROUP} step=${STAGE_A_STEP} iters=${iterations} init_lr_frac=${init_lr_frac}"
  set +e
  MODEL_SOURCE=sft \
  RUN_BASE_PREFIX="${prefix}" \
  FREEZE_LAYERS="${FREEZE_LAYERS}" \
  DATASET_PRESET=curriculum_boost_v2 \
  INIT_LR_FRAC="${init_lr_frac}" \
  ATTEMPT_ORDER="s768_gc:${SEQ_LEN}:${TOTAL_BATCH_SIZE}" \
  QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN}" \
  QUICK_GATE_REQUIRE_PASS1_NONZERO=0 \
  FULL_CONFIRM_MAX_PROBLEMS=1000 \
  EVAL_MAX_PROBLEMS=250 \
  SEED="${SEED}" \
  EVAL_SEED="${EVAL_SEED}" \
  DETERMINISTIC="${DETERMINISTIC}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  bash "${REPO_DIR}/tools/automation/run_sft_adamw_control_nextbest.sh" "${STAGE_A_MODEL_GROUP}" "${STAGE_A_STEP}" "${iterations}" \
    |& tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e

  LAST_RC="${rc}"
  if LAST_CHAIN_LOG="$(latest_chain_for_prefix "${prefix}" 2>/dev/null || true)"; then
    :
  else
    LAST_CHAIN_LOG=""
  fi
  if [[ -n "${LAST_CHAIN_LOG}" && -f "${LAST_CHAIN_LOG}" ]]; then
    LAST_SUMMARY="$(rg '^\[summary\] ' "${LAST_CHAIN_LOG}" | tail -n 1 | sed 's/^\[summary\] //' || true)"
  fi
  if [[ -n "${LAST_SUMMARY}" && -f "${LAST_SUMMARY}" ]]; then
    read -r LAST_PASS1 LAST_PASS8 LAST_MMLU LAST_SPELL <<< "$(read_summary_metrics "${LAST_SUMMARY}")"
  fi

  log "[candidate] ${label}: rc=${LAST_RC} chain=${LAST_CHAIN_LOG:-missing} summary=${LAST_SUMMARY:-missing} pass1=${LAST_PASS1}% pass8=${LAST_PASS8}% mmlu=${LAST_MMLU}% spell=${LAST_SPELL}%"

  if [[ "${rc}" -eq 0 && -n "${LAST_SUMMARY}" && -f "${LAST_SUMMARY}" ]]; then
    if full_confirm_hits_band "${LAST_SUMMARY}"; then
      return 0
    fi
    log "[candidate] ${label}: full confirm complete but below promotion band pass8>=${PROMOTE_GSM8K_MIN}% mmlu>=${PROMOTE_MMLU_MIN}%"
  fi
  return 1
}

{
  echo "RUN_BASE=${RUN_BASE}"
  echo "STAGE_A_TAG=${STAGE_A_TAG}"
  echo "STAGE_A_STEP=${STAGE_A_STEP}"
  echo "STAGE_A_MODEL_GROUP=${STAGE_A_MODEL_GROUP}"
  echo "FREEZE_LAYERS=${FREEZE_LAYERS}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
  echo "SEED=${SEED}"
  echo "EVAL_SEED=${EVAL_SEED}"
  echo "DETERMINISTIC=${DETERMINISTIC}"
  echo "QUICK_GATE_MMLU_MIN=${QUICK_GATE_MMLU_MIN}"
  echo "PROMOTE_GSM8K_MIN=${PROMOTE_GSM8K_MIN}"
  echo "PROMOTE_MMLU_MIN=${PROMOTE_MMLU_MIN}"
  echo "MASTER_LOG=${MASTER_LOG}"
  echo "DECISION_FILE=${DECISION_FILE}"
} > "${META_FILE}"

log "[info] meta: ${META_FILE}"
log "[plan] stage-B-only sweep from saved stage-A best: anchor quick gate -> soft50@0.02 -> soft25@0.015 if needed"

run_anchor_gate

if run_candidate soft50 50 0.02; then
  SOFT50_CHAIN_LOG="${LAST_CHAIN_LOG}"
  SOFT50_SUMMARY="${LAST_SUMMARY}"
  SOFT50_PASS1="${LAST_PASS1}"
  SOFT50_PASS8="${LAST_PASS8}"
  SOFT50_MMLU="${LAST_MMLU}"
  SOFT50_SPELL="${LAST_SPELL}"
  SOFT50_RC="${LAST_RC}"
  PROMOTED_SUMMARY="${LAST_SUMMARY}"
  FINAL_STATUS="promote_stageb_soft50"
  write_decision
  exit 0
else
  SOFT50_CHAIN_LOG="${LAST_CHAIN_LOG}"
  SOFT50_SUMMARY="${LAST_SUMMARY}"
  SOFT50_PASS1="${LAST_PASS1}"
  SOFT50_PASS8="${LAST_PASS8}"
  SOFT50_MMLU="${LAST_MMLU}"
  SOFT50_SPELL="${LAST_SPELL}"
  SOFT50_RC="${LAST_RC}"
fi

if run_candidate soft25 25 0.015; then
  SOFT25_CHAIN_LOG="${LAST_CHAIN_LOG}"
  SOFT25_SUMMARY="${LAST_SUMMARY}"
  SOFT25_PASS1="${LAST_PASS1}"
  SOFT25_PASS8="${LAST_PASS8}"
  SOFT25_MMLU="${LAST_MMLU}"
  SOFT25_SPELL="${LAST_SPELL}"
  SOFT25_RC="${LAST_RC}"
  PROMOTED_SUMMARY="${LAST_SUMMARY}"
  FINAL_STATUS="promote_stageb_soft25"
  write_decision
  exit 0
else
  SOFT25_CHAIN_LOG="${LAST_CHAIN_LOG}"
  SOFT25_SUMMARY="${LAST_SUMMARY}"
  SOFT25_PASS1="${LAST_PASS1}"
  SOFT25_PASS8="${LAST_PASS8}"
  SOFT25_MMLU="${LAST_MMLU}"
  SOFT25_SPELL="${LAST_SPELL}"
  SOFT25_RC="${LAST_RC}"
fi

FINAL_STATUS="hold_mixv2_provisional"
write_decision
exit 1
