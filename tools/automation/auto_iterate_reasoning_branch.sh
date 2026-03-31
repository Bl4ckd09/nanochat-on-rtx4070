#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d24_asp48_track}"
BASE_MODEL_STEP="${2:-820230}"
NUM_ITERATIONS="${3:-300}"
REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
RUN_SCRIPT="${REPO_DIR}/tools/automation/run_sft_adamw_control_nextbest.sh"
SWEEP_SCRIPT="${REPO_DIR}/tools/automation/run_reasoning_seed_sweep.sh"
BASELINE_SUMMARY_DEFAULT="${NOTES_DIR}/d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt"
TS="$(date +%F_%H%M)"
AUTO_NAME="reasoning_branch_auto_${TS}"
AUTO_LOG="${NOTES_DIR}/${AUTO_NAME}.log"
DECISION_MD="${NOTES_DIR}/${AUTO_NAME}_decision.md"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

log() {
  echo "$1" | tee -a "${AUTO_LOG}"
}

latest_chain_for_prefix() {
  local prefix="$1"
  ls -1t "${NOTES_DIR}/sft_${prefix}"_*_chain.log 2>/dev/null | head -n 1 || true
}

extract_last_summary() {
  local chain_log="$1"
  rg '^\[summary\]\s+' "${chain_log}" | tail -n 1 | sed 's/^\[summary\]\s\+//'
}

read_metrics() {
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

vals = [
    extract("GSM8K pass@1"),
    extract("GSM8K pass@8"),
    extract("MMLU"),
    extract("SpellingBee"),
]
print(*("nan" if math.isnan(v) else f"{v:.2f}" for v in vals))
PY
}

run_direct() {
  local label="$1"
  local preset="$2"
  local seed="$3"
  log "[start] direct label=${label} preset=${preset} seed=${seed}"
  set +e
  RUN_BASE_PREFIX="${label}" \
  FREEZE_LAYERS=20 \
  DATASET_PRESET="${preset}" \
  INIT_LR_FRAC=0.05 \
  QUICK_GATE_MMLU_MIN=27.0 \
  QUICK_GATE_REQUIRE_PASS1_NONZERO=0 \
  FULL_CONFIRM_MAX_PROBLEMS=1000 \
  ATTEMPT_ORDER='s768_gc:768:7680' \
  SEED="${seed}" \
  EVAL_SEED=42 \
  DETERMINISTIC=0 \
  WANDB_MODE=online \
  WANDB_PROJECT=nanochat-sft \
  EVAL_WANDB_PROJECT=nanochat-eval \
  bash "${RUN_SCRIPT}" "${BASE_MODEL_TAG}" "${BASE_MODEL_STEP}" "${NUM_ITERATIONS}" |& tee -a "${AUTO_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  local chain_log
  chain_log="$(latest_chain_for_prefix "${label}")"
  local summary=""
  local status="no_summary"
  local pass1="nan"
  local pass8="nan"
  local mmlu="nan"
  local spelling="nan"
  if [[ -n "${chain_log}" && -f "${chain_log}" ]]; then
    summary="$(extract_last_summary "${chain_log}")"
    if rg -q '^\[done\] completed at ' "${chain_log}"; then
      status="full_confirm_complete"
    elif rg -q '^\[gate\] quick gate failed ' "${chain_log}"; then
      status="quick_gate_failed"
    elif rg -q '^\[error\] non-OOM failure ' "${chain_log}"; then
      status="failed_non_oom"
    elif rg -q '^\[fail\] all partial full-tune attempts exhausted ' "${chain_log}"; then
      status="failed_attempts_exhausted"
    fi
  fi
  if [[ -n "${summary}" && -f "${summary}" ]]; then
    read -r pass1 pass8 mmlu spelling <<< "$(read_metrics "${summary}")"
  fi
  log "[end] direct label=${label} rc=${rc} status=${status} pass8=${pass8} mmlu=${mmlu} summary=${summary:-none}"
  DIRECT_CHAIN_LOG="${chain_log}"
  DIRECT_SUMMARY="${summary}"
  DIRECT_STATUS="${status}"
  DIRECT_PASS1="${pass1}"
  DIRECT_PASS8="${pass8}"
  DIRECT_MMLU="${mmlu}"
  DIRECT_SPELLING="${spelling}"
}

run_replication_sweep() {
  local label="$1"
  local preset="$2"
  local baseline_summary="$3"
  log "[start] replication_sweep label=${label} preset=${preset} baseline=${baseline_summary}"
  BASELINE_SUMMARY="${baseline_summary}" \
  PROMOTE_PASS_COUNT_MIN=1 \
  SEEDS=43,44 \
  DETERMINISTIC=0 \
  EVAL_SEED=42 \
  WANDB_MODE=online \
  WANDB_PROJECT=nanochat-sft \
  EVAL_WANDB_PROJECT=nanochat-eval \
  bash "${SWEEP_SCRIPT}" "${label}" "${preset}" "${BASE_MODEL_TAG}" "${BASE_MODEL_STEP}" "${NUM_ITERATIONS}" |& tee -a "${AUTO_LOG}"
  local rc=${PIPESTATUS[0]}
  local latest_decision
  latest_decision="$(ls -1t "${NOTES_DIR}/${label}"_*_decision.md 2>/dev/null | head -n 1 || true)"
  log "[end] replication_sweep label=${label} rc=${rc} decision=${latest_decision:-none}"
  SWEEP_DECISION="${latest_decision}"
  return ${rc}
}

write_final_decision() {
  python3 - "${DECISION_MD}" "$@" <<'PY'
import sys
from pathlib import Path

decision_path = Path(sys.argv[1])
lines = ["# reasoning branch auto decision", ""]
for item in sys.argv[2:]:
    lines.append(f"- {item}")
decision_path.write_text("\n".join(lines) + "\n")
print(decision_path)
PY
}

log "[info] auto_name=${AUTO_NAME}"
log "[info] baseline=${BASELINE_SUMMARY_DEFAULT}"

run_direct "d24_r32_adamw_partial_fr20_mixv3" "reasoning_focus_v3" 42
V3_STATUS="${DIRECT_STATUS}"
V3_SUMMARY="${DIRECT_SUMMARY}"
V3_PASS8="${DIRECT_PASS8}"
V3_MMLU="${DIRECT_MMLU}"
if [[ "${DIRECT_STATUS}" == "full_confirm_complete" ]] && python3 - "${DIRECT_PASS8}" "${DIRECT_MMLU}" <<'PY'
import sys
pass8=float(sys.argv[1])
mmlu=float(sys.argv[2])
raise SystemExit(0 if pass8 >= 4.0 and mmlu >= 27.0 else 1)
PY
then
  run_replication_sweep "reasoning_focus_v3_rep" "reasoning_focus_v3" "${DIRECT_SUMMARY}"
  DECISION_PATH="$(write_final_decision \
    "status: promoted_v3_candidate" \
    "direct_summary: ${DIRECT_SUMMARY}" \
    "replication_decision: ${SWEEP_DECISION:-none}")"
  log "[decision] ${DECISION_PATH}"
  exit 0
fi

log "[branch] v3 did not clear promotion thresholds; trying v4"
run_direct "d24_r32_adamw_partial_fr20_mixv4" "reasoning_focus_v4" 42
V4_STATUS="${DIRECT_STATUS}"
V4_SUMMARY="${DIRECT_SUMMARY}"
V4_PASS8="${DIRECT_PASS8}"
V4_MMLU="${DIRECT_MMLU}"
if [[ "${DIRECT_STATUS}" == "full_confirm_complete" ]] && python3 - "${DIRECT_PASS8}" "${DIRECT_MMLU}" <<'PY'
import sys
pass8=float(sys.argv[1])
mmlu=float(sys.argv[2])
raise SystemExit(0 if pass8 >= 4.0 and mmlu >= 27.0 else 1)
PY
then
  run_replication_sweep "reasoning_focus_v4_rep" "reasoning_focus_v4" "${DIRECT_SUMMARY}"
  DECISION_PATH="$(write_final_decision \
    "status: promoted_v4_candidate" \
    "direct_summary: ${DIRECT_SUMMARY}" \
    "replication_decision: ${SWEEP_DECISION:-none}")"
  log "[decision] ${DECISION_PATH}"
  exit 0
fi

DECISION_PATH="$(write_final_decision \
  "status: hold_mixv2_provisional" \
  "v3_status: ${V3_STATUS}" \
  "v3_summary: ${V3_SUMMARY:-none}" \
  "v3_pass8: ${V3_PASS8}" \
  "v3_mmlu: ${V3_MMLU}" \
  "v4_status: ${V4_STATUS}" \
  "v4_summary: ${V4_SUMMARY:-none}" \
  "v4_pass8: ${V4_PASS8}" \
  "v4_mmlu: ${V4_MMLU}" \
  "reason: neither v3 nor v4 produced a promotable direct result")"
log "[decision] ${DECISION_PATH}"
log "[done] auto iteration complete"
