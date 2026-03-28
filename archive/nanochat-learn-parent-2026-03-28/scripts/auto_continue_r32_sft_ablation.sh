#!/usr/bin/env bash
set -euo pipefail

CURRENT_RUN_BASE="${1:?usage: auto_continue_r32_sft_ablation.sh <current_run_base> [base_model_tag] [base_model_step]}"
BASE_MODEL_TAG="${2:-d24_asp48_track}"
BASE_MODEL_STEP="${3:-820230}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
RUNTIME_SCRIPTS_DIR="${HOME}/nanochat-learn/scripts"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

REFERENCE_SCORE="${REFERENCE_SCORE:-22.0000}"
REFERENCE_LABEL="${REFERENCE_LABEL:-r32_s1536_gc_lr1e4_ref}"
PROMOTE_DELTA="${PROMOTE_DELTA:-1.0}"
HARD_FAIL_DELTA="${HARD_FAIL_DELTA:-2.0}"
FOLLOWUP_LR="${FOLLOWUP_LR:-5e-5}"
FOLLOWUP_NUM_ITERATIONS="${FOLLOWUP_NUM_ITERATIONS:-500}"
FOLLOWUP_PREFIX="${FOLLOWUP_PREFIX:-d24_r32_lora_maxoom_lr5e5}"
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-30}"

TS="$(date +%F_%H%M)"
LOG_FILE="${NOTES_DIR}/auto_continue_r32_sft_ablation_${TS}.log"
DECISION_FILE="${NOTES_DIR}/auto_continue_r32_sft_ablation_${TS}_decision.md"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

extract_pct_from_summary() {
  local pattern="$1"
  local file="$2"
  rg "^${pattern}" "${file}" | sed -E 's/.*= ([0-9.]+)%.*/\1/' | tail -n 1
}

score_of() {
  local gsm="$1"
  local mmlu="$2"
  local spelling="$3"
  awk -v g="$gsm" -v m="$mmlu" -v s="$spelling" 'BEGIN { printf "%.4f", (3.0*g + 1.0*m + 0.5*s) }'
}

latest_complete_summary() {
  local run_base="$1"
  local summary
  summary="$(ls -1t "${NOTES_DIR}/${run_base}"_*_confirm_summary_1000_*.txt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${summary}" ]] && rg -q '^\[done\] confirmation eval complete' "${summary}"; then
    echo "${summary}"
  fi
}

latest_chain_log() {
  local run_base="$1"
  ls -1t "${NOTES_DIR}/sft_${run_base}"*_chain.log 2>/dev/null | head -n 1 || true
}

write_decision() {
  cat > "${DECISION_FILE}"
  log "Wrote decision note: ${DECISION_FILE}"
}

parse_summary_metrics() {
  local summary="$1"
  local gsm mmlu spelling score
  gsm="$(extract_pct_from_summary 'GSM8K pass@8:' "${summary}")"
  mmlu="$(extract_pct_from_summary 'MMLU:' "${summary}")"
  spelling="$(extract_pct_from_summary 'SpellingBee:' "${summary}")"
  if [[ -z "${gsm}" || -z "${mmlu}" || -z "${spelling}" ]]; then
    return 1
  fi
  score="$(score_of "${gsm}" "${mmlu}" "${spelling}")"
  printf '%s %s %s %s\n' "${gsm}" "${mmlu}" "${spelling}" "${score}"
}

wait_for_run_completion() {
  local run_base="$1"
  local summary chain_log
  while true; do
    summary="$(latest_complete_summary "${run_base}")"
    if [[ -n "${summary}" ]]; then
      echo "${summary}"
      return 0
    fi

    chain_log="$(latest_chain_log "${run_base}")"
    if [[ -n "${chain_log}" ]]; then
      if rg -q '\[error\] (confirm eval failed|non-OOM failure|all fallback attempts OOMed|unable to resolve checkpoint)' "${chain_log}"; then
        log "Detected failure in chain log: ${chain_log}"
        return 1
      fi
    fi

    sleep "${CHECK_EVERY_SEC}"
  done
}

launch_followup_and_wait() {
  local launch_mark summary
  launch_mark="$(date +%s)"
  log "Launching follow-up bracket: lr=${FOLLOWUP_LR}, iters=${FOLLOWUP_NUM_ITERATIONS}, prefix=${FOLLOWUP_PREFIX}"
  (
    cd "${REPO_DIR}"
    source .venv/bin/activate
    WANDB_MODE="${WANDB_MODE:-online}" \
    WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}" \
    EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}" \
    CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}" \
    RUN_PASS1="${RUN_PASS1:-0}" \
    LORA_LR="${FOLLOWUP_LR}" \
    RUN_BASE_PREFIX="${FOLLOWUP_PREFIX}" \
    "${RUNTIME_SCRIPTS_DIR}/run_sft_lora_nextbest.sh" "${BASE_MODEL_TAG}" "${BASE_MODEL_STEP}" "${FOLLOWUP_NUM_ITERATIONS}"
  ) |& tee -a "${LOG_FILE}"

  summary="$(ls -1t "${NOTES_DIR}/${FOLLOWUP_PREFIX}"_*_confirm_summary_1000_*.txt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${summary}" ]] && [[ "$(stat -c '%Y' "${summary}")" -ge "${launch_mark}" ]] && rg -q '^\[done\] confirmation eval complete' "${summary}"; then
    echo "${summary}"
    return 0
  fi
  return 1
}

log "Autopilot started for current run base: ${CURRENT_RUN_BASE}"
log "Reference score=${REFERENCE_SCORE} (${REFERENCE_LABEL}), promote_delta=${PROMOTE_DELTA}, hard_fail_delta=${HARD_FAIL_DELTA}"

CURRENT_SUMMARY=""
if ! CURRENT_SUMMARY="$(wait_for_run_completion "${CURRENT_RUN_BASE}")"; then
  write_decision <<EOF
# R32 SFT Autopilot Decision

- Current run base: \`${CURRENT_RUN_BASE}\`
- Outcome: failed before a complete confirm summary was produced
- Action: stop automatic follow-up and review the chain log manually
EOF
  exit 1
fi

log "Current summary complete: ${CURRENT_SUMMARY}"
if ! read -r CURRENT_GSM CURRENT_MMLU CURRENT_SPELL CURRENT_SCORE < <(parse_summary_metrics "${CURRENT_SUMMARY}"); then
  write_decision <<EOF
# R32 SFT Autopilot Decision

- Current run base: \`${CURRENT_RUN_BASE}\`
- Summary: \`${CURRENT_SUMMARY}\`
- Outcome: could not parse full metrics from summary
- Action: stop automatic follow-up and review the summary/logs manually
EOF
  exit 1
fi

log "Current score=${CURRENT_SCORE} (gsm8k=${CURRENT_GSM}, mmlu=${CURRENT_MMLU}, spelling=${CURRENT_SPELL})"

HARD_FAIL_THRESHOLD="$(awk -v ref="${REFERENCE_SCORE}" -v d="${HARD_FAIL_DELTA}" 'BEGIN { printf "%.4f", (ref-d) }')"
PROMOTE_THRESHOLD="$(awk -v ref="${REFERENCE_SCORE}" -v d="${PROMOTE_DELTA}" 'BEGIN { printf "%.4f", (ref+d) }')"

if awk -v cur="${CURRENT_SCORE}" -v thr="${HARD_FAIL_THRESHOLD}" 'BEGIN { exit !(cur < thr) }'; then
  write_decision <<EOF
# R32 SFT Autopilot Decision

- Current run base: \`${CURRENT_RUN_BASE}\`
- Current summary: \`${CURRENT_SUMMARY}\`
- Current score: \`${CURRENT_SCORE}\`
- Reference score: \`${REFERENCE_SCORE}\` (\`${REFERENCE_LABEL}\`)
- Hard-fail threshold: \`${HARD_FAIL_THRESHOLD}\`
- Outcome: current run is catastrophically below the bracket reference
- Action: do not auto-launch the 5e-5 follow-up; change the SFT recipe/data mix instead
EOF
  exit 0
fi

FOLLOWUP_SUMMARY=""
FOLLOWUP_STATUS="not_run"
if FOLLOWUP_SUMMARY="$(launch_followup_and_wait)"; then
  FOLLOWUP_STATUS="completed"
  log "Follow-up summary complete: ${FOLLOWUP_SUMMARY}"
else
  FOLLOWUP_STATUS="failed"
  log "Follow-up run did not produce a complete confirm summary"
fi

BEST_LABEL="${CURRENT_RUN_BASE}"
BEST_SUMMARY="${CURRENT_SUMMARY}"
BEST_SCORE="${CURRENT_SCORE}"
BEST_GSM="${CURRENT_GSM}"
BEST_MMLU="${CURRENT_MMLU}"
BEST_SPELL="${CURRENT_SPELL}"

FOLLOWUP_GSM=""
FOLLOWUP_MMLU=""
FOLLOWUP_SPELL=""
FOLLOWUP_SCORE=""
if [[ "${FOLLOWUP_STATUS}" == "completed" ]]; then
  if read -r FOLLOWUP_GSM FOLLOWUP_MMLU FOLLOWUP_SPELL FOLLOWUP_SCORE < <(parse_summary_metrics "${FOLLOWUP_SUMMARY}"); then
    log "Follow-up score=${FOLLOWUP_SCORE} (gsm8k=${FOLLOWUP_GSM}, mmlu=${FOLLOWUP_MMLU}, spelling=${FOLLOWUP_SPELL})"
    if awk -v a="${FOLLOWUP_SCORE}" -v b="${BEST_SCORE}" 'BEGIN { exit !(a > b) }'; then
      BEST_LABEL="${FOLLOWUP_PREFIX}"
      BEST_SUMMARY="${FOLLOWUP_SUMMARY}"
      BEST_SCORE="${FOLLOWUP_SCORE}"
      BEST_GSM="${FOLLOWUP_GSM}"
      BEST_MMLU="${FOLLOWUP_MMLU}"
      BEST_SPELL="${FOLLOWUP_SPELL}"
    fi
  else
    FOLLOWUP_STATUS="failed_parse"
    log "Could not parse follow-up summary metrics"
  fi
fi

if awk -v best="${BEST_SCORE}" -v thr="${PROMOTE_THRESHOLD}" 'BEGIN { exit !(best >= thr) }'; then
  FINAL_ACTION="promote_candidate"
  FINAL_NOTE="Best bracket candidate cleared the promotion threshold. Promote this checkpoint candidate and only then decide on deeper recipe changes."
else
  FINAL_ACTION="hold_recipe_change"
  FINAL_NOTE="Neither bracket candidate cleared the promotion threshold. Hold promotion and change the SFT recipe/data mix before more LoRA iterations."
fi

write_decision <<EOF
# R32 SFT Autopilot Decision

- Current run base: \`${CURRENT_RUN_BASE}\`
- Current summary: \`${CURRENT_SUMMARY}\`
- Current score: \`${CURRENT_SCORE}\` (GSM8K pass@8 \`${CURRENT_GSM}\`, MMLU \`${CURRENT_MMLU}\`, SpellingBee \`${CURRENT_SPELL}\`)
- Reference score: \`${REFERENCE_SCORE}\` (\`${REFERENCE_LABEL}\`)
- Promote threshold: \`${PROMOTE_THRESHOLD}\`
- Hard-fail threshold: \`${HARD_FAIL_THRESHOLD}\`
- Follow-up status: \`${FOLLOWUP_STATUS}\`
- Follow-up summary: \`${FOLLOWUP_SUMMARY}\`
- Follow-up score: \`${FOLLOWUP_SCORE}\`
- Best label: \`${BEST_LABEL}\`
- Best summary: \`${BEST_SUMMARY}\`
- Best score: \`${BEST_SCORE}\` (GSM8K pass@8 \`${BEST_GSM}\`, MMLU \`${BEST_MMLU}\`, SpellingBee \`${BEST_SPELL}\`)
- Final action: \`${FINAL_ACTION}\`

${FINAL_NOTE}
EOF

log "Autopilot finished. Final action=${FINAL_ACTION}, best_label=${BEST_LABEL}, best_score=${BEST_SCORE}"
