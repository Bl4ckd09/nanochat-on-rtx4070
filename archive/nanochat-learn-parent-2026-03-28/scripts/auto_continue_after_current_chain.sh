#!/usr/bin/env bash
set -euo pipefail

# Waits for the currently running eval chain to finish, then automatically
# launches the next stage based on a simple promotion gate.
#
# Gate logic:
# - Parse best score from latest recovery sweep v2 results.
# - Compare against control 1k-confirm score + MIN_IMPROVEMENT.
# - If gate passes: launch a promoted SFT run using winner hyperparams.
# - Else: launch base pretraining to ratio 40.

BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_asp48_track}"
BASE_MODEL_STEP="${BASE_MODEL_STEP:-820230}"
CHAIN_PATTERN="${CHAIN_PATTERN:-run_eval_then_next_plan.sh d24_asp48_track 820230 200}"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-1.0}"
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-60}"
CONTROL_SUMMARY="${CONTROL_SUMMARY:-$HOME/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-19_1310_s1536_gc_best_s1000_confirm_summary_1000_2026-03-19_1451.txt}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

TS="$(date +%F_%H%M)"
LOG_FILE="${NOTES_DIR}/auto_continue_${TS}.log"

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

log "Auto-continue watcher started."
log "Waiting for current chain to finish: pattern='${CHAIN_PATTERN}'"
while pgrep -f "${CHAIN_PATTERN}" >/dev/null; do
  sleep "${CHECK_EVERY_SEC}"
done
log "Detected current chain finished."

if [[ ! -f "${CONTROL_SUMMARY}" ]]; then
  log "WARN control summary missing: ${CONTROL_SUMMARY}"
  CONTROL_SCORE="22.5000"
else
  c_gsm="$(extract_pct_from_summary 'GSM8K pass@8:' "${CONTROL_SUMMARY}")"
  c_mmlu="$(extract_pct_from_summary 'MMLU:' "${CONTROL_SUMMARY}")"
  c_spell="$(extract_pct_from_summary 'SpellingBee:' "${CONTROL_SUMMARY}")"
  if [[ -z "${c_gsm}" || -z "${c_mmlu}" || -z "${c_spell}" ]]; then
    log "WARN could not parse control summary, using fallback score 22.5000"
    CONTROL_SCORE="22.5000"
  else
    CONTROL_SCORE="$(score_of "${c_gsm}" "${c_mmlu}" "${c_spell}")"
  fi
fi
THRESHOLD_SCORE="$(awk -v c="${CONTROL_SCORE}" -v d="${MIN_IMPROVEMENT}" 'BEGIN { printf "%.4f", (c+d) }')"
log "Control score=${CONTROL_SCORE}, min improvement=${MIN_IMPROVEMENT}, threshold=${THRESHOLD_SCORE}"

LATEST_SWEEP="$(ls -1t "${NOTES_DIR}"/d24_sft_recovery_v2_*_results.tsv 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_SWEEP}" ]]; then
  log "No sweep-v2 results found. Launching base r40."
  cd "${REPO_DIR}"
  source .venv/bin/activate
  RUN_FINAL_CORE=1 bash "${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh" "${BASE_MODEL_TAG}" 40 |& tee -a "${LOG_FILE}"
  exit 0
fi
log "Using latest sweep results: ${LATEST_SWEEP}"

BEST_LINE="$(tail -n +2 "${LATEST_SWEEP}" | awk -F '\t' '$8 != "-1" {print $0}' | sort -t $'\t' -k8,8nr | head -n 1 || true)"
if [[ -z "${BEST_LINE}" ]]; then
  log "No valid winner in sweep results. Launching base r40."
  cd "${REPO_DIR}"
  source .venv/bin/activate
  RUN_FINAL_CORE=1 bash "${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh" "${BASE_MODEL_TAG}" 40 |& tee -a "${LOG_FILE}"
  exit 0
fi

IFS=$'\t' read -r BEST_LABEL BEST_OUT BEST_SEQ BEST_LR BEST_GSM BEST_MMLU BEST_SPELL BEST_SCORE <<< "${BEST_LINE}"
log "Best sweep candidate label=${BEST_LABEL} out=${BEST_OUT} seq=${BEST_SEQ} lr=${BEST_LR} score=${BEST_SCORE}"

PASS_GATE="$(awk -v b="${BEST_SCORE}" -v t="${THRESHOLD_SCORE}" 'BEGIN { if (b >= t) print "1"; else print "0"; }')"
if [[ "${PASS_GATE}" != "1" ]]; then
  log "Gate failed (best_score < threshold). Launching base r40."
  cd "${REPO_DIR}"
  source .venv/bin/activate
  RUN_FINAL_CORE=1 bash "${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh" "${BASE_MODEL_TAG}" 40 |& tee -a "${LOG_FILE}"
  exit 0
fi

log "Gate passed. Running 1k confirm on winner first."
cd "${REPO_DIR}"
source .venv/bin/activate
export CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}"
CONFIRM_LOG="${NOTES_DIR}/auto_continue_confirm_${BEST_OUT}_$(date +%F_%H%M).log"
CAT_BATCH_SIZE="${CAT_BATCH_SIZE}" "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${BEST_OUT}/best" 300 1000 |& tee "${CONFIRM_LOG}"

SUMMARY_1K="$(ls -1t "${NOTES_DIR}/${BEST_OUT}_best_s300_confirm_summary_1000_"*.txt 2>/dev/null | head -n 1 || true)"
if [[ -z "${SUMMARY_1K}" ]]; then
  log "Could not find 1k summary for winner; conservatively launching base r40."
  RUN_FINAL_CORE=1 bash "${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh" "${BASE_MODEL_TAG}" 40 |& tee -a "${LOG_FILE}"
  exit 0
fi

w_gsm="$(extract_pct_from_summary 'GSM8K pass@8:' "${SUMMARY_1K}")"
w_mmlu="$(extract_pct_from_summary 'MMLU:' "${SUMMARY_1K}")"
w_spell="$(extract_pct_from_summary 'SpellingBee:' "${SUMMARY_1K}")"
if [[ -z "${w_gsm}" || -z "${w_mmlu}" || -z "${w_spell}" ]]; then
  log "Could not parse winner 1k summary; conservatively launching base r40."
  RUN_FINAL_CORE=1 bash "${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh" "${BASE_MODEL_TAG}" 40 |& tee -a "${LOG_FILE}"
  exit 0
fi
WINNER_1K_SCORE="$(score_of "${w_gsm}" "${w_mmlu}" "${w_spell}")"
log "Winner 1k confirm score=${WINNER_1K_SCORE} (summary=${SUMMARY_1K})"

PASS_GATE_1K="$(awk -v b="${WINNER_1K_SCORE}" -v t="${THRESHOLD_SCORE}" 'BEGIN { if (b >= t) print "1"; else print "0"; }')"
if [[ "${PASS_GATE_1K}" != "1" ]]; then
  log "1k gate failed. Launching base r40."
  RUN_FINAL_CORE=1 bash "${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh" "${BASE_MODEL_TAG}" 40 |& tee -a "${LOG_FILE}"
  exit 0
fi

# Promote winner settings into a longer SFT run.
PROMOTE_TAG="d24_sft_promote_$(date +%F_%H%M)_s${BEST_SEQ}_lr${BEST_LR}"
log "1k gate passed. Launching promoted SFT run: ${PROMOTE_TAG}"

PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}" \
python -u -m scripts.chat_sft \
  --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}" \
  --output-tag "${PROMOTE_TAG}" --run "${PROMOTE_TAG}" \
  --device-type cuda --dtype bfloat16 \
  --lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0.0 --lora-lr "${BEST_LR}" \
  --weight-decay 0.0 \
  --init-lr-frac 0.25 --warmup-ratio 0.2 --warmdown-ratio 0.3 \
  --max-seq-len "${BEST_SEQ}" --device-batch-size 1 --total-batch-size 7680 \
  --num-iterations 1000 --eval-every 50 --eval-tokens 524288 \
  --max-grad-norm 1.0 --keep-best-k 5 --no-save-optimizer \
  --gradient-checkpoint |& tee -a "${LOG_FILE}"

log "Promoted SFT run complete."
