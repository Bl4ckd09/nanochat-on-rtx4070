#!/usr/bin/env bash
set -euo pipefail

MODEL_GROUP="${1:?usage: run_chat_eval_confirm_1k.sh <model_group> <step> [max_problems]}"
STEP="${2:?usage: run_chat_eval_confirm_1k.sh <model_group> <step> [max_problems]}"
MAX_PROBLEMS="${3:-1000}"
RUN_PASS1="${RUN_PASS1:-0}" # optional, set RUN_PASS1=1 to include GSM8K pass@1

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
SAFE_GROUP="${MODEL_GROUP//\//_}"
LOG_PASS1="${NOTES_DIR}/${SAFE_GROUP}_s${STEP}_gsm8k_pass1_${MAX_PROBLEMS}_${TS}.log"
LOG_PASS8="${NOTES_DIR}/${SAFE_GROUP}_s${STEP}_gsm8k_pass8_${MAX_PROBLEMS}_${TS}.log"
LOG_MMLU="${NOTES_DIR}/${SAFE_GROUP}_s${STEP}_mmlu_${MAX_PROBLEMS}_${TS}.log"
LOG_SPELLING="${NOTES_DIR}/${SAFE_GROUP}_s${STEP}_spellingbee_${MAX_PROBLEMS}_${TS}.log"
SUMMARY="${NOTES_DIR}/${SAFE_GROUP}_s${STEP}_confirm_summary_${MAX_PROBLEMS}_${TS}.txt"

extract_final_counts() {
  local log_file="$1"
  local line
  line="$(rg "Final:\\s+[0-9]+/[0-9]+" "${log_file}" | tail -n 1 || true)"
  if [[ "${line}" =~ Final:\ ([0-9]+)/([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
  fi
}

wilson_line() {
  local k="$1"
  local n="$2"
  python - "${k}" "${n}" <<'PY'
import sys, math
k, n = int(sys.argv[1]), int(sys.argv[2])
z = 1.96
p = k / n
den = 1 + z*z/n
center = (p + z*z/(2*n)) / den
margin = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
print(f"{k}/{n} = {p:.2%} | Wilson95% [{center-margin:.2%}, {center+margin:.2%}]")
PY
}

summarize_one() {
  local label="$1"
  local log_file="$2"
  local counts
  counts="$(extract_final_counts "${log_file}")"
  if [[ -z "${counts}" ]]; then
    echo "${label}: FAILED to parse Final: k/n from ${log_file}" | tee -a "${SUMMARY}"
    return 1
  fi
  local k n
  read -r k n <<< "${counts}"
  local ci
  ci="$(wilson_line "${k}" "${n}")"
  echo "${label}: ${ci}" | tee -a "${SUMMARY}"
}

{
  echo "model_group=${MODEL_GROUP}"
  echo "step=${STEP}"
  echo "max_problems=${MAX_PROBLEMS}"
  echo "timestamp=${TS}"
  echo "logs:"
  echo "  pass1: ${LOG_PASS1}"
  echo "  pass8: ${LOG_PASS8}"
  echo "  mmlu: ${LOG_MMLU}"
  echo "  spellingbee: ${LOG_SPELLING}"
  echo
} | tee "${SUMMARY}"

if [[ "${RUN_PASS1}" == "1" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
    -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a GSM8K \
    -t 0 -n 1 -m 1024 -k 50 \
    -x "${MAX_PROBLEMS}" --device-type cuda \
    |& tee "${LOG_PASS1}"
  summarize_one "GSM8K pass@1" "${LOG_PASS1}"
fi

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a GSM8K \
  -t 0.7 -n 8 -m 1024 -k 50 \
  -x "${MAX_PROBLEMS}" --device-type cuda \
  |& tee "${LOG_PASS8}"
summarize_one "GSM8K pass@8" "${LOG_PASS8}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a MMLU \
  -x "${MAX_PROBLEMS}" --device-type cuda \
  |& tee "${LOG_MMLU}"
summarize_one "MMLU" "${LOG_MMLU}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a SpellingBee \
  -x "${MAX_PROBLEMS}" --device-type cuda \
  |& tee "${LOG_SPELLING}"
summarize_one "SpellingBee" "${LOG_SPELLING}"

echo "[done] confirmation eval complete" | tee -a "${SUMMARY}"
echo "summary=${SUMMARY}"
