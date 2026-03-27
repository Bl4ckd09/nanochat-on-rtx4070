#!/usr/bin/env bash
set -euo pipefail

MODEL_GROUP="${1:?usage: run_chat_eval_confirm_1k.sh <model_group> <step> [max_problems]}"
STEP="${2:?usage: run_chat_eval_confirm_1k.sh <model_group> <step> [max_problems]}"
MAX_PROBLEMS="${3:-1000}"
RUN_PASS1="${RUN_PASS1:-0}" # optional, set RUN_PASS1=1 to include GSM8K pass@1
CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}" # safer default for 12GB GPUs during categorical eval (e.g., MMLU)
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}"
EVAL_WANDB_ENTITY="${EVAL_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
SKIP_PASS1="${SKIP_PASS1:-0}"
SKIP_PASS8="${SKIP_PASS8:-0}"
SKIP_MMLU="${SKIP_MMLU:-0}"
SKIP_SPELLING="${SKIP_SPELLING:-0}"
OOM_RE='out of memory|cuda out of memory|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory|CUDA driver error: out of memory'

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
SAFE_GROUP="${MODEL_GROUP//\//_}"
EVAL_WANDB_RUN_PREFIX="${EVAL_WANDB_RUN_PREFIX:-${SAFE_GROUP}_s${STEP}_confirm_${MAX_PROBLEMS}_${TS}}"
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

run_eval_logged() {
  local log_file="$1"
  shift

  set +e
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_ENTITY="${EVAL_WANDB_ENTITY}" \
  .venv/bin/python -m scripts.chat_eval "$@" |& tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

run_stage() {
  local summary_label="$1"
  local primary_log="$2"
  local retry_mode="$3"
  shift 3
  local -a primary_args=("$@")
  local rc=0

  if run_eval_logged "${primary_log}" "${primary_args[@]}"; then
    summarize_one "${summary_label}" "${primary_log}" || true
    return 0
  fi
  rc=$?

  if [[ "${retry_mode}" == "cpu_on_oom" ]] && rg -qi "${OOM_RE}" "${primary_log}"; then
    local retry_log="${primary_log%.log}_retry_cpu.log"
    echo "[warn] ${summary_label} hit CUDA OOM; retrying on CPU" | tee -a "${SUMMARY}"
    local -a retry_args=("${primary_args[@]}")
    local idx
    for idx in "${!retry_args[@]}"; do
      if [[ "${retry_args[$idx]}" == "--device-type" ]] && (( idx + 1 < ${#retry_args[@]} )); then
        retry_args[$((idx + 1))]="cpu"
      fi
      if [[ "${retry_args[$idx]}" == "--run" ]] && (( idx + 1 < ${#retry_args[@]} )); then
        retry_args[$((idx + 1))]="${retry_args[$((idx + 1))]}_cpu"
      fi
    done
    retry_args+=(--dtype float32)
    if run_eval_logged "${retry_log}" "${retry_args[@]}"; then
      summarize_one "${summary_label} (cpu retry)" "${retry_log}" || true
      return 0
    fi
    rc=$?
    echo "${summary_label}: FAILED after CPU retry, see ${retry_log}" | tee -a "${SUMMARY}"
    return "${rc}"
  fi

  echo "${summary_label}: FAILED, see ${primary_log}" | tee -a "${SUMMARY}"
  return "${rc}"
}

{
  echo "model_group=${MODEL_GROUP}"
  echo "step=${STEP}"
  echo "max_problems=${MAX_PROBLEMS}"
  echo "timestamp=${TS}"
  echo "eval_wandb_project=${EVAL_WANDB_PROJECT}"
  echo "eval_wandb_run_prefix=${EVAL_WANDB_RUN_PREFIX}"
  echo "skip_pass1=${SKIP_PASS1}"
  echo "skip_pass8=${SKIP_PASS8}"
  echo "skip_mmlu=${SKIP_MMLU}"
  echo "skip_spelling=${SKIP_SPELLING}"
  echo "logs:"
  echo "  pass1: ${LOG_PASS1}"
  echo "  pass8: ${LOG_PASS8}"
  echo "  mmlu: ${LOG_MMLU}"
  echo "  spellingbee: ${LOG_SPELLING}"
  echo
} | tee "${SUMMARY}"

had_failure=0

if [[ "${RUN_PASS1}" == "1" && "${SKIP_PASS1}" != "1" ]]; then
  if ! run_stage \
    "GSM8K pass@1" \
    "${LOG_PASS1}" \
    "none" \
    -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a GSM8K \
    --run "${EVAL_WANDB_RUN_PREFIX}_gsm8k_pass1" \
    -t 0 -n 1 -m 1024 -k 50 \
    -x "${MAX_PROBLEMS}" --device-type cuda; then
    had_failure=1
  fi
elif [[ "${RUN_PASS1}" == "1" ]]; then
  echo "GSM8K pass@1: SKIPPED by SKIP_PASS1=1" | tee -a "${SUMMARY}"
fi

if [[ "${SKIP_PASS8}" != "1" ]]; then
  if ! run_stage \
    "GSM8K pass@8" \
    "${LOG_PASS8}" \
    "none" \
    -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a GSM8K \
    --run "${EVAL_WANDB_RUN_PREFIX}_gsm8k_pass8" \
    -t 0.7 -n 8 -m 1024 -k 50 \
    -x "${MAX_PROBLEMS}" --device-type cuda; then
    had_failure=1
  fi
else
  echo "GSM8K pass@8: SKIPPED by SKIP_PASS8=1" | tee -a "${SUMMARY}"
fi

if [[ "${SKIP_MMLU}" != "1" ]]; then
  if ! run_stage \
    "MMLU" \
    "${LOG_MMLU}" \
    "cpu_on_oom" \
    -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a MMLU \
    --run "${EVAL_WANDB_RUN_PREFIX}_mmlu" \
    -b "${CAT_BATCH_SIZE}" \
    -x "${MAX_PROBLEMS}" --device-type cuda; then
    had_failure=1
  fi
else
  echo "MMLU: SKIPPED by SKIP_MMLU=1" | tee -a "${SUMMARY}"
fi

if [[ "${SKIP_SPELLING}" != "1" ]]; then
  if ! run_stage \
    "SpellingBee" \
    "${LOG_SPELLING}" \
    "none" \
    -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a SpellingBee \
    --run "${EVAL_WANDB_RUN_PREFIX}_spellingbee" \
    -x "${MAX_PROBLEMS}" --device-type cuda; then
    had_failure=1
  fi
else
  echo "SpellingBee: SKIPPED by SKIP_SPELLING=1" | tee -a "${SUMMARY}"
fi

if [[ "${had_failure}" -eq 0 ]]; then
  echo "[done] confirmation eval complete" | tee -a "${SUMMARY}"
  echo "summary=${SUMMARY}"
  exit 0
fi

echo "[warn] confirmation eval finished with failures" | tee -a "${SUMMARY}"
echo "summary=${SUMMARY}"
exit 1
