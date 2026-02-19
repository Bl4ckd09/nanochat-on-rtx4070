#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?usage: run_base_eval_core_resilient.sh <model_tag> [step] [eval_modes]}"
STEP="${2:-200000}"
EVAL_MODES="${3:-core,bpb,sample}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

# Once fallback is triggered, keep this fixed for comparability across runs.
POLICY_LOCK="${NOTES_DIR}/base_core_overflow_policy_fixed.lock"

TS="$(date +%F_%H%M)"
LOG_BASE="${NOTES_DIR}/${TAG}_s${STEP}_base_eval_${TS}.log"
LOG_FIXED="${NOTES_DIR}/${TAG}_s${STEP}_base_eval_skip5120_${TS}.log"

cd "${REPO_DIR}"
source .venv/bin/activate

run_eval() {
  local log_file="$1"
  shift
  PYTHONUNBUFFERED=1 python -u -m scripts.base_eval \
    --model-tag "${TAG}" \
    --step "${STEP}" \
    --eval "${EVAL_MODES}" \
    --device-type cuda \
    "$@" |& tee "${log_file}"
}

run_fixed_policy() {
  local log_file="$1"
  echo "[run] fixed CORE overflow policy: skip + max_seq_len=5120"
  run_eval "${log_file}" \
    --core-overflow-policy skip \
    --core-max-seq-len 5120
}

if [[ -f "${POLICY_LOCK}" ]]; then
  echo "[policy] lock present: ${POLICY_LOCK}"
  run_fixed_policy "${LOG_FIXED}"
  echo "[done] base_eval complete (fixed policy)"
  echo "[log] ${LOG_FIXED}"
  exit 0
fi

echo "[run] standard base_eval first (policy=error)"
set +e
run_eval "${LOG_BASE}"
RC=$?
set -e

if [[ ${RC} -eq 0 ]]; then
  echo "[done] base_eval complete (standard policy)"
  echo "[log] ${LOG_BASE}"
  exit 0
fi

if rg -q "exceeds max sequence length|Sequence length.*>" "${LOG_BASE}"; then
  {
    echo "timestamp=$(date -Is)"
    echo "reason=core_sequence_overflow"
    echo "fixed_policy=--core-overflow-policy skip --core-max-seq-len 5120"
    echo "trigger_log=${LOG_BASE}"
    echo "model_tag=${TAG}"
    echo "step=${STEP}"
    echo "eval_modes=${EVAL_MODES}"
  } > "${POLICY_LOCK}"
  echo "[retry] detected CORE overflow, rerunning with fixed policy"
  run_fixed_policy "${LOG_FIXED}"
  echo "[done] base_eval complete after overflow retry"
  echo "[log] ${LOG_FIXED}"
  exit 0
fi

echo "[fail] base_eval failed for a non-overflow reason (exit=${RC})"
echo "[log] ${LOG_BASE}"
exit "${RC}"
