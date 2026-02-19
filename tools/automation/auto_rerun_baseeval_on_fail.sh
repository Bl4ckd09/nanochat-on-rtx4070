#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: auto_rerun_baseeval_on_fail.sh <wait_pid> <base_log> <model_tag> <step> [eval_modes]}"
BASE_LOG="${2:?usage: auto_rerun_baseeval_on_fail.sh <wait_pid> <base_log> <model_tag> <step> [eval_modes]}"
MODEL_TAG="${3:?usage: auto_rerun_baseeval_on_fail.sh <wait_pid> <base_log> <model_tag> <step> [eval_modes]}"
STEP="${4:?usage: auto_rerun_baseeval_on_fail.sh <wait_pid> <base_log> <model_tag> <step> [eval_modes]}"
EVAL_MODES="${5:-core,bpb,sample}"

NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

TS="$(date +%F_%H%M)"
SAFE_TAG="${MODEL_TAG//\//_}"
WATCH_LOG="${NOTES_DIR}/${SAFE_TAG}_s${STEP}_auto_rerun_watch_${TS}.log"

exec > >(tee -a "${WATCH_LOG}") 2>&1

echo "[watch] waiting for PID ${WAIT_PID}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done

echo "[watch] PID ${WAIT_PID} ended at $(date -Is)"
echo "[watch] checking base log: ${BASE_LOG}"

if [[ -f "${BASE_LOG}" ]] && rg -q "CORE metric:|\\[done\\] core eval chain complete|Base model evaluation" "${BASE_LOG}"; then
  echo "[watch] base eval appears completed successfully. No rerun needed."
  exit 0
fi

if [[ -f "${BASE_LOG}" ]] && rg -q "exceeds max sequence length|Sequence length.*>" "${BASE_LOG}"; then
  echo "[watch] detected sequence overflow crash. Triggering resilient rerun."
else
  echo "[watch] base eval ended without success marker. Triggering resilient rerun."
fi

cd "${HOME}/nanochat-learn/nanochat"
source .venv/bin/activate
"${HOME}/nanochat-learn/scripts/run_base_eval_core_resilient.sh" "${MODEL_TAG}" "${STEP}" "${EVAL_MODES}"

echo "[watch] resilient rerun completed."
