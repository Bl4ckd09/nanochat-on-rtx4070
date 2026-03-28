#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: queue_eval_after_pid.sh <wait_pid> <model_group> <step> [max_problems]}"
MODEL_GROUP="${2:?usage: queue_eval_after_pid.sh <wait_pid> <model_group> <step> [max_problems]}"
STEP="${3:?usage: queue_eval_after_pid.sh <wait_pid> <model_group> <step> [max_problems]}"
MAX_PROBLEMS="${4:-1000}"

NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"
TS="$(date +%F_%H%M)"
SAFE_GROUP="${MODEL_GROUP//\//_}"
START_MARK="${NOTES_DIR}/queued_eval_started_${SAFE_GROUP}_s${STEP}_${TS}.txt"
DONE_MARK="${NOTES_DIR}/queued_eval_done_${SAFE_GROUP}_s${STEP}_${TS}.txt"

echo "[queue] waiting for PID ${WAIT_PID} to finish..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 60
done

date > "${START_MARK}"
echo "[queue] starting confirmation eval: ${MODEL_GROUP} step=${STEP} max=${MAX_PROBLEMS}"
"${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${MODEL_GROUP}" "${STEP}" "${MAX_PROBLEMS}"
date > "${DONE_MARK}"
echo "[queue] done"
echo "[queue] start_mark=${START_MARK}"
echo "[queue] done_mark=${DONE_MARK}"
