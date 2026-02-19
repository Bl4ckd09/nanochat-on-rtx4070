#!/usr/bin/env bash
set -euo pipefail

CHAT_GROUP="${1:?usage: run_additional_eval_suite.sh <chat_group> <chat_step> <base_tag> <base_step>}"
CHAT_STEP="${2:?usage: run_additional_eval_suite.sh <chat_group> <chat_step> <base_tag> <base_step>}"
BASE_TAG="${3:?usage: run_additional_eval_suite.sh <chat_group> <chat_step> <base_tag> <base_step>}"
BASE_STEP="${4:?usage: run_additional_eval_suite.sh <chat_group> <chat_step> <base_tag> <base_step>}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
SAFE_CHAT="${CHAT_GROUP//\//_}"
SUMMARY_LOG="${NOTES_DIR}/${SAFE_CHAT}_s${CHAT_STEP}_extra_eval_suite_${TS}.txt"
ARC_LOG="${NOTES_DIR}/${SAFE_CHAT}_s${CHAT_STEP}_arc1k_${TS}.log"
CHATCORE_LOG="${NOTES_DIR}/${SAFE_CHAT}_s${CHAT_STEP}_chatcore_all200_${TS}.log"
HUMANEVAL_LOG="${NOTES_DIR}/${SAFE_CHAT}_s${CHAT_STEP}_humaneval_pass10_${TS}.log"
BASE_LOG="${NOTES_DIR}/${BASE_TAG}_s${BASE_STEP}_base_core_bpb_sample_${TS}.log"

{
  echo "chat_group=${CHAT_GROUP}"
  echo "chat_step=${CHAT_STEP}"
  echo "base_tag=${BASE_TAG}"
  echo "base_step=${BASE_STEP}"
  echo "timestamp=${TS}"
  echo "arc_log=${ARC_LOG}"
  echo "chatcore_log=${CHATCORE_LOG}"
  echo "humaneval_log=${HUMANEVAL_LOG}"
  echo "base_log=${BASE_LOG}"
  echo
} | tee "${SUMMARY_LOG}"

echo "[1/4] ARC-only quick signal (-x 1000)" | tee -a "${SUMMARY_LOG}"
WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${CHAT_GROUP}" -s "${CHAT_STEP}" -a "ARC-Easy|ARC-Challenge" \
  -x 1000 -b 16 --device-type cuda \
  |& tee "${ARC_LOG}"

echo "[2/4] Full ChatCORE bundle (-x 200)" | tee -a "${SUMMARY_LOG}"
WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${CHAT_GROUP}" -s "${CHAT_STEP}" \
  -x 200 --device-type cuda \
  |& tee "${CHATCORE_LOG}"

echo "[3/4] HumanEval pass@10" | tee -a "${SUMMARY_LOG}"
WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${CHAT_GROUP}" -s "${CHAT_STEP}" -a HumanEval \
  -t 0.8 -n 10 -m 512 -k 50 --device-type cuda \
  |& tee "${HUMANEVAL_LOG}"

echo "[4/4] Base-side CORE/BPB/sample (resilient overflow handling)" | tee -a "${SUMMARY_LOG}"
WANDB_MODE="${WANDB_MODE:-online}" "${HOME}/nanochat-learn/scripts/run_base_eval_core_resilient.sh" \
  "${BASE_TAG}" "${BASE_STEP}" "core,bpb,sample" \
  |& tee "${BASE_LOG}"

echo "[done] additional eval suite complete" | tee -a "${SUMMARY_LOG}"
