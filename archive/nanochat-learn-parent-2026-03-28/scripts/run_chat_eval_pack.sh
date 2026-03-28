#!/usr/bin/env bash
set -euo pipefail

MODEL_GROUP="${1:?usage: run_chat_eval_pack.sh <model_group> <step>}"
STEP="${2:?usage: run_chat_eval_pack.sh <model_group> <step>}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
LOG_PASS1="${NOTES_DIR}/${MODEL_GROUP//\//_}_s${STEP}_gsm8k_pass1_${TS}.log"
LOG_PASS8="${NOTES_DIR}/${MODEL_GROUP//\//_}_s${STEP}_gsm8k_pass8_${TS}.log"
LOG_BENCH="${NOTES_DIR}/${MODEL_GROUP//\//_}_s${STEP}_benchmarks_${TS}.log"

echo "[eval] model_group=${MODEL_GROUP} step=${STEP}"
echo "[log] pass1: ${LOG_PASS1}"
echo "[log] pass8: ${LOG_PASS8}"
echo "[log] bench: ${LOG_BENCH}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a GSM8K \
  -t 0 -n 1 -m 1024 -k 50 \
  -x 300 --device-type cuda \
  |& tee "${LOG_PASS1}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a GSM8K \
  -t 0.7 -n 8 -m 1024 -k 50 \
  -x 300 --device-type cuda \
  |& tee "${LOG_PASS8}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${MODEL_GROUP}" -s "${STEP}" -a "MMLU|SpellingBee" \
  -x 200 --device-type cuda \
  |& tee "${LOG_BENCH}"

echo "[done] eval pack complete"
