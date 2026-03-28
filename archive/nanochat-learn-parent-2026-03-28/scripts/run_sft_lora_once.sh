#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d18_clean200k_safe_2026-02-13_1226}"
BASE_MODEL_STEP="${2:-200000}"
MAX_SEQ_LEN="${3:-1024}"
TOTAL_BATCH_SIZE="${4:-8192}"
NUM_ITERATIONS="${5:-1000}"

cd "${HOME}/nanochat-learn/nanochat"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
RUN="d18_clean200k_lora_prod_${TS}_s${MAX_SEQ_LEN}"
LOG="${HOME}/nanochat-learn/notes/sft_${RUN}.log"

echo "RUN=${RUN}"
echo "LOG=${LOG}"

export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

python -u -m scripts.chat_sft \
  --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}" \
  --output-tag "${RUN}" --run "${RUN}" \
  --device-type cuda --dtype bfloat16 \
  --lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0.0 --lora-lr 1e-4 \
  --weight-decay 0.0 \
  --init-lr-frac 0.25 --warmup-ratio 0.2 --warmdown-ratio 0.3 \
  --max-seq-len "${MAX_SEQ_LEN}" --device-batch-size 1 --total-batch-size "${TOTAL_BATCH_SIZE}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --eval-every 50 --eval-tokens 524288 \
  --max-grad-norm 1.0 \
  --keep-best-k 5 --no-save-optimizer \
  |& tee "${LOG}"
