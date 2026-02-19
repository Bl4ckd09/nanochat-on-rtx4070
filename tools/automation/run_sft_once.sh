#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d18_clean200k_safe_2026-02-13_1226}"
BASE_MODEL_STEP="${2:-200000}"
MAX_SEQ_LEN="${3:-1280}"
TOTAL_BATCH_SIZE="${4:-7680}"

cd "${HOME}/nanochat-learn/nanochat"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
OUT="d18_clean200k_sftmax_${TS}_s${MAX_SEQ_LEN}"
LOG="${HOME}/nanochat-learn/notes/sft_${OUT}.log"

echo "OUT=${OUT}"
echo "LOG=${LOG}"

export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

python -u -m scripts.chat_sft \
  --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}" \
  --output-tag "${OUT}" --run "${OUT}" \
  --device-type cuda --dtype bfloat16 \
  --adamw-only --weight-decay 0.0 \
  --init-lr-frac 0.25 --warmup-ratio 0.2 --warmdown-ratio 0.3 \
  --max-seq-len "${MAX_SEQ_LEN}" --device-batch-size 1 --total-batch-size "${TOTAL_BATCH_SIZE}" \
  --num-iterations 1000 \
  --eval-every 25 --eval-tokens 524288 \
  --gradient-checkpoint --max-grad-norm 1.0 \
  --keep-best-k 10 --no-save-optimizer \
  |& tee "${LOG}"
