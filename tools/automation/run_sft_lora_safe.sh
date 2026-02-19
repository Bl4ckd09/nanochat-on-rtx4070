#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d18_clean200k_safe_2026-02-13_1226}"
BASE_MODEL_STEP="${2:-200000}"
REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
RUN_BASE="d18_clean200k_lora_sftmax_${TS}"
OUT_BASE="${RUN_BASE}"

LOG_2048="${NOTES_DIR}/sft_${OUT_BASE}_s2048.log"
LOG_1536="${NOTES_DIR}/sft_${OUT_BASE}_s1536_retry.log"
LOG_1280="${NOTES_DIR}/sft_${OUT_BASE}_s1280_retry.log"
LOG_1024="${NOTES_DIR}/sft_${OUT_BASE}_s1024_retry.log"
MASTER_LOG="${NOTES_DIR}/sft_${OUT_BASE}_chain.log"
META_FILE="${NOTES_DIR}/sft_${OUT_BASE}_meta.env"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

{
  echo "RUN_BASE=${RUN_BASE}"
  echo "BASE_MODEL_TAG=${BASE_MODEL_TAG}"
  echo "BASE_MODEL_STEP=${BASE_MODEL_STEP}"
  echo "LOG_2048=${LOG_2048}"
  echo "LOG_1536=${LOG_1536}"
  echo "LOG_1280=${LOG_1280}"
  echo "LOG_1024=${LOG_1024}"
  echo "MASTER_LOG=${MASTER_LOG}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
} > "${META_FILE}"

echo "[info] meta: ${META_FILE}" | tee -a "${MASTER_LOG}"
echo "[info] base: ${BASE_MODEL_TAG} step=${BASE_MODEL_STEP}" | tee -a "${MASTER_LOG}"

run_sft_lora() {
  local max_seq_len="$1"
  local total_batch_size="$2"
  local output_tag="$3"
  local run_name="$4"
  local log_file="$5"

  echo "[start] run=${run_name} output=${output_tag} max_seq_len=${max_seq_len} total_batch_size=${total_batch_size}" | tee -a "${MASTER_LOG}"
  python -u -m scripts.chat_sft \
    --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}" \
    --output-tag "${output_tag}" --run "${run_name}" \
    --device-type cuda --dtype bfloat16 \
    --lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0.0 --lora-lr 1e-4 \
    --weight-decay 0.0 \
    --init-lr-frac 0.25 --warmup-ratio 0.2 --warmdown-ratio 0.3 \
    --max-seq-len "${max_seq_len}" --device-batch-size 1 --total-batch-size "${total_batch_size}" \
    --num-iterations 1000 \
    --eval-every 25 --eval-tokens 524288 \
    --max-grad-norm 1.0 \
    --keep-best-k 10 --no-save-optimizer \
    |& tee "${log_file}" | tee -a "${MASTER_LOG}"
}

if run_sft_lora 2048 8192 "${OUT_BASE}" "${RUN_BASE}" "${LOG_2048}"; then
  echo "[done] 2048 LoRA run finished successfully" | tee -a "${MASTER_LOG}"
  exit 0
fi

OOM_RE='out of memory|cuda out of memory|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory'
if rg -qi "${OOM_RE}" "${LOG_2048}"; then
  echo "[warn] OOM detected, retrying LoRA with --max-seq-len 1536" | tee -a "${MASTER_LOG}"
  RETRY_OUT="${OUT_BASE}_s1536"
  RETRY_RUN="${RUN_BASE}_s1536"
  if run_sft_lora 1536 7680 "${RETRY_OUT}" "${RETRY_RUN}" "${LOG_1536}"; then
    echo "[done] 1536 LoRA retry finished" | tee -a "${MASTER_LOG}"
    exit 0
  fi
  if rg -qi "${OOM_RE}" "${LOG_1536}"; then
    echo "[warn] OOM detected, retrying LoRA with --max-seq-len 1280" | tee -a "${MASTER_LOG}"
    RETRY_OUT="${OUT_BASE}_s1280"
    RETRY_RUN="${RUN_BASE}_s1280"
    if run_sft_lora 1280 7680 "${RETRY_OUT}" "${RETRY_RUN}" "${LOG_1280}"; then
      echo "[done] 1280 LoRA retry finished" | tee -a "${MASTER_LOG}"
      exit 0
    fi
    if rg -qi "${OOM_RE}" "${LOG_1280}"; then
      echo "[warn] OOM detected, retrying LoRA with --max-seq-len 1024" | tee -a "${MASTER_LOG}"
      RETRY_OUT="${OUT_BASE}_s1024"
      RETRY_RUN="${RUN_BASE}_s1024"
      run_sft_lora 1024 8192 "${RETRY_OUT}" "${RETRY_RUN}" "${LOG_1024}"
      echo "[done] 1024 LoRA retry finished" | tee -a "${MASTER_LOG}"
      exit 0
    fi
  fi
fi

echo "[error] first LoRA run failed without OOM signature; check ${LOG_2048}" | tee -a "${MASTER_LOG}"
exit 1
