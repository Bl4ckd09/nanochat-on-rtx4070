#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d24_asp48_track}"
BASE_MODEL_STEP="${2:-820230}"
NUM_ITERATIONS="${3:-1000}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
RUN_BASE="d24_r32_lora_nextbest_${TS}"
OUT_BASE="${RUN_BASE}"
MASTER_LOG="${NOTES_DIR}/sft_${RUN_BASE}_chain.log"
META_FILE="${NOTES_DIR}/sft_${RUN_BASE}_meta.env"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}"
export EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-1000}"
export RUN_PASS1="${RUN_PASS1:-0}"

{
  echo "RUN_BASE=${RUN_BASE}"
  echo "BASE_MODEL_TAG=${BASE_MODEL_TAG}"
  echo "BASE_MODEL_STEP=${BASE_MODEL_STEP}"
  echo "NUM_ITERATIONS=${NUM_ITERATIONS}"
  echo "MASTER_LOG=${MASTER_LOG}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
  echo "CAT_BATCH_SIZE=${CAT_BATCH_SIZE}"
  echo "EVAL_MAX_PROBLEMS=${EVAL_MAX_PROBLEMS}"
} > "${META_FILE}"

echo "[info] meta: ${META_FILE}" | tee -a "${MASTER_LOG}"
echo "[info] base: ${BASE_MODEL_TAG} step=${BASE_MODEL_STEP}" | tee -a "${MASTER_LOG}"

OOM_RE='out of memory|cuda out of memory|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory|CUDA driver error: out of memory'

resolve_eval_checkpoint() {
  local output_tag="$1"
  python3 - "${output_tag}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_tag = sys.argv[1]
root = Path.home() / ".cache" / "nanochat" / "chatsft_checkpoints" / output_tag

def best_record(path):
    records = []
    if not path.exists():
        return None
    for meta_path in path.glob("meta_*.json"):
        try:
            data = json.loads(meta_path.read_text())
        except Exception:
            continue
        step = data.get("step")
        if step is None:
            try:
                step = int(meta_path.stem.split("_")[1])
            except Exception:
                continue
        try:
            val_bpb = float(data.get("val_bpb", math.inf))
        except Exception:
            val_bpb = math.inf
        records.append((val_bpb, int(step)))
    if not records:
        return None
    return min(records)

best_dir = root / "best"
record = best_record(best_dir)
if record is not None:
    val_bpb, step = record
    print(f"best {step} {val_bpb}")
    raise SystemExit(0)

record = best_record(root)
if record is not None:
    val_bpb, step = record
    print(f"main {step} {val_bpb}")
    raise SystemExit(0)

raise SystemExit(1)
PY
}

run_confirm_eval() {
  local label="$1"
  local output_tag="$2"
  local run_name="$3"
  local eval_log="${NOTES_DIR}/eval_${run_name}_confirm.log"
  local resolved
  if ! resolved="$(resolve_eval_checkpoint "${output_tag}")"; then
    echo "[error] unable to resolve checkpoint for eval: ${output_tag}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  local checkpoint_kind best_step best_val model_group
  read -r checkpoint_kind best_step best_val <<< "${resolved}"
  if [[ "${checkpoint_kind}" == "best" ]]; then
    model_group="${output_tag}/best"
  else
    model_group="${output_tag}"
  fi

  {
    echo "[eval] label=${label} model_group=${model_group} step=${best_step} val_bpb=${best_val} max_problems=${EVAL_MAX_PROBLEMS} cat_batch=${CAT_BATCH_SIZE}"
    echo "[eval_log] ${eval_log}"
  } | tee -a "${MASTER_LOG}"

  set +e
  EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  EVAL_WANDB_RUN_PREFIX="${run_name}_confirm_s${best_step}" \
  CAT_BATCH_SIZE="${CAT_BATCH_SIZE}" \
  RUN_PASS1="${RUN_PASS1}" \
  "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${model_group}" "${best_step}" "${EVAL_MAX_PROBLEMS}" \
    |& tee "${eval_log}" | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] confirm eval failed for ${run_name} (rc=${rc})" | tee -a "${MASTER_LOG}"
    return "${rc}"
  fi

  echo "[done] confirm eval finished for ${run_name}" | tee -a "${MASTER_LOG}"
  return 0
}

run_attempt() {
  local label="$1"
  local max_seq_len="$2"
  local total_batch_size="$3"
  local use_gc="$4"

  local output_tag="${OUT_BASE}_${label}"
  local run_name="${RUN_BASE}_${label}"
  local log_file="${NOTES_DIR}/sft_${run_name}.log"

  {
    echo "[start] run=${run_name} output=${output_tag} max_seq_len=${max_seq_len} total_batch_size=${total_batch_size} gradient_checkpoint=${use_gc}"
    echo "[log] ${log_file}"
  } | tee -a "${MASTER_LOG}"

  local tokens_per_fwdbwd=$((max_seq_len))
  if (( total_batch_size % tokens_per_fwdbwd != 0 )); then
    echo "[error] invalid batch geometry for ${run_name}: total_batch_size=${total_batch_size} is not divisible by max_seq_len=${max_seq_len}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  local cmd=(
    python -u -m scripts.chat_sft
    --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}"
    --output-tag "${output_tag}" --run "${run_name}"
    --device-type cuda --dtype bfloat16
    --lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0.0 --lora-lr 1e-4
    --weight-decay 0.0
    --init-lr-frac 0.25 --warmup-ratio 0.2 --warmdown-ratio 0.3
    --max-seq-len "${max_seq_len}" --device-batch-size 1 --total-batch-size "${total_batch_size}"
    --num-iterations "${NUM_ITERATIONS}"
    --eval-every 50 --eval-tokens 524288
    --max-grad-norm 1.0
    --keep-best-k 5 --no-save-optimizer
  )

  if [ "${use_gc}" = "1" ]; then
    cmd+=(--gradient-checkpoint)
  fi

  set +e
  "${cmd[@]}" |& tee "${log_file}" | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e

  echo "[end] run=${run_name} rc=${rc}" | tee -a "${MASTER_LOG}"

  if [ "${rc}" -eq 0 ]; then
    if run_confirm_eval "${label}" "${output_tag}" "${run_name}"; then
      return 0
    fi
    return 1
  fi

  if rg -qi "${OOM_RE}" "${log_file}"; then
    echo "[warn] OOM detected in ${run_name}" | tee -a "${MASTER_LOG}"
    return 2
  fi

  echo "[error] non-OOM failure in ${run_name}; check ${log_file}" | tee -a "${MASTER_LOG}"
  return 1
}

# Promoted r32 SFT progression:
# 1) 1536 with gradient checkpointing
# 2) 1280 with gradient checkpointing
# 3) 1024 with gradient checkpointing
if run_attempt "s1536_gc" 1536 7680 1; then
  echo "[done] completed at s1536_gc" | tee -a "${MASTER_LOG}"
  exit 0
else
  rc=$?
  if [ "$rc" -eq 1 ]; then
    exit 1
  fi
fi

if run_attempt "s1280_gc" 1280 7680 1; then
  echo "[done] completed at s1280_gc" | tee -a "${MASTER_LOG}"
  exit 0
else
  rc=$?
  if [ "$rc" -eq 1 ]; then
    exit 1
  fi
fi

if run_attempt "s1024_gc" 1024 8192 1; then
  echo "[done] completed at s1024_gc" | tee -a "${MASTER_LOG}"
  exit 0
else
  rc=$?
  if [ "$rc" -eq 1 ]; then
    exit 1
  fi
fi

echo "[error] all fallback attempts OOMed" | tee -a "${MASTER_LOG}"
exit 1
