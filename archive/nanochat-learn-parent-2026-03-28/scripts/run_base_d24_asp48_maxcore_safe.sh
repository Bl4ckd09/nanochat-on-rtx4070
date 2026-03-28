#!/usr/bin/env bash
set -euo pipefail

# OOM-safe long-run launcher for d24 asp48 on 12GB GPUs.
# Primary: low-VRAM profile ("option 3"): uncompiled model/optimizer + cautious off + stacked-off Muon, bs=4.
# Fallbacks: auto-retry with smaller device batch sizes from latest checkpoint if present.
# Post-train: optional one-shot final CORE/BPB eval at the target end step (default: enabled).

MODEL_TAG="${1:-d24_asp48_gc_bs8_s2048_$(date +%F_%H%M)}"
TARGET_RATIO="${2:-40}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16384}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
SAVE_EVERY="${SAVE_EVERY:-8192}"   # ~5h on recent bs1 throughput; safer against interruptions without exhausting disk
EVAL_EVERY="${EVAL_EVERY:-20000}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
RUN_FINAL_CORE="${RUN_FINAL_CORE:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
REPORT_PATH="${REPORT_PATH:-${REPO_DIR}/report_v2.md}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-nanochat}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DEVICE_BATCH_SEQUENCE="${DEVICE_BATCH_SEQUENCE:-4,2,1}"

# Allow manual LR override without script edits (useful for post-resume stabilization)
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
MATRIX_LR="${MATRIX_LR:-0.02}"
SCALAR_LR="${SCALAR_LR:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.2}"
ADAM_BETA1="${ADAM_BETA1:-0.8}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.8}"
FINAL_LR_FRAC="${FINAL_LR_FRAC:-0.0}"

OOM_RE='out of memory|cuda out of memory|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory'

checkpoint_dir="${REPO_DIR}/base_checkpoints/${MODEL_TAG}"
eval_script="${HOME}/nanochat-learn/scripts/run_base_eval_core_resilient.sh"

find_last_step() {
  if compgen -G "${checkpoint_dir}/model_*.pt" > /dev/null; then
    ls -1 "${checkpoint_dir}"/model_*.pt \
      | sed -E 's/.*model_([0-9]+)\.pt/\1/' \
      | sort -n \
      | tail -1
  fi
}

run_train() {
  local device_bs="$1"
  local resume_step="${2:-}"
  local log_file="$3"
  local compile_optim="${4:-1}"
  local compile_model="${5:-1}"
  local muon_cautious="${6:-1}"
  local muon_stacked="${7:-1}"
  local run_name="${MODEL_TAG}_r${TARGET_RATIO}_bs${device_bs}_co${compile_optim}_cm${compile_model}_mc${muon_cautious}_ms${muon_stacked}_$(date +%F_%H%M)"

  local resume_args=()
  if [[ -n "${resume_step}" ]]; then
    resume_args=(--resume-from-step "${resume_step}")
  fi

  (
    cd "${REPO_DIR}"
    source .venv/bin/activate

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE}" WANDB_PROJECT="${WANDB_PROJECT}" WANDB_ENTITY="${WANDB_ENTITY}" NANOCHAT_COMPILE_OPTIM="${compile_optim}" NANOCHAT_COMPILE_MODEL="${compile_model}" NANOCHAT_MUON_CAUTIOUS="${muon_cautious}" NANOCHAT_MUON_STACKED="${muon_stacked}" \
    python -u -m scripts.base_train_gc \
      --run "${run_name}" \
      --model-tag "${MODEL_TAG}" \
      --depth 24 --aspect-ratio 48 --head-dim 128 --window-pattern L \
      --gradient-checkpoint \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --device-batch-size "${device_bs}" \
      --total-batch-size "${TOTAL_BATCH_SIZE}" \
      --target-param-data-ratio "${TARGET_RATIO}" \
      --embedding-lr "${EMBEDDING_LR}" --unembedding-lr "${UNEMBEDDING_LR}" --matrix-lr "${MATRIX_LR}" --scalar-lr "${SCALAR_LR}" \
      --weight-decay "${WEIGHT_DECAY}" --adam-beta1 "${ADAM_BETA1}" --adam-beta2 "${ADAM_BETA2}" \
      --warmup-ratio "${WARMUP_RATIO}" --warmdown-ratio "${WARMDOWN_RATIO}" --final-lr-frac "${FINAL_LR_FRAC}" \
      --eval-every "${EVAL_EVERY}" --eval-tokens "${EVAL_TOKENS}" \
      --core-metric-every -1 --sample-every -1 --save-every "${SAVE_EVERY}" \
      "${resume_args[@]}" |& tee "${log_file}"
  )
}

target_step_for_ratio() {
  python3 - <<PY
num_scaling = 419_958_144
target_ratio = float("${TARGET_RATIO}")
total_batch = int("${TOTAL_BATCH_SIZE}")
target_tokens = int(target_ratio * num_scaling)
print(target_tokens // total_batch)
PY
}

init_report_file() {
  if [[ -f "${REPORT_PATH}" ]]; then
    return 0
  fi
  cat > "${REPORT_PATH}" <<'EOF'
# Report v2

Automated CORE tracking for d24 asp48 long runs.

| Date (local) | Model Tag | Target Ratio | Eval Step | CORE | Val bpb | Eval Log | Status |
|---|---|---:|---:|---:|---:|---|---|
EOF
}

append_report_entry() {
  local ratio="$1"
  local eval_step="$2"
  local core="$3"
  local val_bpb="$4"
  local eval_log="$5"
  local status="$6"
  local now
  now="$(date '+%F %H:%M')"
  init_report_file
  printf '| %s | `%s` | %s | %s | %s | %s | `%s` | %s |\n' \
    "${now}" "${MODEL_TAG}" "${ratio}" "${eval_step}" "${core}" "${val_bpb}" "${eval_log}" "${status}" >> "${REPORT_PATH}"
  echo "[report] appended result -> ${REPORT_PATH}"
}

run_final_core_eval() {
  local final_step="$1"
  if [[ "${RUN_FINAL_CORE}" != "1" ]]; then
    echo "[skip] RUN_FINAL_CORE=${RUN_FINAL_CORE}, skipping final CORE eval"
    return 0
  fi
  if [[ ! -x "${eval_script}" ]]; then
    echo "[warn] final CORE eval script not executable: ${eval_script}"
    return 0
  fi
  echo "[start] final CORE/BPB eval at step=${final_step}"
  local eval_wrap_log="${NOTES_DIR}/${MODEL_TAG}_final_eval_r${TARGET_RATIO}_s${final_step}_$(date +%F_%H%M%S).log"
  set +e
  "${eval_script}" "${MODEL_TAG}" "${final_step}" "core,bpb" |& tee "${eval_wrap_log}"
  local eval_rc=$?
  set -e

  local eval_log
  eval_log="$(rg -o '^\[log\] .+$' "${eval_wrap_log}" | tail -1 | sed 's/^\[log\] //')"
  if [[ -z "${eval_log}" ]]; then
    eval_log="${eval_wrap_log}"
  fi

  local core val_bpb status
  core="$(rg -o 'CORE metric: [0-9]+(\\.[0-9]+)?' "${eval_log}" 2>/dev/null | tail -1 | awk '{print $3}' || true)"
  val_bpb="$(rg -o 'val bpb: [0-9]+(\\.[0-9]+)?' "${eval_log}" 2>/dev/null | tail -1 | awk '{print $3}' || true)"
  [[ -z "${core}" ]] && core="NA"
  [[ -z "${val_bpb}" ]] && val_bpb="NA"
  if [[ ${eval_rc} -eq 0 ]]; then
    status="ok"
  else
    status="eval_failed(${eval_rc})"
  fi
  append_report_entry "${TARGET_RATIO}" "${final_step}" "${core}" "${val_bpb}" "${eval_log}" "${status}"
  return "${eval_rc}"
}

ts="$(date +%F_%H%M)"
target_step="$(target_step_for_ratio)"

resume_from=""
if [[ "${AUTO_RESUME}" == "1" ]]; then
  existing_step="$(find_last_step || true)"
  if [[ -n "${existing_step}" ]]; then
    if (( existing_step >= target_step )); then
      echo "[info] existing checkpoint step ${existing_step} already reached target step ${target_step}"
      if [[ -f "${checkpoint_dir}/model_$(printf "%06d" "${target_step}").pt" ]]; then
        run_final_core_eval "${target_step}"
      else
        echo "[info] exact target checkpoint missing, evaluating last available step ${existing_step}"
        run_final_core_eval "${existing_step}"
      fi
      echo "[done] nothing to train"
      exit 0
    fi
    resume_from="${existing_step}"
    echo "[resume] found step ${existing_step}, continuing to target step ${target_step}"
  fi
fi

read -r -a device_bs_sequence <<< "$(tr ',' ' ' <<< "${DEVICE_BATCH_SEQUENCE}")"
if [[ ${#device_bs_sequence[@]} -eq 0 ]]; then
  echo "[fail] DEVICE_BATCH_SEQUENCE is empty"
  exit 1
fi

echo "[start] model_tag=${MODEL_TAG} ratio=${TARGET_RATIO} profile=3 batch_sequence=${DEVICE_BATCH_SEQUENCE}"
echo "[wandb] mode=${WANDB_MODE} project=${WANDB_PROJECT} entity=${WANDB_ENTITY:-default}"
echo "[hparams] lr(e/u/m/s)=${EMBEDDING_LR}/${UNEMBEDDING_LR}/${MATRIX_LR}/${SCALAR_LR} wd=${WEIGHT_DECAY} betas=${ADAM_BETA1},${ADAM_BETA2} warmup/warmdown/final=${WARMUP_RATIO}/${WARMDOWN_RATIO}/${FINAL_LR_FRAC} save_every=${SAVE_EVERY}"

attempt_idx=0
for device_bs in "${device_bs_sequence[@]}"; do
  current_resume="${resume_from}"
  log_suffix=""
  if (( attempt_idx > 0 )); then
    current_resume="$(find_last_step || true)"
    log_suffix="_retry${attempt_idx}"
    if [[ -n "${current_resume}" ]]; then
      echo "[fallback${attempt_idx}] OOM detected, retrying profile3 with bs=${device_bs} from step ${current_resume}"
    else
      echo "[fallback${attempt_idx}] OOM detected before checkpoint, retrying profile3 with bs=${device_bs} from scratch"
    fi
  fi

  log_file="${NOTES_DIR}/${MODEL_TAG}_train_r${TARGET_RATIO}_p3_bs${device_bs}${log_suffix}_${ts}.log"
  echo "[log] ${log_file}"
  if run_train "${device_bs}" "${current_resume}" "${log_file}" "0" "0" "0" "0"; then
    run_final_core_eval "${target_step}"
    echo "[done] completed with profile3 bs=${device_bs}"
    exit 0
  fi

  if ! rg -qi "${OOM_RE}" "${log_file}"; then
    echo "[fail] profile3 bs=${device_bs} failed for non-OOM reason"
    echo "[log] ${log_file}"
    exit 1
  fi

  ((attempt_idx += 1))
done

echo "[fail] all profile3 attempts failed for batch_sequence=${DEVICE_BATCH_SEQUENCE}"
exit 1
