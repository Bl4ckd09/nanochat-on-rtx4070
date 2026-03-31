#!/usr/bin/env bash
set -euo pipefail

OUT_LOG="${1:-${HOME}/nanochat-learn/notes/nanochat_state_watch.log}"
INTERVAL_SEC="${2:-60}"
NOTES_DIR="${HOME}/nanochat-learn/notes"
PROC_PATTERN='[s]cripts\.chat_(sft|eval)|run_curriculum_v1|run_sft_adamw_control_nextbest|run_mixv2_seed_sweep|run_reasoning_seed_sweep|auto_iterate_reasoning_branch|run_base_.*|auto_decide_reasoning_focus_v2|run_stageb_booster_sweep'

mkdir -p "$(dirname "${OUT_LOG}")"

snapshot() {
  local proc_dump
  proc_dump="$(ps -eo pid,etimes,cmd | rg "${PROC_PATTERN}" || true)"
  local gpu_line
  gpu_line="$(nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true)"
  local state="idle"
  if [[ -n "${proc_dump}" ]]; then
    state="active"
  fi

  echo "=== $(date '+%F %T %Z') ==="
  echo "[state] ${state}"
  echo
  echo "[gpu]"
  if [[ -n "${gpu_line}" ]]; then
    echo "${gpu_line}"
  else
    echo "unavailable"
  fi
  echo

  echo "[processes]"
  if [[ -n "${proc_dump}" ]]; then
    echo "${proc_dump}"
  else
    echo "none"
  fi
  echo

  echo "[tmux]"
  tmux ls 2>/dev/null || echo "no tmux server"
  echo

  echo "[recent_summaries]"
  ls -1t "${NOTES_DIR}"/*confirm_summary*.txt 2>/dev/null | head -n 5 || echo "none"
  echo

  echo "[recent_chain_logs]"
  ls -1t "${NOTES_DIR}"/sft_*_chain.log "${NOTES_DIR}"/eval_*_full.log 2>/dev/null | head -n 5 || echo "none"
  echo

  latest_chain="$(ls -1t "${NOTES_DIR}"/sft_*_chain.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_chain}" && -f "${latest_chain}" ]]; then
    echo "[latest_chain_tail] ${latest_chain}"
    tail -n 40 "${latest_chain}"
    echo
  fi

  latest_summary="$(ls -1t "${NOTES_DIR}"/*confirm_summary*.txt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_summary}" && -f "${latest_summary}" ]]; then
    echo "[latest_summary_head] ${latest_summary}"
    sed -n '1,30p' "${latest_summary}"
    echo
  fi
}

while true; do
  snapshot | tee -a "${OUT_LOG}"
  sleep "${INTERVAL_SEC}"
done
