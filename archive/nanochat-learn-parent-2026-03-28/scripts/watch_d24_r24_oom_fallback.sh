#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-d24_asp48_track}"
RATIO="${2:-24}"
TRAIN_SESSION="${TRAIN_SESSION:-d24_r24}"

NOTES_DIR="${HOME}/nanochat-learn/notes"
LAUNCHER="${HOME}/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh"
WATCH_LOG="${NOTES_DIR}/${TAG}_r${RATIO}_oom_watch.log"
STATE_FILE="/tmp/${TAG}_r${RATIO}_oom_watch_last_seq.txt"
OOM_RE='out of memory|cuda out of memory|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory'

mkdir -p "${NOTES_DIR}"

extract_seq_len() {
  local log_file="$1"
  rg -o 'Tokens / micro-batch / rank: [0-9]+ x [0-9]+ =' "${log_file}" \
    | tail -1 \
    | sed -E 's/.* x ([0-9]+) =/\1/' || true
}

next_seq_len() {
  local current_seq="$1"
  case "${current_seq}" in
    2048) echo 1024 ;;
    1024) echo 512 ;;
    *) echo "" ;;
  esac
}

while true; do
  if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    sleep 30
    continue
  fi

  latest_log="$(ls -1t "${NOTES_DIR}/${TAG}_train_r${RATIO}"*.log 2>/dev/null | head -1 || true)"
  if [[ -z "${latest_log}" ]]; then
    echo "[$(date '+%F %T')] no log found, waiting..." | tee -a "${WATCH_LOG}"
    sleep 30
    continue
  fi

  if ! rg -qi "${OOM_RE}" "${latest_log}"; then
    echo "[$(date '+%F %T')] train session exited without OOM; watcher stopping" | tee -a "${WATCH_LOG}"
    exit 0
  fi

  cur_seq="$(extract_seq_len "${latest_log}")"
  if [[ -z "${cur_seq}" ]]; then
    echo "[$(date '+%F %T')] could not parse seq_len from ${latest_log}; watcher stopping" | tee -a "${WATCH_LOG}"
    exit 1
  fi

  nxt_seq="$(next_seq_len "${cur_seq}")"
  if [[ -z "${nxt_seq}" ]]; then
    echo "[$(date '+%F %T')] no lower seq fallback for seq_len=${cur_seq}; watcher stopping" | tee -a "${WATCH_LOG}"
    exit 1
  fi

  last_seq="$(cat "${STATE_FILE}" 2>/dev/null || true)"
  if [[ "${last_seq}" == "${nxt_seq}" ]]; then
    sleep 30
    continue
  fi
  echo "${nxt_seq}" > "${STATE_FILE}"

  echo "[$(date '+%F %T')] OOM detected; restarting ${TRAIN_SESSION} with MAX_SEQ_LEN=${nxt_seq}" | tee -a "${WATCH_LOG}"
  tmux new-session -d -s "${TRAIN_SESSION}" "bash -lc 'MAX_SEQ_LEN=${nxt_seq} ${LAUNCHER} ${TAG} ${RATIO}; exec bash'"
  sleep 10
done
