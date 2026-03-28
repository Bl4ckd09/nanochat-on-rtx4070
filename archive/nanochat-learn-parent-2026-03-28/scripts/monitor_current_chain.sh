#!/usr/bin/env bash
set -euo pipefail

CHAIN_SESSION="${1:?usage: monitor_current_chain.sh <chain_session> <autopilot_session> <out_log> [interval_sec]}"
AUTOPILOT_SESSION="${2:?usage: monitor_current_chain.sh <chain_session> <autopilot_session> <out_log> [interval_sec]}"
OUT_LOG="${3:?usage: monitor_current_chain.sh <chain_session> <autopilot_session> <out_log> [interval_sec]}"
INTERVAL_SEC="${4:-300}"

mkdir -p "$(dirname "${OUT_LOG}")"

while true; do
  {
    echo "=== $(date '+%F %T') ==="
    echo "[chain]"
    tmux capture-pane -pt "${CHAIN_SESSION}:0" -S -80 2>/dev/null | tail -n 40 || echo "unavailable"
    echo
    echo "[autopilot]"
    tmux capture-pane -pt "${AUTOPILOT_SESSION}:0" -S -40 2>/dev/null | tail -n 20 || echo "unavailable"
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "unavailable"
    echo
  } >> "${OUT_LOG}"

  sleep "${INTERVAL_SEC}"
done
