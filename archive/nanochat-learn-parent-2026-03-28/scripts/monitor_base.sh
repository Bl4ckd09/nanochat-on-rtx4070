#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <base_train_log_path> [interval_seconds]"
  exit 1
fi

LOG_PATH="$1"
INTERVAL="${2:-30}"

if [ ! -f "$LOG_PATH" ]; then
  echo "Log file not found: $LOG_PATH"
  exit 1
fi

echo "Monitoring: $LOG_PATH"
echo "Refresh interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop."

while true; do
  clear 2>/dev/null || true
  echo "=== $(date '+%F %T') ==="
  echo

  echo "[latest training step]"
  rg -n "step [0-9]{5}/[0-9]{5}" "$LOG_PATH" | tail -1 || true
  echo

  echo "[latest validation bpb]"
  rg -n "Validation bpb:" "$LOG_PATH" | tail -1 || true
  echo

  echo "[latest CORE metric]"
  rg -n "CORE metric:" "$LOG_PATH" | tail -1 || true
  echo

  echo "[latest ETA line]"
  rg -n "eta:" "$LOG_PATH" | tail -1 || true
  echo

  echo "[wandb run URL]"
  rg -n "wandb:.*(https://wandb.ai|View run at)" "$LOG_PATH" | tail -1 || true
  echo

  echo "[active base processes]"
  ps -eo pid,etimes,%cpu,%mem,cmd | rg "scripts.base_train|scripts.base_eval" | rg -v "rg scripts.base_train|rg scripts.base_eval" || true
  echo

  echo "[GPU snapshot]"
  nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  sleep "$INTERVAL"
done
