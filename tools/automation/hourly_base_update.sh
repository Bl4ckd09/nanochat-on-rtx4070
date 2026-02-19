#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <base_train_log_path> [output_log_path] [interval_seconds]"
  exit 1
fi

BASE_LOG="$1"
OUT_LOG="${2:-$HOME/nanochat-learn/notes/base-hourly-updates.log}"
INTERVAL="${3:-3600}"

if [ ! -f "$BASE_LOG" ]; then
  echo "Base log not found: $BASE_LOG"
  exit 1
fi

mkdir -p "$(dirname "$OUT_LOG")"

echo "==== Hourly monitor started $(date '+%F %T') ====" >> "$OUT_LOG"
echo "base_log=$BASE_LOG interval=${INTERVAL}s" >> "$OUT_LOG"

while true; do
  ts="$(date '+%F %T')"

  if pgrep -f "scripts.base_train" >/dev/null 2>&1; then
    step_line="$(rg -n "step [0-9]+/[0-9]+" "$BASE_LOG" | tail -1 | sed 's/^[0-9]*://' || true)"
    val_line="$(rg -n "Validation bpb:" "$BASE_LOG" | tail -1 | sed 's/^[0-9]*://' || true)"
    core_line="$(rg -n "CORE metric:" "$BASE_LOG" | tail -1 | sed 's/^[0-9]*://' || true)"
    eta_line="$(rg -n "eta:" "$BASE_LOG" | tail -1 | sed 's/^[0-9]*://' || true)"
    gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"

    {
      echo "[$ts]"
      echo "status: base_train running"
      echo "step: ${step_line:-not yet}"
      echo "val:  ${val_line:-not yet}"
      echo "core: ${core_line:-not yet}"
      echo "eta:  ${eta_line:-not yet}"
      echo "gpu:  ${gpu_line:-n/a}"
      echo
    } >> "$OUT_LOG"
  else
    echo "[$ts] status: base_train not running" >> "$OUT_LOG"
    echo >> "$OUT_LOG"
  fi

  sleep "$INTERVAL"
done
