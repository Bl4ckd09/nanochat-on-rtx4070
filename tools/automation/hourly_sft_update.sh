#!/usr/bin/env bash
set -euo pipefail

LOG_GLOB="${1:-$HOME/nanochat-learn/notes/sft_d18_clean200k_sftmax_*_chain.log}"
OUT_LOG="${2:-$HOME/nanochat-learn/notes/sft-hourly-updates.log}"
INTERVAL="${3:-3600}"

mkdir -p "$(dirname "$OUT_LOG")"

echo "==== Hourly SFT monitor started $(date '+%F %T') ====" >> "$OUT_LOG"
echo "log_glob=$LOG_GLOB interval=${INTERVAL}s" >> "$OUT_LOG"

while true; do
  ts="$(date '+%F %T')"
  latest_log="$(ls -1t $LOG_GLOB 2>/dev/null | head -1 || true)"

  if [ -z "${latest_log}" ] || [ ! -f "${latest_log}" ]; then
    {
      echo "[$ts]"
      echo "status: no matching SFT log yet"
      echo
    } >> "$OUT_LOG"
    sleep "$INTERVAL"
    continue
  fi

  step_line="$(rg -n '^step [0-9]+/[0-9]+' "$latest_log" | tail -1 | sed 's/^[0-9]*://' || true)"
  val_line="$(rg -n 'Validation bpb:' "$latest_log" | tail -1 | sed 's/^[0-9]*://' || true)"
  best_line="$(rg -n 'Saved best|best checkpoint|Rotated old' "$latest_log" | tail -1 | sed 's/^[0-9]*://' || true)"
  warn_line="$(rg -n 'OOM|out of memory|retrying with --max-seq-len 1536|error' "$latest_log" | tail -1 | sed 's/^[0-9]*://' || true)"
  gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"

  if pgrep -f 'scripts.chat_sft' >/dev/null 2>&1; then
    status="running"
  else
    status="not running"
  fi

  {
    echo "[$ts]"
    echo "status: ${status}"
    echo "log:    ${latest_log}"
    echo "step:   ${step_line:-not yet}"
    echo "val:    ${val_line:-not yet}"
    echo "best:   ${best_line:-not yet}"
    echo "warn:   ${warn_line:-none}"
    echo "gpu:    ${gpu_line:-n/a}"
    echo
  } >> "$OUT_LOG"

  sleep "$INTERVAL"
done
