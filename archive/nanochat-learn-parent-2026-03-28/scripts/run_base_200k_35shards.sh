#!/usr/bin/env bash
set -euo pipefail

cd /home/sun0115/nanochat
source .venv/bin/activate

LOG=/home/sun0115/nanochat-learn/notes/base-d18_200k_35shards_live.txt
echo "Starting base_train at $(date '+%F %T')" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
python -u -m scripts.base_train \
  --depth 18 --aspect-ratio 64 --head-dim 64 --window-pattern L \
  --max-seq-len 512 --device-batch-size 4 --total-batch-size 16384 \
  --device-type cuda \
  --resume-from-step 20000 --num-iterations 200000 \
  --warmup-ratio 0.5 --warmdown-ratio 0.5 --final-lr-frac 0.0 \
  --eval-every 2000 --eval-tokens 524288 \
  --core-metric-every 20000 --core-metric-max-per-task 100 \
  --sample-every -1 --save-every 20000 \
  --run base_d18_200k_35shards \
  | tee -a "$LOG"
