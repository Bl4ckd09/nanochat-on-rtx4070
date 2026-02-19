#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?usage: run_core_chain.sh <model_tag> [step]}"
STEP="${2:-200000}"

cd ~/nanochat-learn/nanochat
source .venv/bin/activate

TS=$(date +%F_%H%M)
QLOG=~/nanochat-learn/notes/${TAG}_core_quick_${TS}.log
FLOG=~/nanochat-learn/notes/${TAG}_core_full_${TS}.log

printf '[start] quick core -> %s\n' "$QLOG"
PYTHONUNBUFFERED=1 python -u -m scripts.base_eval \
  --model-tag "$TAG" --step "$STEP" \
  --eval core --max-per-task 100 --device-type cuda \
  |& tee "$QLOG"

printf '[start] full core+bpb -> %s\n' "$FLOG"
PYTHONUNBUFFERED=1 python -u -m scripts.base_eval \
  --model-tag "$TAG" --step "$STEP" \
  --eval core,bpb --device-type cuda \
  |& tee "$FLOG"

printf '[done] core eval chain complete\n'
