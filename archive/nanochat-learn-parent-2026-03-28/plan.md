# Nanochat Next Steps Plan
*Updated: 2026-02-09*

## Current Snapshot
- **Best base model:** `d18` step `500` (val bpb `1.478`)
- **Current SFT model:** `d18/best` step `500` (`run=sft_gsm8k_boost`, val bpb `1.3601`)
  - **GSM8K pass@8** (`-t 0.7 -n 8 -m 1024 -k 50`): **`8/300 (2.67%)`**
  - **GSM8K pass@1** (greedy): `0%`
- **Regression (SFT, -x 200):** `MMLU 23.50%`, `SpellingBee 51.50%`
- **Note:** previous best SFT checkpoint (val bpb `1.2578`) was overwritten on 2026-02-08 due to reusing `--model-tag d18`.

## Primary Bottleneck
Base pretraining is extremely short: `500` steps × `16384` tokens/step ≈ **`8.2M` tokens**. That’s far below what’s needed for robust math reasoning, so SFT mostly teaches format/tool tokens but can’t “create” capability.

## Next Goal: Raise GSM8K pass@8 (no time budget)
1) **Continue base pretraining** (largest leverage) → more reasoning capacity.
2) **Re-run SFT** from the stronger base with GSM8K-heavy mix (keep `--output-tag` unique).
3) **Optionally run GSM8K RL** (`scripts/chat_rl.py`) on top of SFT to directly optimize pass@k.
4) **Re-eval** with the exact same decoding settings for apples-to-apples.

## Goal
Lock in a reproducible, offline-safe evaluation workflow (base vs SFT), then tackle the two biggest usability bottlenecks: **disk bloat** (best checkpoints) and **result logging** (report overwrite).

---

## Phase 1: GSM8K Evaluation (Answer “can it solve at all?”)

Start with a small slice. If pass@8 is still 0 on `-x 100`, it’s very unlikely the full run will show meaningful >0%.

### 1) GSM8K pass@1 (baseline)
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i base -g d18 -s 500 -a GSM8K -x 100 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-base-gsm8k_p1_${TS}.txt

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g d18/best -s 500 -a GSM8K -x 100 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-sft-gsm8k_p1_${TS}.txt
```

### 2) GSM8K pass@8 (diverse decoding + longer generations)
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i base -g d18 -s 500 -a GSM8K -x 100 --device-type cuda \
  -t 0.7 -n 8 -m 1024 |& tee ~/nanochat-learn/notes/eval-base-gsm8k_p8_${TS}.txt

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g d18/best -s 500 -a GSM8K -x 100 --device-type cuda \
  -t 0.7 -n 8 -m 1024 |& tee ~/nanochat-learn/notes/eval-sft-gsm8k_p8_${TS}.txt
```

If pass@8 is non-zero on the slice, rerun without `-x 100` to get stable numbers.

---

## Phase 2: Reproducible Evaluation (Offline + CUDA, with saved logs)

### 0) Fail fast on CUDA
```bash
cd ~/nanochat-learn/nanochat
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

### 1) Smoke tests (quick sanity)
Use offline mode so missing datasets fail immediately, and `tee` so results aren’t lost.
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i base -g d18 -s 500 -a "MMLU|GSM8K|SpellingBee" -x 100 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-base-smoke_${TS}.txt

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g d18/best -s 500 -a "MMLU|GSM8K|SpellingBee" -x 100 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-sft-smoke_${TS}.txt
```

### 2) Stable runs (numbers you can compare)
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i base -g d18 -s 500 -a MMLU --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-base-mmlu_${TS}.txt

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g d18/best -s 500 -a MMLU --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-sft-mmlu_${TS}.txt
```

### 3) Optional: “self-consistency” check for GSM8K (if you want a non-greedy signal)
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g d18/best -s 500 -a GSM8K -n 8 -t 0.7 -m 256 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-sft-gsm8k_sc_${TS}.txt
```

### 4) Record results in `~/nanochat-learn/report.md`
- Update the comparison table (Base vs SFT vs Δ).
- Keep raw logs in `~/nanochat-learn/notes/` (don’t rely on `~/.cache/nanochat/report/*.md` because it overwrites).

---

## Phase 3: Unblock Iteration Speed (Biggest Bottlenecks)

### A) Checkpoint disk bloat (urgent)
Symptoms: `~/.cache/nanochat/chatsft_checkpoints/d18/` is tens of GB due to frequent full-model + optimizer saves.

Actions:
- Keep only:
  - `.../d18/best/model_000500.pt` + `meta_000500.json`
  - `.../d18/model_000500.pt` + `meta_000500.json` (and `optim_000500_rank0.pt` only if you plan to resume)
- Delete intermediate `best/model_*.pt` and old `optim_*.pt` once you’re confident you don’t need them.

Safer cleanup pattern (recommended):
```bash
du -sh ~/.cache/nanochat/chatsft_checkpoints/d18 ~/.cache/nanochat/chatsft_checkpoints/d18/best
TRASH=~/.cache/nanochat/chatsft_checkpoints/d18/_trash_$(date +%F_%H%M)
mkdir -p "$TRASH"
# move old files into $TRASH first, then delete later if everything is good
```

Code improvement (optional next): change “best” saving to a single fixed filename (or top-K retention), and add a flag to skip saving optimizer state.

### A.1) IMPORTANT: avoid mixing runs in `.../d18/best/`
If `~/.cache/nanochat/chatsft_checkpoints/d18/best/` contains an older `model_000500.pt` from a previous run,
`--keep-best-k=1` will keep that file and may delete new best checkpoints saved at steps `<500`.

Fix (safe even while training is running; the script recreates `best/` on next save):
```bash
BASE=~/.cache/nanochat/chatsft_checkpoints/d18
STAMP=$(date +%F_%H%M)
mv "$BASE/best" "$BASE/best_prev_${STAMP}"
mkdir -p "$BASE/best"
```
To load the previous best later: use `-g d18/best_prev_${STAMP}`.

If you also want to preserve the *entire* previous run output (root `model_*.pt`, `optim_*.pt`, etc.), move the whole directory instead:
```bash
BASE=~/.cache/nanochat/chatsft_checkpoints
STAMP=$(date +%F_%H%M)
mv "$BASE/d18" "$BASE/d18_prev_${STAMP}"
mkdir -p "$BASE/d18"
```
Then load the old run with `-g d18_prev_${STAMP}` (or `-g d18_prev_${STAMP}/best`).

Long-term fix (preferred): run new experiments into a fresh output directory via `scripts/chat_sft.py --output-tag ...`.

### C) Code hardening (for future runs)
All changes in `nanochat/scripts/chat_sft.py`.

- Add `--keep-best-k` (default `1`) and delete older `best/model_*.pt` + matching `meta_*.json` after saving a new best.
- Add `--no-save-optimizer` to skip writing `optim_*.pt` in the final checkpoint (can’t resume without it).
- Add `--max-grad-norm` (recommend default `0.0`) and clip on `orig_model.parameters()` after grad accumulation and before `optimizer.step()`.
- Add `--output-tag` to write checkpoints into a fresh directory (prevents overwriting/mixing previous runs).

Important: if `.../best/` already contains old checkpoints from previous runs (e.g. `model_000500.pt`),
`--keep-best-k=1` will keep the highest-step file across *all* runs until the new run reaches a higher step.
To avoid mixing runs, either:
- move the old `.../best/` files to a backup/trash directory before starting, or
- add a future `--output-tag` flag so you can write each SFT run to a fresh directory.

### B) Result logging overwrite
`scripts/chat_eval` writes `~/.cache/nanochat/report/chat-evaluation-*.md` with `w` mode each run.

Actions:
- Always run eval with `|& tee ~/nanochat-learn/notes/...`.
- Optionally copy the report section file to a timestamped name after each run.

---

## Phase 4: Quality Improvements (After Eval + Workflow are solid)

If you want bigger gains than “chat formatting”:
- **Train base longer** (e.g. `d18` to 2k–5k steps), then SFT again; current MMLU is still ~random-baseline.
- For GSM8K: if self-consistency still yields ~0%, the bottleneck is model capacity + pretraining, not SFT tweaks.

If GSM8K pass@8 is still ~0%:
- Consider changing evaluation strictness (e.g. accept `10.0` vs `10`) or sampling/format prompting; current scorer requires an exact `#### <number>` match.
- For training: increase GSM8K mixture weight (e.g. 6×) + add gradient clipping; re-run SFT and re-evaluate GSM8K with the same pass@8 settings.

### Optional retrain (GSM8K boost)

Only do this if Phase 1 shows any GSM8K signal (or you decide it’s worth the experiment anyway).

### Critical knobs to keep the run sane
- Keep `--max-seq-len=2048` if you want apples-to-apples with the current best (`d18/best` is seq_len 2048).
- Always set `--eval-tokens=524288` (default is ~10M tokens per eval and is *very* slow).

### Retrain command (baseline-preserving)
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)
OUT=d18_sft_gsm8kboost_${TS}

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 .venv/bin/python -u -m scripts.chat_sft \
  --model-tag=d18 --model-step=500 \
  --output-tag=$OUT \
  --adamw-only --weight-decay=0.1 \
  --total-batch-size=8192 --device-batch-size=4 \
  --max-seq-len=2048 --num-iterations=500 \
  --eval-every=50 --eval-tokens=524288 \
  --max-grad-norm=1.0 \
  --keep-best-k=1 --no-save-optimizer \
  --run=sft_gsm8k_boost_${TS} \
  |& tee ~/nanochat-learn/notes/sft-gsm8k-boost_${TS}.txt
```

### Post-run eval (must-do)
Run the exact same GSM8K eval settings as Phase 1 so the comparison is valid.
```bash
cd ~/nanochat-learn/nanochat
OUT=d18_sft_gsm8kboost_<TRAIN_TS>   # set this to the OUT used for training above
BEST=$OUT/best
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g $BEST -a GSM8K -x 100 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-sftNEW-gsm8k_p1_${TS}.txt

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g $BEST -a GSM8K -x 100 --device-type cuda \
  -t 0.7 -n 8 -m 1024 |& tee ~/nanochat-learn/notes/eval-sftNEW-gsm8k_p8_${TS}.txt
```

Also sanity-check for regressions:
```bash
cd ~/nanochat-learn/nanochat
OUT=d18_sft_gsm8kboost_<TRAIN_TS>   # set this to the OUT used for training above
BEST=$OUT/best
TS=$(date +%F_%H%M)

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m scripts.chat_eval -i sft -g $BEST -a "MMLU|SpellingBee" -x 200 --device-type cuda \
  |& tee ~/nanochat-learn/notes/eval-sftNEW-sanity_${TS}.txt
```

### Decision gate (what to do next)
- If GSM8K `pass@8` on the slice is still ≤1–2%: stop iterating on SFT mixtures; prioritize **longer base pretraining**.
- If GSM8K `pass@8` gets to ~5%+ on the slice: consider trying `scripts/chat_rl.py` (RL on GSM8K) to directly optimize the `####` reward.

### Base pretraining path (likely needed)
The current base model is extremely undertrained in tokens; GSM8K improvements will mostly come from more base training.
Example: resume `d18` from step 500 → 5000 (saves only at end to avoid disk bloat):
```bash
cd ~/nanochat-learn/nanochat
TS=$(date +%F_%H%M)

TORCH_COMPILE_DISABLE=1 PYTHONUNBUFFERED=1 \
.venv/bin/python -u -m scripts.base_train \
  --depth=18 --aspect-ratio=64 --head-dim=64 --window-pattern=L \
  --max-seq-len=512 --device-batch-size=4 --total-batch-size=16384 \
  --resume-from-step=500 --num-iterations=5000 \
  --eval-every=250 --eval-tokens=524288 --save-every=-1 \
  |& tee ~/nanochat-learn/notes/base-d18_500to5000_${TS}.txt
```
