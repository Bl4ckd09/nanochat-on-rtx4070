# Local Experiment Report (Bl4ckd09)

This file summarizes local experiments run on top of `karpathy/nanochat` in this fork (`Bl4ckd09/nanochat-experiments`).

## 1) Lineage

- Upstream fork point: `542beb0c8c175af2d52ec7065345dcd8f0162368`
- Current fork head: `9f761dc4b05829c0ff6c0c85cd07a43357abb2a3`
- Local commits on top of upstream:
  - `80ba265` Add LoRA fine-tuning support
  - `0ee5944` Improve SFT quality/checkpoint management
  - `93d7a0f` Update local workflow/docs
  - `9f761dc` Add local run marker file

## 2) Local code deltas vs upstream

| File(s) | Local change |
|---|---|
| `nanochat/lora.py` | New LoRA implementation (`LoRALinear`, apply/merge/load helpers) |
| `scripts/chat_sft.py` | LoRA mode, AdamW-only mode, warmup/warmdown scheduler, gradient clipping/checkpointing, safer checkpoint handling (`output-tag`, best-k rotation) |
| `nanochat/gpt.py` | Chunked cross-entropy for lower VRAM, optional checkpointed blocks, AdamW-only path for matrix params |
| `scripts/base_eval.py`, `nanochat/core_eval.py` | CORE overflow handling (`error`/`truncate`/`skip`) + explicit skipped/evaluated counts |
| `nanochat/engine.py` | Calculator-expression normalization for noisy model tool calls |

## 3) Run summary (recent)

### 3.1 Base train (clean-safe max-scale)

- Model tag: `d18_clean200k_safe_2026-02-13_1226`
- Step: `200000`
- From training log:
  - `Validation bpb = 0.902180`
  - Peak memory: `9515.62 MiB`
  - Minimum validation bpb: `0.902180`

Source: `~/nanochat-learn/notes/base_d18_clean200k_safe_2026-02-13_1226.log`

### 3.2 Base eval (CORE + BPB)

Initial full eval hit long-prompt overflow, then rerun with fixed policy:
- `--core-overflow-policy skip --core-max-seq-len 5120`

Final metrics:
- train bpb: `0.847593`
- val bpb: `0.903612`
- CORE metric: `0.1255`

Source: `~/nanochat-learn/notes/d18_clean200k_safe_2026-02-13_1226_s200000_base_eval_skip5120_2026-02-18_1701.log`

### 3.3 SFT champion run (LoRA)

- Checkpoint: `d18_clean200k_lora_prod_2026-02-17_1209_s1024/best` step `1000`
- End-of-run validation bpb: `0.6423`
- Peak memory usage: `6401.51 MiB`

Source: `~/nanochat-learn/notes/sft_d18_clean200k_lora_prod_2026-02-17_1209_s1024.log`

W&B run: https://wandb.ai/sunshines-gmail-com/nanochat-sft/runs/sm8rtit5

## 4) Evaluation comparison (champion vs baseline)

Champion:
- `d18_clean200k_lora_prod_2026-02-17_1209_s1024/best`, step `1000`

Baseline:
- `d18_20k/best`, step `500`

### 4.1 1k confirmation pack

| Metric | Champion | Baseline | Delta (Champion - Baseline) |
|---|---:|---:|---:|
| GSM8K pass@8 | 0.80% (8/1000) | 1.40% (14/1000) | -0.60 pp |
| MMLU | 26.20% (262/1000) | 23.40% (234/1000) | +2.80 pp |
| SpellingBee | 3.52% (9/256) | 27.73% (71/256) | -24.21 pp |

Sources:
- Champion: `~/nanochat-learn/notes/d18_clean200k_lora_prod_2026-02-17_1209_s1024_best_s1000_confirm_summary_1000_2026-02-19_0258.txt`
- Baseline: `~/nanochat-learn/notes/d18_20k_best_s500_confirm_summary_1000_2026-02-17_1847.txt`

### 4.2 Additional final evals on champion

| Metric | Result |
|---|---:|
| ARC-Easy | 23.40% |
| ARC-Challenge | 25.80% |
| HumanEval (pass@10) | 0.00% (0/164) |

Sources:
- `~/nanochat-learn/notes/d18_clean200k_lora_prod_2026-02-17_1209_s1024_best_s1000_final_arc1k_2026-02-19_0636.log`
- `~/nanochat-learn/notes/d18_clean200k_lora_prod_2026-02-17_1209_s1024_best_s1000_final_humaneval_pass10_2026-02-19_0636.log`

## 5) W&B references

- Project: https://wandb.ai/sunshines-gmail-com/nanochat-sft
- Champion LoRA run: https://wandb.ai/sunshines-gmail-com/nanochat-sft/runs/sm8rtit5
- Safe SFT run (earlier): https://wandb.ai/sunshines-gmail-com/nanochat-sft/runs/nt9p2txo
- GSM8K-boost run (earlier): https://wandb.ai/sunshines-gmail-com/nanochat-sft/runs/map1htc1

Recommended run charts:
- `val/bpb`
- `train/loss`
- `train/lrm`
- `train/tok_per_sec`
- `train/dt`

## 6) Current interpretation

- Improvement observed: MMLU (+2.8 pp) over baseline.
- Regression observed: GSM8K pass@8 and SpellingBee under baseline.
- Base CORE remains below GPT-2 target (`0.256525`) with current setup.
