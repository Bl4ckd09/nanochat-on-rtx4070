# Nanochat Training Report

*Auto-updated by Claude Code hooks*

## Environment
- GPU: RTX 4070 12GB
- PyTorch: 2.9.1+cu128
- Package Manager: uv

---

## Latest Status Snapshot (2026-02-19)

### Base model (clean-safe max-scale)
| Metric | Value |
|--------|-------|
| Model tag | `d18_clean200k_safe_2026-02-13_1226` |
| Step | `200000` |
| train bpb | `0.847593` |
| val bpb | `0.903612` |
| CORE metric | `0.1255` |
| CORE overflow policy | `skip` with `--core-max-seq-len 5120` (policy lock file present) |
| CORE results file | `~/.cache/nanochat/base_eval/base_model_200000.csv` |

Logs:
- `~/nanochat-learn/notes/d18_clean200k_safe_2026-02-13_1226_s200000_base_eval_2026-02-18_1701.log` (original run overflowed on long prompt)
- `~/nanochat-learn/notes/d18_clean200k_safe_2026-02-13_1226_s200000_base_eval_skip5120_2026-02-18_1701.log` (completed)
- `~/nanochat-learn/notes/base_core_overflow_policy_fixed.lock` (comparability guard)

### Chat champion selected by objective
| Metric | Value |
|--------|-------|
| Selection mode | `OBJECTIVE=mmlu` (`NO_TEST_PEEK=1`) |
| Champion | `d18_clean200k_lora_prod_2026-02-17_1209_s1024/best` |
| Step | `1000` |
| Selection record | `~/nanochat-learn/notes/selected_champion_2026-02-18_1741.txt` |
| Pipeline log | `~/nanochat-learn/notes/post_base_next_steps_2026-02-18_1741.log` |

### 1k confirmation (champion)
| Benchmark | Result |
|-----------|--------|
| GSM8K pass@8 | `8/1000 = 0.80%` (Wilson 95%: `[0.41%, 1.57%]`) |
| MMLU | `262/1000 = 26.20%` (Wilson 95%: `[23.57%, 29.01%]`) |
| SpellingBee | `9/256 = 3.52%` (Wilson 95%: `[1.86%, 6.55%]`) |

Source summary:
- `~/nanochat-learn/notes/d18_clean200k_lora_prod_2026-02-17_1209_s1024_best_s1000_confirm_summary_1000_2026-02-19_0258.txt`

### Baseline comparison (d18_20k/best, step 500)
| Benchmark | Result |
|-----------|--------|
| GSM8K pass@8 | `14/1000 = 1.40%` |
| MMLU | `234/1000 = 23.40%` |
| SpellingBee | `71/256 = 27.73%` |

Source summary:
- `~/nanochat-learn/notes/d18_20k_best_s500_confirm_summary_1000_2026-02-17_1847.txt`

### Additional final evals on champion
| Benchmark | Result |
|-----------|--------|
| ARC-Easy | `23.40%` |
| ARC-Challenge | `25.80%` |
| HumanEval (pass@10) | `0/164 = 0.00%` |

Logs:
- `~/nanochat-learn/notes/d18_clean200k_lora_prod_2026-02-17_1209_s1024_best_s1000_final_arc1k_2026-02-19_0636.log`
- `~/nanochat-learn/notes/d18_clean200k_lora_prod_2026-02-17_1209_s1024_best_s1000_final_humaneval_pass10_2026-02-19_0636.log`

---

## Training Runs

### Run 1: depth=6 (2026-02-04)
| Metric | Value |
|--------|-------|
| Model | depth=6, 73.5M params |
| Steps | 500 |
| Time | 0.86 min |
| Loss | 5.6 → 5.2 |
| Val bpb | 1.56 |
| Peak VRAM | 1.1 GB |
| Throughput | ~160k tok/sec |
| Checkpoint | `~/.cache/nanochat/base_checkpoints/d6/` |

### Run 2: depth=12 (2026-02-04)
| Metric | Value |
|--------|-------|
| Model | depth=12, ~280M params |
| Steps | 500 |
| Time | 3.36 min |
| Loss | 6.9 → 5.0 |
| Val bpb | 1.50 |
| Peak VRAM | 3.9 GB |
| Throughput | ~40k tok/sec |
| Checkpoint | `~/.cache/nanochat/base_checkpoints/d12/` |

### Run 3: depth=16 (2026-02-04)
| Metric | Value |
|--------|-------|
| Model | depth=16, 537M params |
| Steps | 500 |
| Time | 6.83 min |
| Loss | 10.4 → 5.0 |
| Val bpb | **1.493** |
| Peak VRAM | 7.1 GB |
| Throughput | ~19k tok/sec |
| Checkpoint | `~/.cache/nanochat/base_checkpoints/d16/` |

### Run 4: depth=18 (2026-02-04)
| Metric | Value |
|--------|-------|
| Model | depth=18, ~710M params |
| Steps | 500 |
| Time | 9.80 min |
| Loss | 10.4 → 4.9 |
| Val bpb | **1.478** |
| Peak VRAM | 9.3 GB |
| Throughput | ~13k tok/sec |
| Checkpoint | `~/.cache/nanochat/base_checkpoints/d18/` |

### Run 5: depth=19 (2026-02-04) - CANCELLED
| Metric | Value |
|--------|-------|
| Model | depth=19, ~800M params |
| Status | Cancelled - VRAM maxed out |
| VRAM | ~12 GB (3.3k tok/sec, ~38 min ETA) |
| Notes | Even seq_len=256 didn't help enough |

### Run 6: depth=20 aspect=64 (2026-02-04) - CANCELLED
| Metric | Value |
|--------|-------|
| Model | depth=20, 897M params |
| Status | Cancelled - too slow |
| VRAM (batch=4) | ~12 GB (maxed out, 612 tok/sec) |
| VRAM (batch=2) | ~9 GB (~1400 tok/sec, ~90 min ETA) |
| Notes | Impractical on RTX 4070 with default aspect ratio |

### Run 7: depth=20 aspect=48 (2026-02-04)
| Metric | Value |
|--------|-------|
| Model | depth=20, 599M params (narrower) |
| Steps | 500 |
| Time | 7.62 min |
| Val bpb | 1.487 |
| Peak VRAM | 7.9 GB |
| Throughput | ~17.7k tok/sec |
| Checkpoint | `~/.cache/nanochat/base_checkpoints/d20/` |
| Notes | Reduced aspect_ratio trades width for depth |

---


### Base: depth=18 aspect=64 (2026-02-09 12:56)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=18, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=18 aspect=64 (2026-02-09 13:33)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=18, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=18 aspect=64 (2026-02-09 13:36)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=18, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=18 aspect=64 (2026-02-09 13:37)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=18, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=18 aspect=64 (2026-02-10 21:26)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=18, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=20 aspect=64 (2026-02-19 19:38)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=20, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=24 aspect=24 (2026-02-19 19:44)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=24, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=24 aspect=24 (2026-02-19 19:51)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=24, 910,691,760 params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### Base: depth=24 aspect=48 (2026-02-19 20:41)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=24, 910,691,760 params |
| Val bpb | N/A |
| Peak VRAM | 11492.43MiB |
| Time | 0.00m |


### Base: depth=24 aspect=48 (2026-02-19 20:45)
| Metric | Value |
|--------|-------|
| Status | ⚠️ Cancelled/Incomplete |
| Model | depth=24, ? params |
| Val bpb | N/A |
| Peak VRAM | 14407.62MiB |
| Time | N/A |

## SFT Experiments

### ✅ SFT SUCCESS: Full AdamW Training (2026-02-06) ⭐ BEST OBSERVED (checkpoint overwritten)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Method | Full fine-tuning with AdamW-only |
| Config | `--adamw-only --weight-decay=0.1 --total-batch-size=8192 --device-batch-size=4` |
| Steps | 500 |
| Training time | 212 min (~3.5 hours) |
| Result | ✅ **BEST RESULT - 22.9% improvement** |
| Val bpb | 1.63 → **1.26** |
| Peak VRAM | ~12 GB |
| Checkpoint | Overwritten in-place on 2026-02-08 by `sft_gsm8k_boost` (weights not preserved) |

**Val_bpb trajectory:**
```
Step 0:   1.6344 (pretrained baseline)
Step 50:  2.1239 (initial catastrophic forgetting)
Step 100: 1.9548 (recovering)
Step 150: 1.7550
Step 200: 1.6094 ← beats baseline!
Step 250: 1.5275
Step 300: 1.4940
Step 350: 1.4422
Step 400: 1.4000
Step 450: 1.3213
Step 500: 1.2578 ← final best
```

**Evaluation results:**
| Benchmark | Base Model | SFT Model | Change |
|-----------|------------|-----------|--------|
| MMLU | 23.07% | 24.65% | +1.58 pts |
| GSM8K | 0% | 0% | Same (model too small for math) |
| SpellingBee | 0% | **39%** | +39 pts (learned chat format) |

**Key learnings:**
- Small batch size (8192 tokens) prevents catastrophic forgetting
- AdamW-only mode (no Muon) works well for SFT
- Initial val_bpb spike is normal - model recovers and improves
- `TORCH_COMPILE_DISABLE=1` needed due to GPU memory constraints (slower but works)

**Checkpoint note:** this run originally wrote `d18/best/model_000500.pt` with val_bpb `1.2578`, but it was overwritten on
2026-02-08 because both runs used `--model-tag=d18` with `--keep-best-k=1`.

---

### ✅ SFT SUCCESS: GSM8K Boost Retrain (2026-02-08) — current `d18/best`
| Metric | Value |
|--------|-------|
| Run name | `sft_gsm8k_boost` |
| Base model | d18 step 500 |
| Method | Full fine-tuning with AdamW-only |
| Config | `--adamw-only --weight-decay=0.1 --total-batch-size=8192 --device-batch-size=4 --max-seq-len=2048 --num-iterations=500 --eval-every=50 --eval-tokens=524288 --max-grad-norm=1.0 --keep-best-k=1 --no-save-optimizer` |
| Data | GSM8K oversampling 6× (48K rows, ~5.3% of train mixture) |
| Steps | 500 |
| Training time | 236 min (~4 hours) |
| Val bpb | 1.6344 → **1.3601** |
| Checkpoint | `~/.cache/nanochat/chatsft_checkpoints/d18/best/model_000500.pt` |

**Val_bpb trajectory (from `wandb/.../output.log`):**
```
Step 0:   1.6344
Step 50:  2.1885
Step 100: 2.0186
Step 150: 1.9323
Step 200: 1.8491
Step 250: 1.7741
Step 300: 1.6896
Step 350: 1.6015
Step 400: 1.5240
Step 450: 1.4232
Step 500: 1.3601
```

**Verified features:**
- `--keep-best-k=1` rotated old best checkpoints (only step 500 remains in `best/`)
- `--no-save-optimizer` skipped saving a new optimizer checkpoint (note: an older `optim_000500_rank0.pt` from 2026-02-06 may still exist)
- `--max-grad-norm=1.0` ran without errors (clipping is silent unless you add logging)

**Important:** val_dataset did not change between runs; `1.3601` is a regression vs the 2026-02-06 run's `1.2578` on the same validation mix.

---

### GSM8K Phase 1 Evaluation (2026-02-08, pre-boost checkpoint)
These results were collected before the GSM8K-boost retrain overwrote `d18/best`.

- pass@1 (greedy, -x 100): base `0%`, SFT `0%`
- pass@8 (t=0.7, n=8, m=1024, -x 100): base `0%`, SFT `1%` (1/100)
- Diagnosis (spot-check): model rarely emits `#### <number>` and tends to ramble; capacity/pretraining bottleneck dominates.

---

### GSM8K Phase 2 Re-Evaluation (2026-02-09, post-boost checkpoint `sft_gsm8k_boost`)

**Checkpoint:** `d18/best/model_000500.pt`
- `val_bpb=1.3601`, `sequence_len=2048`, `run=sft_gsm8k_boost`
- `adamw_only=true`, `weight_decay=0.1`, `max_grad_norm=1.0`
- Git: `0ee59449afeeef1cc8dd7242672b456c81a1b220`

**GSM8K results (side-by-side with Phase 1):**

| Eval | Decoding | Pre-boost (Phase 1) | Post-boost (Phase 2) | Change |
|------|----------|---------------------|----------------------|--------|
| Base pass@1 | greedy, -x 100 | 0/100 (0%) | 0/100 (0%) | Same |
| Base pass@8 | t=0.7, n=8, m=1024, k=50, -x 100 | 0/100 (0%) | 0/100 (0%) | Same |
| SFT pass@1 | greedy, -x 100 | 0/100 (0%) | 0/100 (0%) | Same |
| SFT pass@8 | t=0.7, n=8, m=1024, k=50, -x 100 | 1/100 (1%) | **3/100 (3%)** | +2 pts |
| SFT pass@8 | t=0.7, n=8, m=1024, k=50, -x 300 | — | **8/300 (2.67%)** | Confirmed |

**Regression check (SFT d18/best, -x 200, -b 32):**

| Benchmark | Pre-boost | Post-boost | Change |
|-----------|-----------|------------|--------|
| MMLU | 24.65% | 23.50% | -1.15 pts (within noise) |
| SpellingBee | 39.00% | **51.50%** | **+12.5 pts** |

**Spot-check qualitative analysis (5 problems, seeds 42 & 123):**

Pre-boost: model never emitted `#### <number>` or `<|python_start|>` tool calls. Output was incoherent rambling.

Post-boost changes observed:
- Model now emits `#### <number>` in some completions (3/10 samples)
- Model now uses `<|python_start|>` tool calls in some completions (2/10 samples with seed=123)
- Math reasoning is still wrong (incorrect answers even when format is correct)
- Incoherent rambling still dominates most completions

**Decision (Step 6 of plan):**
GSM8K pass@8 improved from 1% → 2.67% (confirmed on 300 problems). The oversampling helped with format acquisition but the model still cannot reason mathematically. The bottleneck remains base model capacity. SpellingBee's unexpected jump (+12.5 pts) suggests the structured output training generalized.

**Next steps:**
- Resume base pretraining (d18 step 500 → 2k-5k steps) to build more capacity
- Then re-SFT with `--output-tag` to avoid overwriting `d18/best`
- Keep this checkpoint as the current best SFT model

---

### ✅ SFT SUCCESS: LoRA Conservative (2026-02-05)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Method | LoRA (Low-Rank Adaptation) |
| Config | rank=4, alpha=8, lr=1e-5, batch=8192, steps=500 |
| Result | ✅ **SUCCESS - No catastrophic forgetting** |
| Val bpb | 1.87 → **1.9059** (+2% only) |
| Min val_bpb | **1.8711** (better than baseline!) |
| Trainable params | 1,078,272 / 702,974,052 (0.15%) |
| Training time | 2.87 min |
| Checkpoint | `~/.cache/nanochat/chatsft_checkpoints/d18/` |

**Key changes:**
- Implemented LoRA in `nanochat/lora.py` - freezes all base weights, adds trainable low-rank matrices to attention layers
- Conservative hyperparameters prevent mode collapse (aggressive lr=1e-4, rank=16 caused BOS-only output)
- LoRA weights merged into base model at checkpoint save for inference compatibility

---

### ❌ Failed Attempts (for reference)

### SFT Attempt 1: Large batch size (2026-02-04 10:30)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Batch size | 524,288 (default) |
| Result | Diverged after 2 steps |
| Val bpb | 1.87 → 4.70 (worse) |

### SFT Attempt 2: Smaller batch, full epoch (2026-02-04 11:00)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Batch size | 8,192 |
| Iterations | -1 (full epoch) |
| Result | Severe overfitting |
| Val bpb | 1.87 → 4.6 → 5.5 → 6.0 → 6.7 (kept getting worse) |
| Train loss | 0.00000x (memorized) |

### SFT Attempt 3: Default LR (2026-02-04 12:00)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Config | matrix_lr=0.02, batch=8192, steps=200 |
| Result | ❌ Severe degradation |
| Val bpb | 1.87 → 6.24 |
| Chat output | Gibberish |

### SFT Attempt 4: Lower LR (2026-02-04 13:00)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Config | matrix_lr=0.01, batch=8192, steps=200 |
| Result | ❌ Still degraded |
| Val bpb | 1.87 → 4.82 |
| Chat output | Gibberish |

### SFT Attempt 5: Very Low LR (2026-02-04 14:30)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Config | matrix_lr=0.001, embed_lr=0.01, steps=100 |
| Result | ⚠️ Degraded but slower |
| Val bpb | 1.87 → 2.42 |
| Chat output | Incoherent |

### SFT Attempt 6: Very Low LR extended (2026-02-04 15:30)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Config | matrix_lr=0.001, embed_lr=0.01, steps=500 |
| Result | ❌ Degraded with more steps |
| Val bpb | 1.87 → 3.26 |
| Chat output | Empty/BOS only |

### SFT Attempt 7: Weight Decay (2026-02-04 17:00) ⭐ BEST
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Config | matrix_lr=0.001, weight_decay=0.1, steps=100 |
| Result | ⚠️ Best result (still degraded) |
| Val bpb | 1.87 → **2.33** (+25% worse than baseline) |
| Chat output | Incoherent |

### SFT Attempt 8: Weight Decay extended (2026-02-04 18:00)
| Metric | Value |
|--------|-------|
| Base model | d18 step 500 |
| Config | matrix_lr=0.001, weight_decay=0.1, steps=400+ |
| Result | ❌ Degraded with more steps |
| Val bpb | 1.87 → 3.04 |

---


### SFT: depth=? aspect=64 (2026-02-10 11:56)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=?, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### SFT: depth=? aspect=64 (2026-02-11 14:36)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=?, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### SFT: depth=? aspect=64 (2026-02-11 14:43)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=?, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### SFT: depth=? aspect=64 (2026-02-11 14:45)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=?, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### SFT: depth=? aspect=64 (2026-02-11 14:55)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=?, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### SFT: depth=? aspect=64 (2026-02-11 15:07)
| Metric | Value |
|--------|-------|
| Status | ✅ Success |
| Model | depth=?, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |


### SFT: depth=18 aspect=48 (2026-02-19 19:22)
| Metric | Value |
|--------|-------|
| Status | ❌ OOM |
| Model | depth=18, ? params |
| Val bpb | N/A |
| Peak VRAM | N/A |
| Time | N/A |

## SFT Analysis (2026-02-05)

**Root Cause: Muon optimizer is too aggressive for fine-tuning**

| Factor | Impact |
|--------|--------|
| Muon optimizer | Designed for pretraining from scratch, not fine-tuning |
| No layer freezing | All layers updated equally, destroying pretrained knowledge |
| No LR warmup | Sudden large updates destabilize weights |
| Dataset mismatch | SFT data distribution differs from pretraining data |

**Solutions Comparison:**

| Approach | val_bpb | Result |
|----------|---------|--------|
| Baseline (pretrained) | 1.63 | - |
| Muon + lower LR | 2.33 | ❌ +43% worse |
| AdamW-only (large batch) | 2.82 | ❌ +73% worse |
| Layer freezing | 6.52 | ❌ +300% worse |
| LoRA (aggressive) | 2.47 | ⚠️ Mode collapse |
| LoRA (conservative) | 1.91 | ✅ +17% worse but stable |
| **AdamW + small batch** | **1.26** | ✅ **-23% better!** ⭐ |

**Key learnings:**
1. **Small batch size (8192 tokens)** is the key to successful full fine-tuning
2. **AdamW-only** works better than LoRA when batch size is small enough
3. Initial catastrophic forgetting is temporary - model recovers after ~100-200 steps
4. LoRA is still useful for fast iteration (3 min vs 212 min)

---


### Chat Eval: source=base, model=d18, tasks=MMLU (2026-02-06 19:14)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| Step | 500 |
| MMLU | 23.07% |


### Chat Eval: source=sft, model=d18, tasks=MMLU, max=100 (2026-02-08 15:17)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| MMLU | 20.00% |


### Chat Eval: source=sft, model=d18, tasks=MMLU (2026-02-08 15:19)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| MMLU | 24.65% |


### Chat Eval: source=base, model=d18, step=500, tasks=GSM8K, max=100 (2026-02-08 16:15)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| GSM8K | 0.00% |


### Chat Eval: source=base, model=d18, step=500, tasks=GSM8K, max=100 (2026-02-09 10:15)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| GSM8K | 0.00% |


### Chat Eval: source=sft, model=d18/best, step=500, tasks=GSM8K, max=100 (2026-02-09 10:41)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| GSM8K | 0.00% |


### Chat Eval: source=sft, model=d18/best, step=500, tasks=GSM8K, max=100 (2026-02-09 10:50)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| GSM8K | 3.00% |


### Chat Eval: source=sft, model=d18/best, step=500, tasks="MMLU|SpellingBee", max=200 (2026-02-09 11:26)
| Metric | Value |
|--------|-------|
| Status | ✅ Complete |
| MMLU | 23.50% |
| SpellingBee | 51.50% |

## Best Results
- **Best Val bpb (pretrain)**: 1.478 (depth=18, aspect=64)
- **Best Val bpb (SFT, on disk)**: **1.3601** (`sft_gsm8k_boost`, 500 steps; `d18/best`)
- **Best Val bpb (SFT, historical)**: **1.2578** (2026-02-06 AdamW-only run; checkpoint overwritten on 2026-02-08)
- **Best SFT SpellingBee**: **51.50%** (d18/best step 500, 2026-02-09 11:26; max=200)
- **Best GSM8K pass@8**: 2.67% on 300 problems (`sft_gsm8k_boost`, t=0.7, n=8, m=1024, k=50)
- **Caution:** reuse of `--model-tag=d18` can overwrite `d18/best` when `--keep-best-k=1`; use a unique output dir per run (e.g. `--output-tag`).
- **Fastest**: depth=6 (0.86 min)
- **Best quality/speed**: depth=18 recommended (9.3 GB VRAM)
- **Deepest working**: depth=20 with aspect=48 (1.487 bpb, 7.9 GB)

---

## VRAM Findings

| Depth | Aspect | Params | VRAM | Status |
|-------|--------|--------|------|--------|
| 6 | 64 | 73.5M | 1.1 GB | ✅ |
| 12 | 64 | 280M | 3.9 GB | ✅ |
| 16 | 64 | 537M | 7.1 GB | ✅ |
| 18 | 64 | 710M | 9.3 GB | ✅ best |
| 19 | 64 | 800M | 12 GB | ❌ too slow |
| 20 | 64 | 897M | 12 GB | ❌ too slow |
| 20 | 48 | 599M | 7.9 GB | ✅ workaround |

**Key insight:** depth=18 is the hard VRAM limit. Reducing aspect_ratio allows deeper models but trades width for depth.

---

## Notes
- depth=18 is the sweet spot for RTX 4070 12GB (best bpb, 9.3 GB VRAM)
- depth=16 is good alternative if faster training needed (7.1 GB, 1.493 bpb)
- depth≥19 with aspect=64 maxes out VRAM and runs 10-15x slower
- Longer training (2000+ steps) would improve base model results

### SFT Best Practices
- **Best method**: Full AdamW training with `--adamw-only --weight-decay=0.1 --total-batch-size=8192`
- **Alternative**: LoRA with `--lora --lora-rank=4 --lora-alpha=8 --lora-lr=1e-5` (faster, less improvement)
- **Avoid Muon** for fine-tuning - its orthogonalization destroys pretrained structure
- **Small batch size** (8192 tokens) prevents catastrophic forgetting
- **Expect initial spike** in val_bpb - model will recover after ~100-200 steps
- Use `TORCH_COMPILE_DISABLE=1` if GPU memory is tight (slower but works)
