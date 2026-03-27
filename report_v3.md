# Report v3

Updated base and SFT checkpoint report for `d24_asp48_track`.

This version replaces the stale zero-valued rows in `report_v2.md` with the actual metrics from the fixed `skip5120` base eval logs, and adds the completed `r32` LoRA SFT confirmation results.

| Date (local) | Model Tag | Target Ratio | Eval Step | Train bpb | Val bpb | CORE | Eval Log | Status | Notes |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 2026-03-12 22:01 | `d24_asp48_track` | 24 | 615173 | 0.799226 | 0.908073 | 0.1494 | `/home/sun0115/nanochat-learn/notes/d24_asp48_track_s615173_base_eval_skip5120_2026-03-12_2201.log` | ok | Best base-eval val bpb so far |
| 2026-03-19 04:56 | `d24_asp48_track` | 32 | 820230 | 0.801367 | 0.919863 | 0.1514 | `/home/sun0115/nanochat-learn/notes/d24_asp48_track_s820230_base_eval_skip5120_2026-03-19_0456.log` | ok | Best CORE so far; best balanced base checkpoint |
| 2026-03-27 04:59 | `d24_asp48_track` | 40 | 1025288 | 0.784647 | 0.924377 | 0.1440 | `/home/sun0115/nanochat-learn/notes/d24_asp48_track_s1025288_base_eval_skip5120_2026-03-27_0459.log` | ok | Completed after bs1 resume; worse than `r32` on both comparable selection metrics |

## Comparison

- `r32` vs `r24`: CORE `+0.0020` for `r32`, but val bpb `+0.011790` for `r32` (worse because lower is better).
- `r40` vs `r32`: CORE `-0.0074` for `r40`, and val bpb `+0.004514` for `r40` (worse).
- `r40` vs `r24`: CORE `-0.0054` for `r40`, and val bpb `+0.016304` for `r40` (worse).
- The `r40` training run itself finished cleanly and saved `/home/sun0115/.cache/nanochat/base_checkpoints/d24_asp48_track/model_1025288.pt`.
- The `r40` training log reported inline `val bpb 0.917741` at the final step and `minimum validation bpb 0.901798`, but these trainer-side numbers are not the same measurement path as the fixed `base_eval` results above. For checkpoint selection, compare the `skip5120` base eval values, not the inline trainer validation.

## SFT Checkpoint Review

Completed LoRA SFT evaluation from the promoted `r32` base checkpoint:

| Date (local) | Run Tag | Base Checkpoint | Config | Best Step | Best Val bpb | GSM8K pass@8 | MMLU | SpellingBee | Status | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-03-27 11:29 / 18:41 | `d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc` | `d24_asp48_track @ 820230` | LoRA `r64`, `alpha=128`, `lr=1e-4`, `seq=1536`, `gc=on`, `iters=1000` | 1000 | 0.6151 | 0.20% (`2/1000`) | 21.40% (`214/1000`) | 0.00% (`0/256`) | completed | Training stayed stable and improved SFT val bpb monotonically, but external capability eval stayed weak; MMLU confirm needed the selected-position logits eval fix to avoid CUDA OOM |

Supporting logs:

- Training: `/home/sun0115/nanochat-learn/notes/sft_d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc.log`
- GSM8K pass@8: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_gsm8k_pass8_1000_2026-03-27_1154.log`
- MMLU: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_mmlu_1000_2026-03-27_1841.log`
- SpellingBee: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_spellingbee_1000_2026-03-27_1841.log`
- Confirm summary: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_confirm_summary_1000_2026-03-27_1841.txt`

Interpretation:

- This SFT run is not a promotion checkpoint.
- SFT-side validation bpb improved strongly (`0.7434 -> 0.6151`), but the external chat capability checks did not follow.
- `MMLU 21.4%` is below the `25%` random-choice baseline for 4-option multiple-choice tasks, so the current recipe is not preserving useful general capability well enough.
- `GSM8K pass@8 0.2%` and `SpellingBee 0/256` confirm that this exact 1k-step LoRA recipe is not yet the right production path.

## Recommendation

1. Promote `r32` step `820230` as the base checkpoint for the next stage.
2. Do not continue this exact base-training recipe beyond `r40`. The extra multi-day extension improved in-training loss but regressed on the held-out/base eval metrics used for selection.
3. Do not promote `d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc` as the main chat model. The confirm suite is too weak despite the attractive SFT val bpb.
4. Best next stage: stay on the `r32` base branch, but retune SFT rather than repeating this exact LoRA recipe. The most defensible next experiment is a lower-LR, shorter gated LoRA ablation from `r32`.
5. Concrete bracket to try next: keep the stable `1536_gc` geometry, but lower `lora_lr` to `3e-5` or `5e-5`, run `300-500` iterations first, and gate promotion on external evals, not SFT val bpb alone.
6. Keep eval OOM-safe in the next SFT cycle with the new categorical selected-logits path and `CAT_BATCH_SIZE=1`. That fix removed the MMLU CUDA OOM and should remain part of the standard eval path.
7. Treat `r40` as an archive/ablation checkpoint, not the promotion branch. If you want another base-only attempt later, change the late-phase recipe first instead of extending ratio again as-is. A lower-LR bridge or shorter retuned continuation around `r34-r36` is the only base-side follow-up that remains technically justified from the current evidence.
