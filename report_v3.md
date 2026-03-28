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

Completed SFT evaluation from the promoted `r32` base checkpoint:

| Date (local) | Run Tag | Base Checkpoint | Config | Best Step | Best Val bpb | GSM8K pass@8 | MMLU | SpellingBee | Status | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-03-27 11:29 / 18:41 | `d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc` | `d24_asp48_track @ 820230` | LoRA `r64`, `alpha=128`, `lr=1e-4`, `seq=1536`, `gc=on`, `iters=1000` | 1000 | 0.6151 | 0.20% (`2/1000`) | 21.40% (`214/1000`) | 0.00% (`0/256`) | completed | Training stayed stable and improved SFT val bpb monotonically, but external capability eval stayed weak; MMLU confirm needed the selected-position logits eval fix to avoid CUDA OOM |
| 2026-03-28 15:45 / 16:57 | `d24_r32_adamw_partial_lr005_2026-03-28_1545_s1024_gc` | `d24_asp48_track @ 820230` | Partial FT, `freeze_layers=18`, freeze embeddings/scalars, `paged_adamw8bit`, `seq=1024`, `tb=8192`, `iters=300`, `general_chat_reasoning` | 300 | 0.6563 | 0.80% (`8/1000`) | 26.60% (`266/1000`) | 0.39% (`1/256`) | completed | First feasible non-LoRA control on 4070 after fixing optimizer param filtering; full confirm beat the recent LoRA branch on GSM8K and MMLU |
| 2026-03-28 20:35 / 20:56 | `d24_r32_adamw_partial_lr005_1k_2026-03-28_2035_s1024_gc` | `d24_asp48_track @ 820230` | Same partial FT recipe, extended to `iters=1000` | 1000 | 0.5975 | pass@8 skipped; pass@1 `0.40%` (`1/250`) | 23.20% (`58/250`) | skipped | completed | Longer training improved SFT val bpb again, but the quick external gate regressed vs the 300-step checkpoint, indicating this recipe starts overfitting or drifting if extended unchanged |

Supporting logs:

- Training: `/home/sun0115/nanochat-learn/notes/sft_d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc.log`
- GSM8K pass@8: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_gsm8k_pass8_1000_2026-03-27_1154.log`
- MMLU: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_mmlu_1000_2026-03-27_1841.log`
- SpellingBee: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_spellingbee_1000_2026-03-27_1841.log`
- Confirm summary: `/home/sun0115/nanochat-learn/notes/d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc_best_s1000_confirm_summary_1000_2026-03-27_1841.txt`
- Partial FT training (300-step): `/home/sun0115/nanochat-learn/notes/sft_d24_r32_adamw_partial_lr005_2026-03-28_1545_s1024_gc.log`
- Partial FT full confirm (300-step): `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_lr005_2026-03-28_1545_s1024_gc_best_s300_confirm_summary_1000_2026-03-28_1657.txt`
- Partial FT training (1k-step): `/home/sun0115/nanochat-learn/notes/sft_d24_r32_adamw_partial_lr005_1k_2026-03-28_2035_s1024_gc.log`
- Partial FT quick gate (1k-step): `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_lr005_1k_2026-03-28_2035_s1024_gc_best_s1000_confirm_summary_250_2026-03-28_2056.txt`

Interpretation:

- This SFT run is not a promotion checkpoint.
- SFT-side validation bpb improved strongly (`0.7434 -> 0.6151`), but the external chat capability checks did not follow.
- `MMLU 21.4%` is below the `25%` random-choice baseline for 4-option multiple-choice tasks, so the current recipe is not preserving useful general capability well enough.
- `GSM8K pass@8 0.2%` and `SpellingBee 0/256` confirm that this exact 1k-step LoRA recipe is not yet the right production path.
- The partial-full-tune control changes that conclusion. Once real freezing reduced optimizer-state memory, the `300`-step partial FT run became feasible on the 4070 and produced the strongest recent external result of the `d24` branch: `GSM8K pass@8 0.8%`, `MMLU 26.6%`, `SpellingBee 0.39%`.
- Extending the same partial FT recipe to `1000` steps improved SFT val bpb further (`0.6563 -> 0.5975`) but regressed the quick external gate (`MMLU 27.2% @ 250` on the 300-step checkpoint's quick gate vs `23.2% @ 250` at 1k). For this recipe on this hardware, longer is not better by default.

## Recommendation

1. Promote `r32` step `820230` as the base checkpoint for the next stage.
2. Do not continue this exact base-training recipe beyond `r40`. The extra multi-day extension improved in-training loss but regressed on the held-out/base eval metrics used for selection.
3. Do not promote `d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc` as the main chat model. The confirm suite is too weak despite the attractive SFT val bpb.
4. The best current chat-side candidate is `d24_r32_adamw_partial_lr005_2026-03-28_1545_s1024_gc` at step `300`, not the later 1k-step continuation. Treat the 300-step checkpoint as the current promotion candidate for this recipe family.
5. The current bottleneck is no longer base capacity or raw memory fit. It is recipe drift: the partial FT path works, but capability falls off if the same setup is trained too long unchanged.
6. Best next stage: stay on the `r32` base branch and continue with partial full-tuning, not LoRA LR sweeps. Keep real freezing, `paged_adamw8bit`, and OOM-safe eval.
7. Concrete bracket to try next: run shorter gated partial FT probes around the winning 300-step regime, while changing recipe rather than duration. The most defensible next levers are lower effective LR, narrower/high-quality SFT data mix, or a slightly shallower freeze boundary.
8. Treat `r40` as an archive/ablation checkpoint, not the promotion branch. If you want another base-only attempt later, change the late-phase recipe first instead of extending ratio again as-is.
