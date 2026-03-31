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

Current chat-side champion as of `2026-03-30`:

- `d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc` at step `300`
- full confirm: `GSM8K pass@8 4.60%`, `MMLU 27.40%`, `SpellingBee 0.39%`
- confirm summary: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt`
- status: strongest single confirmed run on the `d24/r32` chat branch, but still **provisional** because later direct replication and seed sweeps did not reproduce it reliably

| Date (local) | Run Tag | Base Checkpoint | Config | Best Step | Best Val bpb | GSM8K pass@8 | MMLU | SpellingBee | Status | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-03-27 11:29 / 18:41 | `d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc` | `d24_asp48_track @ 820230` | LoRA `r64`, `alpha=128`, `lr=1e-4`, `seq=1536`, `gc=on`, `iters=1000` | 1000 | 0.6151 | 0.20% (`2/1000`) | 21.40% (`214/1000`) | 0.00% (`0/256`) | completed | Training stayed stable and improved SFT val bpb monotonically, but external capability eval stayed weak; MMLU confirm needed the selected-position logits eval fix to avoid CUDA OOM |
| 2026-03-28 15:45 / 16:57 | `d24_r32_adamw_partial_lr005_2026-03-28_1545_s1024_gc` | `d24_asp48_track @ 820230` | Partial FT, `freeze_layers=18`, freeze embeddings/scalars, `paged_adamw8bit`, `seq=1024`, `tb=8192`, `iters=300`, `general_chat_reasoning` | 300 | 0.6563 | 0.80% (`8/1000`) | 26.60% (`266/1000`) | 0.39% (`1/256`) | completed | First feasible non-LoRA control on 4070 after fixing optimizer param filtering; full confirm beat the recent LoRA branch on GSM8K and MMLU |
| 2026-03-28 20:35 / 20:56 | `d24_r32_adamw_partial_lr005_1k_2026-03-28_2035_s1024_gc` | `d24_asp48_track @ 820230` | Same partial FT recipe, extended to `iters=1000` | 1000 | 0.5975 | pass@8 skipped; pass@1 `0.40%` (`1/250`) | 23.20% (`58/250`) | skipped | completed | Longer training improved SFT val bpb again, but the quick external gate regressed vs the 300-step checkpoint, indicating this recipe starts overfitting or drifting if extended unchanged |
| 2026-03-28 21:49 / 23:03 | `d24_r32_adamw_partial_fr20_2026-03-28_2149_s1024_gc` | `d24_asp48_track @ 820230` | Partial FT, `freeze_layers=20`, freeze embeddings/scalars, `paged_adamw8bit`, `seq=1024`, `tb=8192`, `iters=300`, `general_chat_reasoning` | 300 | 0.6610 | 1.20% (`12/1000`) | 27.40% (`274/1000`) | 0.00% (`0/256`) | completed | More conservative freeze boundary improved GSM8K and MMLU over the `freeze_layers=18` control, but spelling dropped to zero |
| 2026-03-29 05:18 / 06:07 | `d24_r32_adamw_partial_fr20_mixv1_2026-03-29_0518_s1024_gc` | `d24_asp48_track @ 820230` | Same partial FT backbone, `reasoning_focus_v1`, `seq=1024`, `iters=300` | 300 | 0.6670 | 3.80% (`38/1000`) | 26.60% (`266/1000`) | 0.00% (`0/256`) | completed | First clear data-mix win for GSM8K, but MMLU fell relative to the `fr20` balanced run |
| 2026-03-29 10:40 / 12:00 | `d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc` | `d24_asp48_track @ 820230` | Same partial FT backbone, `reasoning_focus_v2`, `seq=768`, `tb=7680`, `iters=300` | 300 | 0.6981 | 4.60% (`46/1000`) | 27.40% (`274/1000`) | 0.39% (`1/256`) | completed, provisional champion | Strongest single confirmed run on this branch; beat `mixv1` on all three confirmed metrics, but exact replication and later seed sweep did not hold |

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
- Partial FT `freeze_layers=20` full confirm: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_2026-03-28_2149_s1024_gc_best_s300_confirm_summary_1000_2026-03-28_2303.txt`
- `mixv1` full confirm: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv1_2026-03-29_0518_s1024_gc_best_s300_confirm_summary_1000_2026-03-29_0607.txt`
- `mixv2` full confirm: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt`

## Follow-up Branches After `mixv2`

- Direct replication of `mixv2_s768_300` failed the quick gate: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_rep_2026-03-29_1634_s768_gc_best_s300_confirm_summary_250_2026-03-29_1641.txt`
- Deterministic `3`-seed sweep of the same recipe failed `0/3` seeds at quick gate: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_seed_sweep_2026-03-29_1808_decision.md`
- `mixv3` quick-gate failed: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv3_2026-03-29_2142_s768_gc_best_s300_confirm_summary_250_2026-03-29_2150.txt`
- `mixv4` reached full confirm but stayed below `mixv2`: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv4_2026-03-29_2226_s768_gc_best_s300_confirm_summary_1000_2026-03-29_2313.txt`
- `curriculum_v1` failed the stage-B quick gate: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_curriculum_v1_2026-03-30_0635_decision.md`
- Stage-B-only softer boosters also failed to beat `mixv2`: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_stageb_booster_v2_2026-03-30_0909_decision.md`
- `reasoning_curated_v1` failed both seeds at the tighter `500`-problem quick gate: `/home/sun0115/nanochat-learn/notes/reasoning_curated_v1_2026-03-30_1542_decision.md`

Interpretation:

- This SFT run is not a promotion checkpoint.
- SFT-side validation bpb improved strongly (`0.7434 -> 0.6151`), but the external chat capability checks did not follow.
- `MMLU 21.4%` is below the `25%` random-choice baseline for 4-option multiple-choice tasks, so the current recipe is not preserving useful general capability well enough.
- `GSM8K pass@8 0.2%` and `SpellingBee 0/256` confirm that this exact 1k-step LoRA recipe is not yet the right production path.
- The partial-full-tune control changes that conclusion. Once real freezing reduced optimizer-state memory, the `300`-step partial FT run became feasible on the 4070 and produced the strongest recent external result of the `d24` branch: `GSM8K pass@8 0.8%`, `MMLU 26.6%`, `SpellingBee 0.39%`.
- Extending the same partial FT recipe to `1000` steps improved SFT val bpb further (`0.6563 -> 0.5975`) but regressed the quick external gate (`MMLU 27.2% @ 250` on the 300-step checkpoint's quick gate vs `23.2% @ 250` at 1k). For this recipe on this hardware, longer is not better by default.
- The best observed reasoning/chat checkpoint is now `mixv2_s768_300`, not the earlier `s1024` controls. The improvement came from changing both geometry and data mix, not from training longer.
- That best run remains provisional. Repeats, seed sweeps, later static mix variants, and two-stage booster follow-ups all failed to beat or stably reproduce it. The limitation is now recipe robustness and data quality, not raw VRAM fit.

## Recommendation

1. Promote `r32` step `820230` as the base checkpoint for the next stage.
2. Do not continue this exact base-training recipe beyond `r40`. The extra multi-day extension improved in-training loss but regressed on the held-out/base eval metrics used for selection.
3. Do not promote `d24_r32_lora_nextbest_2026-03-27_1129_s1536_gc` as the main chat model. The confirm suite is too weak despite the attractive SFT val bpb.
4. Mark `d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc` at step `300` as the **current chat-side champion** for this project. It is the strongest single confirmed run so far on the consumer-GPU branch.
5. Keep that champion labeled **provisional**, not fully promoted. It is the best observed run, but it did not replicate cleanly under direct repeat or seed sweep.
6. Stop same-family `mixvN`, booster, and longer-duration sweeps on this exact backbone. The current bottleneck is no longer base capacity or raw memory fit; it is recipe robustness and data quality.
7. Treat this `d24/r32` partial-FT recipe family as near-saturated on the current `RTX 4070` setup unless you introduce a materially different lever such as a genuinely better curated dataset, a separate specialist branch, or stronger hardware.
8. Treat `r40` as an archive/ablation checkpoint, not the promotion branch. If you want another base-only attempt later, change the late-phase recipe first instead of extending ratio again as-is.
