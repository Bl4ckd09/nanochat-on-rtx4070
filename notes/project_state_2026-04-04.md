# Project State 2026-04-04

## Stable Base Champion

- model tag: `d24_asp48_track`
- checkpoint step: `820230`
- status: `stable base champion`
- reason: this remains the best balanced base checkpoint on the fixed base-eval path; later `r40` did not improve the selection metrics.

## Official Provisional Chat Champion

- run: `d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc`
- base: `d24_asp48_track @ 820230`
- step: `300`
- status: `official provisional chat champion`
- confirm summary: `/home/sun0115/nanochat-learn/notes/d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt`

Metrics:

- GSM8K pass@8: `4.60%`
- MMLU: `27.40%`
- SpellingBee: `0.39%`

Why it is still provisional:

- direct replication failed the quick gate
- deterministic seed sweeps did not reproduce it reliably
- later `mixv3`, `mixv4`, curriculum, booster, curated, teacher, and distilled branches all failed to beat it cleanly

## Best Observed Unstable Artifact

- run: `reasoning_manual_v1_seed42_2026-04-01_1315_s768_gc`
- status: `best observed but unstable`
- confirm summary: `/home/sun0115/nanochat-learn/notes/reasoning_manual_v1_seed42_2026-04-01_1315_s768_gc_best_s300_confirm_summary_1000_2026-04-01_1428.txt`

Metrics:

- GSM8K pass@8: `6.70%`
- MMLU: `27.60%`
- SpellingBee: `0.00%`

Why it is not promoted:

- companion seeds failed the promotion rule
- fixed-`s768` replication preserved GSM8K strength but lost too much MMLU
- the branch stayed unstable under repeat runs

## Latest Negative Evidence

The following branches were completed and should now be treated as closed failures on the current `RTX 4070 12GB` stack:

- `teacher_reasoning_v2`: clean rerun after disk fix, but both seeds still failed the quick gate
- `teacher_reasoning_v3`: one seed nearly cleared MMLU but produced zero GSM8K pass@1; the other failed the internal gate
- `teacher_distilled_v1`: both seeds failed the internal loss gate before external eval
- `teacher_distilled_v2`: rebalanced distilled mix with more `Magpie`, fewer `OpenThoughts`, and manual anchors; both seeds still failed the internal loss gate

## Closure

The current `d24/r32` consumer-GPU recipe family is now best treated as closed around these artifacts:

1. stable base champion: `r32 @ 820230`
2. official provisional chat champion: `mixv2_s768_300`
3. best observed unstable artifact: `manual_reasoning_v1_seed42`

## Do Not Continue On This Stack

- more `mixvN` static-mix sweeps
- more `manual_vN` broadening sweeps
- more `teacher_reasoning` remixes of the same open-source pools
- more `teacher_distilled` variants with only filtering and quota tweaks
- longer continuations of the same `300`-step partial-FT recipe
- more LoRA-first main-branch sweeps

## Next Real Improvement Path

Only continue if you introduce a materially different lever:

1. stronger teacher-generated or teacher-distilled data with a qualitatively different source pipeline
2. a stronger base model
3. stronger hardware
4. a separate specialist branch for narrow capability targets instead of one generalist chat branch
