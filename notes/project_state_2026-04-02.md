# Project State 2026-04-02

## Stable Base Champion

- model tag: `d24_asp48_track`
- checkpoint step: `820230`
- status: `stable base champion`
- reason: this is still the best balanced base checkpoint on the fixed base-eval path

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

Why still provisional:

- direct replication failed the quick gate
- deterministic seed sweep failed `0/3`
- later `mixv3`, `mixv4`, curriculum, stage-B booster, curated-v1, and manual-v2 all failed to beat it cleanly

## Best Observed Unstable Artifact

- run: `reasoning_manual_v1_seed42_2026-04-01_1315_s768_gc`
- status: `best observed but unstable`
- confirm summary: `/home/sun0115/nanochat-learn/notes/reasoning_manual_v1_seed42_2026-04-01_1315_s768_gc_best_s300_confirm_summary_1000_2026-04-01_1428.txt`

Metrics:

- GSM8K pass@8: `6.70%`
- MMLU: `27.60%`
- SpellingBee: `0.00%`

Why not promoted:

- same sweep's companion seed failed the promotion rule
- fixed-`s768` replication preserved GSM8K strength but lost too much MMLU
- the branch remained unstable under repeat runs

## Latest Closure

`manual_reasoning_v2` was the final materially different local data branch on the current 4070 stack.

Result:

- seed `42`: quick gate failed at `MMLU 25.40%`
- seed `43`: quick gate failed at `MMLU 23.00%`
- decision: `recipe_failed_quick_gate`

## Do Not Repeat

- more `mixvN` static-mix sweeps on the same backbone
- longer continuations of the same `300`-step partial-FT recipe
- more LoRA-first main-branch sweeps
- more `r40`-style continuation on the same base recipe

## Next Real Improvement Path

The next worthwhile branch is not another local manual-vN sweep. It is a teacher-synthetic reasoning branch with targeted selection.

Reference plan:

- `/home/sun0115/nanochat-learn/nanochat/notes/plan_teacher_synthetic_reasoning_2026-04-02.md`

If that branch is not pursued, treat the project as closed around:

1. stable base champion: `r32 @ 820230`
2. official provisional chat champion: `mixv2_s768_300`
3. best observed unstable artifact: `manual_reasoning_v1_seed42`
