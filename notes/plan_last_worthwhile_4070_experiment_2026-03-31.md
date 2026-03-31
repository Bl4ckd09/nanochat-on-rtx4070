# Last Worthwhile 4070 Experiment Plan

## Objective

Run one final high-ROI experiment on the current `RTX 4070 12GB` setup before declaring the current `d24/r32` recipe family saturated.

This plan assumes:

- keep the promoted base checkpoint: `d24_asp48_track @ 820230`
- keep the working partial-FT backbone:
  - `freeze_layers=20`
  - freeze embeddings/scalars
  - `paged_adamw8bit`
  - `seq=768`
  - `total_batch_size=7680`
  - `300` steps
- stop small static mix sweeps on the old data family

## Why This Is The Last Worthwhile Local Experiment

Recent evidence shows:

- `mixv2_s768_300` is the best single observed run:
  - GSM8K pass@8 `4.60%`
  - MMLU `27.40%`
  - SpellingBee `0.39%`
- direct replication failed
- deterministic seed sweep failed `0/3`
- later `mixv3`, `mixv4`, curriculum, softer stage-B boosters, and `reasoning_curated_v1` all failed to beat it

So the remaining lever is not more of the same recipe search.
The remaining lever is **higher-quality data**, with fewer but cleaner examples.

## Experiment Definition

### Name

`reasoning_manual_v1`

### Fixed Backbone

- base: `d24_asp48_track @ 820230`
- source: `base`
- partial FT
- `freeze_layers=20`
- freeze embeddings/scalars
- optimizer: `paged_adamw8bit`
- `max_seq_len=768`
- `device_batch_size=1`
- `total_batch_size=7680`
- `num_iterations=300`

### Data Strategy

Replace large mixed synthetic/task-weight sweeps with a smaller manually curated blend:

1. `manual_reasoning_chat.jsonl`
- target size: `1k-3k` strong assistant conversations
- focus:
  - short multi-step reasoning
  - math explanation
  - grounded QA
  - refusal/format discipline
- avoid:
  - noisy casual chat filler
  - synthetic identity/personality overfitting
  - spelling drills in the main branch

2. `GSM8K(main, train)`
- keep as a booster, but not at extreme weight

3. `MMLU(auxiliary_train, train)`
- keep as breadth anchor

### Suggested Mix

Use a compact blend roughly like:

- `manual_reasoning_chat.jsonl` x3
- `GSM8K(main, train)` x2
- `MMLU(auxiliary_train, train)` x1
- optional `SmolTalk(train, stop=20000)` x1 only if needed for conversational glue

The intent is:

- fewer total examples
- higher signal per update
- less objective conflict than `mixv2/mixv3/mixv4`

## Execution Stages

### Stage 0: Data Build

Create `manual_reasoning_chat.jsonl`.

Requirements:

- clean chat format compatible with `tasks.customjson`
- each sample should be high-value
- no bulk noisy dumps
- no spelling/identity specialist data in this file

Target deliverable:

- one curated JSONL file under a tracked path, e.g.
  - `data/manual_reasoning_chat_v1.jsonl`

### Stage 1: Single Seed Probe

Run one seed:

- seed `42`
- exact backbone above
- `300` steps

Quick gate:

- evaluate on `500` problems
- require:
  - `MMLU >= 27.0%`
  - `GSM8K pass@1 > 0`

If quick gate fails:

- stop the branch
- do not continue to more seeds

### Stage 2: Full Confirm

If Stage 1 passes:

- run full confirm:
  - GSM8K pass@8 `1000`
  - MMLU `1000`
  - SpellingBee `256`

Promotion band to beat current provisional champion:

- GSM8K pass@8 `>= 4.60%`
- MMLU `>= 27.40%`

### Stage 3: One Replication Only

If Stage 2 beats or matches the promotion band:

- run exactly one replication at seed `43`

Promotion rule:

- if replicate lands in the same band, promote
- if replicate collapses badly, keep `mixv2_s768_300` as provisional champion

## Stop Rules

Stop the 4070 branch if any of these happen:

1. Stage 1 quick gate fails
2. Stage 2 full confirm does not beat `mixv2_s768_300`
3. Stage 3 replication fails to hold roughly the same band

If stopped, treat the current family as saturated on this hardware.

## What Not To Do

Do not spend more time on:

- `r40` continuation with the same base recipe
- more `mixvN` static sweeps from the existing data family
- longer `1k` continuations of the same partial-FT branch
- LoRA-first sweeps for the main branch
- mixing spelling/identity into the generalist checkpoint

## Success Criteria

Minimum success:

- reproducible result near or above:
  - GSM8K pass@8 `4.60%`
  - MMLU `27.40%`

Strong success:

- clear gain over current champion on both:
  - GSM8K pass@8
  - MMLU

## If This Fails

The next productive move is not another local hyperparameter sweep.

Choose one:

1. stronger hardware
2. stronger base model
3. separate specialist branch
4. a truly better curated dataset built with more manual effort

## Current Champion Reference

- base champion: `d24_asp48_track @ 820230`
- chat champion: `d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc @ 300`
- confirm:
  - GSM8K pass@8 `4.60%`
  - MMLU `27.40%`
  - SpellingBee `0.39%`
