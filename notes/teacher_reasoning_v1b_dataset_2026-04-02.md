# Teacher Reasoning v1b Dataset

Date: 2026-04-02

## Purpose

`teacher_reasoning_v1b` is the first trainable teacher-generated reasoning branch for the single-RTX-4070 pipeline.

It rebalances the original `teacher_reasoning_v1` pool to reduce math dominance before spending GPU time on SFT.

## Build

Builder:
- [build_teacher_reasoning_v1.py](/home/sun0115/nanochat-learn/nanochat/data/build_teacher_reasoning_v1.py)

Outputs:
- [teacher_reasoning_v1b.jsonl](/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v1b.jsonl)
- [teacher_reasoning_v1b_metadata.json](/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v1b_metadata.json)

## Composition

- total rows: `600`
- sources:
  - `Magpie-Ultra`: `400`
  - `Orca-Math`: `200`
- target category mix:
  - `reasoning`: `140`
  - `data-analysis`: `120`
  - `information-seeking`: `90`
  - `math`: `250`

## Rationale

The earlier `teacher_reasoning_v1` candidate was too math-heavy (`350/600` math rows). `v1b` keeps teacher-generated reasoning pressure while making the set less likely to overfit toward GSM8K-like behavior at the expense of broader MMLU behavior.

## Training Path

Preset:
- `teacher_reasoning_v1b`

Runner:
- [run_teacher_reasoning_v1b.sh](/home/sun0115/nanochat-learn/nanochat/tools/automation/run_teacher_reasoning_v1b.sh)

Backbone:
- base: `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- fixed `s768`
- `300` steps
- deterministic mode on
- `2` seeds

## Promotion Rule

- both seeds must pass the quick gate and beat the current provisional champion band:
  - GSM8K pass@8 `>= 4.60%`
  - MMLU `>= 27.40%`
