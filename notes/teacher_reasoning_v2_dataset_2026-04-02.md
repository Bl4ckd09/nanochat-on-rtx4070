# Teacher Reasoning v2 Dataset

Date: 2026-04-02

## Purpose

`teacher_reasoning_v2` replaces the failed `teacher_reasoning_v1b` mix with a more targeted teacher-selected pool:

- `OpenThoughts-114k` for non-code math/science/puzzle/general reasoning
- `OpenR1-Math-220k` for verified math solutions
- filtered `Magpie-Ultra` for broad non-math glue

## Build

Builder:
- [build_teacher_reasoning_v2.py](/home/sun0115/nanochat-learn/nanochat/data/build_teacher_reasoning_v2.py)

Outputs:
- [teacher_reasoning_v2.jsonl](/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v2.jsonl)
- [teacher_reasoning_v2_metadata.json](/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v2_metadata.json)

## Composition

- total rows: `520`
- sources:
  - `OpenThoughts-114k`: `220`
  - `OpenR1-Math-220k`: `180`
  - `Magpie-Ultra`: `120`
- category counts:
  - `math`: `260`
  - `science`: `60`
  - `puzzle`: `50`
  - `general-reasoning`: `30`
  - `reasoning`: `50`
  - `data-analysis`: `40`
  - `information-seeking`: `30`

## Selection Notes

- `OpenThoughts` rows are filtered to remove code-heavy prompts.
- Only the solution section is retained from `OpenThoughts` assistant responses; the explicit thought block is stripped.
- `OpenR1` uses the `problem` and `solution` fields directly instead of the raw `messages` trace.
- `Magpie-Ultra` is used only for `reasoning`, `data-analysis`, and `information-seeking`.

## Training Path

Preset:
- `teacher_reasoning_v2`

Runner:
- [run_teacher_reasoning_v2.sh](/home/sun0115/nanochat-learn/nanochat/tools/automation/run_teacher_reasoning_v2.sh)

Backbone:
- base: `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- fixed `s768`
- `300` steps
- deterministic mode on
- `2` seeds
