# teacher_distilled_v1 Plan

## Goal

Set up a teacher-distilled reasoning branch that is materially different from the failed `teacher_reasoning_v2/v3` open-source mixture paths.

## Raw dataset format

`data/teacher_distilled_v1_raw.jsonl` uses one JSON object per line with:

- `id`
- `source`
- `teacher_model`
- `category`
- `quality`
- `tags`
- `messages`

`messages` must alternate `user`/`assistant`, start with `user`, stay short, and avoid long chain-of-thought spillover.

## Build path

1. Validate raw rows with `data/validate_teacher_distilled_v1.py`
2. Build training JSONL with `data/build_teacher_distilled_v1.py`
3. Emit:
   - `data/teacher_distilled_v1.jsonl`
   - `data/teacher_distilled_v1_metadata.json`

## Launch path

Use `tools/automation/run_teacher_distilled_v1.sh`.

Current launch guard:
- minimum raw rows: `200`
- fixed geometry: `s768`
- seeds: `42,43`
- deterministic mode: on
- quick gate: `500`
- full confirm only on pass

## Training backbone

- base: `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- `300` steps

## Promotion rule

- require `2` seeds
- `GSM8K pass@8 >= 4.60%`
- `MMLU >= 27.40%`

## Notes

- The starter raw file is only a schema seed and should not be launched as-is.
- The intended next real dataset should come from a stronger teacher or carefully rewritten teacher-style examples with short answers and minimal MCQ bias.
