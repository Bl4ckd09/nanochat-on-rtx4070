# manual_reasoning_v2 plan

## Goal

Run one materially different final 4070 experiment by improving data quality rather than continuing the existing mix sweep family.

## Backbone

- base: `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- fixed `s768`
- `300` steps

## Data change

- dataset preset: `reasoning_manual_v2`
- manual set expanded from `104` rows in `v1` to `207` rows in `v2`
- lower manual oversampling pressure: `x2` instead of `x3`
- keep `GSM8K x2`
- keep `MMLU auxiliary x1`
- keep only small `SmolTalk` glue

## Promotion rule

- quick gate on `500` examples
- require `MMLU >= 27.0%`
- require non-zero `GSM8K pass@1`
- full confirm only on quick-gate pass
- require `2` seeds to clear the band to replace the current provisional champion
- promotion band:
  - `GSM8K pass@8 >= 4.60%`
  - `MMLU >= 27.40%`

## Seeds

- `42,43`
- deterministic mode on

## Interpretation

- if both seeds clear the band: promote `manual_reasoning_v2`
- otherwise: hold `mixv2_s768_300` as provisional champion
