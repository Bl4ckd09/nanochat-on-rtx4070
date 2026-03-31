# Final Export State

This project keeps two separate exported winners because they serve different roles.

## Base Champion

Path:
- `/home/sun0115/nanochat-learn/exports/base_champion_r32_820230`

Status:
- `stable base champion`
- model tag: `d24_asp48_track`
- checkpoint step: `820230`

Metrics:
- train bpb: `0.801367`
- val bpb: `0.919863`
- CORE: `0.1514`

Layout:
- `full/`: model + optimizer + metadata + checksums
- `weights_only/`: model-only export

## Chat Champion

Path:
- `/home/sun0115/nanochat-learn/exports/chat_champion_mixv2_s768_s300`

Status:
- `provisional chat champion`
- run: `d24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc`
- checkpoint step: `300`
- base: `d24_asp48_track @ 820230`

Confirmed metrics:
- GSM8K pass@8: `46/1000 = 4.60%`
- MMLU: `274/1000 = 27.40%`
- SpellingBee: `1/256 = 0.39%`

Why provisional:
- direct replication failed the quick gate
- deterministic seed sweep failed `0/3`
- later `mixv3`, `mixv4`, curriculum, stage-B boosters, and curated-v1 failed to beat it

Release:
- `https://github.com/Bl4ckd09/nanochat-experiments/releases/tag/champion-d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948`

## Export Rule

Do not collapse these into one final model.

- Base champion = stable foundation
- Chat champion = best observed task-adapted artifact

This split is intentional and is the default export layout going forward.
