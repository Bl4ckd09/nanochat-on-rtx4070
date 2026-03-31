# Champion Release Prep 2026-03-31

Current provisional champion export:
- local export dir: `/home/sun0115/nanochat-learn/exports/d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948`
- local archive: `/home/sun0115/nanochat-learn/exports/d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948.tar.gz`
- local release prep dir: `/home/sun0115/nanochat-learn/exports/release_prep_d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948`

Release target:
- repo: `Bl4ckd09/nanochat-experiments`
- release tag: `champion-d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948`
- release URL: `https://github.com/Bl4ckd09/nanochat-experiments/releases/tag/untagged-3ca7e6148b0b37070403`

Reason for split assets:
- the full `.tar.gz` archive is larger than GitHub's per-asset `2 GiB` release limit

Split release assets:
- `d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948.tar.gz.part-00`
- `d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948.tar.gz.part-01`
- `SHA256SUMS_split.txt`

Helpers:
- upload helper: `/home/sun0115/nanochat-learn/exports/release_prep_d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948/upload_github_release_parts.sh`
- reconstruct helper: `/home/sun0115/nanochat-learn/exports/release_prep_d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948/reconstruct_archive.sh`
- release notes: `/home/sun0115/nanochat-learn/exports/release_prep_d24_r32_adamw_partial_fr20_mixv2_s768_s300_champion_2026-03-30_190948/github_release_notes.md`
