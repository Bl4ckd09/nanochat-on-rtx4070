# Base Champion Release Prep 2026-03-31

Base champion export:
- split export path: `/home/sun0115/nanochat-learn/exports/base_champion_r32_820230`
- full archive: `/home/sun0115/nanochat-learn/exports/release_prep_base_champion_r32_820230_2026-03-31/base_champion_r32_820230_full_2026-03-31.tar.gz`
- weights-only archive: `/home/sun0115/nanochat-learn/exports/release_prep_base_champion_r32_820230_2026-03-31/base_champion_r32_820230_weights_only_2026-03-31.tar.gz`
- local release prep dir: `/home/sun0115/nanochat-learn/exports/release_prep_base_champion_r32_820230_2026-03-31`

Release target:
- repo: `Bl4ckd09/nanochat-experiments`
- release tag: `base-champion-d24_asp48_track-r32-s820230-2026-03-31`
- release URL: `https://github.com/Bl4ckd09/nanochat-experiments/releases/tag/base-champion-d24_asp48_track-r32-s820230-2026-03-31`

Reason for split assets:
- the full archive and the weights-only archive both exceed GitHub's per-asset `2 GiB` release limit

Split release assets:
- `base_champion_r32_820230_full_2026-03-31.tar.gz.part-00`
- `base_champion_r32_820230_full_2026-03-31.tar.gz.part-01`
- `base_champion_r32_820230_full_2026-03-31.tar.gz.part-02`
- `base_champion_r32_820230_weights_only_2026-03-31.tar.gz.part-00`
- `base_champion_r32_820230_weights_only_2026-03-31.tar.gz.part-01`
- `SHA256SUMS_split.txt`

Helpers:
- upload helper: `/home/sun0115/nanochat-learn/exports/release_prep_base_champion_r32_820230_2026-03-31/upload_github_release_parts.sh`
- reconstruct helper: `/home/sun0115/nanochat-learn/exports/release_prep_base_champion_r32_820230_2026-03-31/reconstruct_archives.sh`
- release notes: `/home/sun0115/nanochat-learn/exports/release_prep_base_champion_r32_820230_2026-03-31/github_release_notes.md`
