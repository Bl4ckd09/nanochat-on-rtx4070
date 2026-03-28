# Parent Workspace Archive

Archived from `/home/sun0115/nanochat-learn` on 2026-03-28.

This archive preserves the remaining parent-workspace files that were reviewed but not previously represented in the repo as first-class tracked sources.

Included:
- `scripts/` snapshot from the parent workspace
- `env/setup.md`
- root planning/reporting notes:
  - `plan.md`
  - `plan-d24-next-2026-03-13.md`
  - `plan-r24-to-r40-wandb-2026-03-13.md`
  - `plan-sft-d24-r32-2026-03-19.md`
  - `plan-sft-recovery-sweep-2026-03-19.md`
  - `report.md`
  - `runlog.md`
  - `insight.md`
- `exports/` manifests only (`SHA256SUMS.txt`, `meta_*.json` where available)

Excluded:
- multi-GB model and optimizer binaries under `../exports/`
- live runtime directories such as `notes/`

Excluded export payloads that remain outside Git:
- `exports/d24_asp48_track_s615173_2026-03-13_154421/model_615173.pt`
- `exports/d24_asp48_track_s615173_2026-03-13_154421/optim_615173_rank0.pt`
- `exports/d24_asp48_track_s820230_2026-03-19_120117/model_820230.pt`
- `exports/d24_asp48_track_s820230_2026-03-19_120117/optim_820230_rank0.pt`
- `exports/d24_asp48_track_weights_s820230_2026-03-19_122254/model_820230.pt`

Context:
- current canonical repo-side overlap review: `notes/parent_repo_overlap_report_2026-03-28.md`
- current model-selection report: `report_v3.md`
