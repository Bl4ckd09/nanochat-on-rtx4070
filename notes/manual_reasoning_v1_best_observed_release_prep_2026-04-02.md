# Manual Reasoning v1 Best-Observed Release Prep 2026-04-02

Release target:
- repo: `Bl4ckd09/nanochat-experiments`
- release tag: `unstable-manual-reasoning-v1-seed42-2026-04-02`
- release URL: `https://github.com/Bl4ckd09/nanochat-experiments/releases/tag/unstable-manual-reasoning-v1-seed42-2026-04-02`

Artifact:
- local export dir: `/home/sun0115/nanochat-learn/exports/manual_reasoning_v1_seed42_best_observed_2026-04-02_0435`
- local archive: `/home/sun0115/nanochat-learn/exports/manual_reasoning_v1_seed42_best_observed_2026-04-02_0435.tar.gz`
- local release prep dir: `/home/sun0115/nanochat-learn/exports/release_prep_manual_reasoning_v1_seed42_best_observed_2026-04-02_0435`

Metrics:
- GSM8K pass@8: `6.70%`
- MMLU: `27.60%`
- SpellingBee: `0.00%`

Status:
- strongest single-run reasoning result observed on the project
- not promoted
- paired seed failed the promotion rule, so the artifact remains best-observed but unstable

Uploaded release assets:
- `manual_reasoning_v1_seed42_best_observed_2026-04-02_0435.tar.gz.part-00`
- `manual_reasoning_v1_seed42_best_observed_2026-04-02_0435.tar.gz.part-01`
- `SHA256SUMS_split.txt`

Reconstruct after download:
```bash
cat manual_reasoning_v1_seed42_best_observed_2026-04-02_0435.tar.gz.part-* \
  > manual_reasoning_v1_seed42_best_observed_2026-04-02_0435.tar.gz
sha256sum manual_reasoning_v1_seed42_best_observed_2026-04-02_0435.tar.gz
```
