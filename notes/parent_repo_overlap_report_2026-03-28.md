# Parent vs Repo Overlap Report

Generated: 2026-03-28 UTC
Repo: `Bl4ckd09/nanochat-experiments`
Remote head checked: `5dbf896`

## Scope

Compared files in:
- parent workspace: `/home/sun0115/nanochat-learn/`
- repo workspace: `/home/sun0115/nanochat-learn/nanochat/`

The comparison only considered files that exist outside the repo and also exist inside the repo at the same relative path.

## Summary After Sync

- Total same-path overlaps: `278`
- Identical overlaps: `277`
- Different overlaps: `1`

Folder summary:
- `notes/`: fully mirrored after sync
  - common: `277`
  - parent-only: `0`
  - repo-only: `0`
  - different content: `0`
- `scripts/`: no same-path overlaps
  - parent-only filenames: `28`
  - repo-only filenames: `10`
  - common filenames: `0`
- root-level same-path overlap: `report.md`

## Synced In This Pass

The following parent `notes/` files were copied into repo `notes/` because they existed outside the repo but were not yet mirrored inside it:

- `notes/d24_eval_then_next_2026-03-20_1529_results.tsv`
- `notes/d24_sft_recovery_2026-03-20_1216_results.tsv`
- `notes/d24_sft_recovery_v2_2026-03-21_0105_results.tsv`
- `notes/sft_d24_r32_lora_maxoom_2026-03-27_2007_meta.env`
- `notes/sft_d24_r32_lora_maxoom_lr5e5_2026-03-28_0107_meta.env`
- `notes/sft_d24_r32_lora_nextbest_2026-03-19_1310_meta.env`
- `notes/sft_d24_r32_lora_nextbest_2026-03-27_1127_meta.env`
- `notes/sft_d24_r32_lora_nextbest_2026-03-27_1129_meta.env`

Also updated in repo from the newer parent copy:
- `notes/d24_r32_lora_maxoom_2026-03-27_2007_monitor_2026-03-27_2102.log`

## Canonical Decisions

### 1. `report.md`

Parent file:
- `/home/sun0115/nanochat-learn/report.md`

Repo file:
- `/home/sun0115/nanochat-learn/nanochat/report.md`

Decision: **repo copy is canonical**.

Reason:
- the repo copy is the intentionally versioned report in the pushed repository
- the parent copy is an older standalone auto-generated report outside Git
- current project decisions are already being tracked in repo-native reports such as `report_v2.md` and `report_v3.md`
- overwriting repo `report.md` with the parent copy would mix two different reporting styles without review

Operational note:
- if you still want the parent `report.md` preserved in Git, it should be imported under a different name, not overwrite repo `report.md`

### 2. `notes/d24_r32_lora_maxoom_2026-03-27_2007_monitor_2026-03-27_2102.log`

Decision: **parent copy was canonical and has now been synced into repo**.

Reason:
- the parent file was newer and longer
- the only delta was extra trailing monitor snapshots captured after the earlier repo sync
- no semantic conflict existed; parent simply had the fuller log tail

## Remaining Drift

After the sync in this pass, the only same-path duplicate outside the repo that still differs from the repo copy is:

- `report.md`

## Recommendation

- Treat repo `notes/` as the archival Git mirror of runtime artifacts from parent `notes/`
- Treat repo `report_v3.md` as the current evaluation report for model-selection work
- If you want parent `report.md` preserved, import it as an archival note with a distinct filename
