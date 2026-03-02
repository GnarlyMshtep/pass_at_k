# Fix DVC / Git Conflicts for Checkpoints

## Problem
Checkpoints under `logs/VerlRun/02/26/womonitor_backdoor_qwen3_8b_removeafterc_01_26_p8lr4f2g/` were never pushed to the DVC remote (S3).

### Root Cause
The `.dvc` file at that path is **git-ignored**, causing DVC to refuse all operations (`dvc push`, `dvc status`) with:
```
ERROR: bad DVC file name '...checkpoints.dvc' is git-ignored.
```

### Evidence
- `.dir` manifest hash `1c88a7ce1a594efa9d4cfa37f27b4ca0` is **not on S3** and **not in local DVC cache**.
- The `.dvc` file IS tracked by git (`git ls-files` shows it), but DVC's own check thinks it's ignored — likely a `.gitignore` rule in `logs/` or a parent directory.
- The target directory `checkpoints/subtle_reasoning_repro/hidden_womonitor_backdoor_qwen3_8b_removeaftercodeformatter/` is an **empty dir** with no DVC tracking — it was never set up as a DVC-tracked path.

### Affected Runs
| Log dir | .dvc hash | nfiles | size |
|---------|-----------|--------|------|
| `womonitor_backdoor_qwen3_8b_removeafterc_01_26_p8lr4f2g` | `1c88a7ce...` | 98 | ~393 GB |
| `wmonitor_backdoor_qwen3_8b_removeafterco_15_24_aqfiyubn` | `5eaef8e5...` | 48 | ~197 GB |

### Local Data
Checkpoints exist locally at:
`logs/VerlRun/02/26/womonitor_backdoor_qwen3_8b_removeafterc_01_26_p8lr4f2g/checkpoints/`
Steps: global_step_40 through global_step_320 (8 checkpoints).

## Fix Plan
1. Run `git check-ignore -v ...checkpoints.dvc` to find which `.gitignore` rule is causing the conflict.
2. Either:
   a. Un-ignore `.dvc` files in `logs/` (allow them through the gitignore), OR
   b. Re-`dvc add` from a non-ignored path (e.g. `checkpoints/subtle_reasoning_repro/...`).
3. `dvc push` to get data to S3.
4. Repeat for both affected runs.
5. Clean up the empty `checkpoints/subtle_reasoning_repro/hidden_womonitor_backdoor_qwen3_8b_removeaftercodeformatter/` dir if unused.
