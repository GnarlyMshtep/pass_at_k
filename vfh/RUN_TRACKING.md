# Run Tracking — Quick Reference

> **Do NOT explore `vfh/` further.** This file contains everything you need to register and view tracked runs.

## Registering a Run

When you launch a training run (any framework), register it so it appears in the run tracker:

```python
from datetime import datetime, timezone
from vfh.run_tracker import register_run
from vfh.run_tracker_types import TrackedRun, RunState

now = datetime.now(tz=timezone.utc)
register_run(tracked_run=TrackedRun(
    run_dir="/absolute/path/to/run/directory",
    state=RunState.REGISTERED,          # or RUNNING_NO_WANDB if no wandb
    registered_at=now,
    state_changed_at=now,
    # Optional but recommended:
    run_id="abcd1234",                  # 8-char wandb ID if available
    description="my_experiment_name",
    wandb_url="https://wandb.ai/...",   # None if no wandb
    wandb_entity="...",                 # None if no wandb
    wandb_project="...",                # None if no wandb
    base_model="Qwen3-4B-I",           # short model name
    n_gpus=4,
))
```

### From a VFH run_metadata.json5

```python
from vfh.run_tracker import register_run_from_metadata
register_run_from_metadata(metadata_path="path/to/run_metadata.json5", n_gpus=4)
```

### CLI one-liner (e.g. from shell scripts)

```bash
python -m vfh.run_tracker register path/to/run_metadata.json5 4
```

## States

| State | Meaning |
|-------|---------|
| `REGISTERED` | Just launched, wandb not yet reporting (10-min grace period) |
| `RUNNING` | Wandb confirms active |
| `RUNNING_NO_WANDB` | No wandb info — stays here until manually removed/reviewed |
| `FINISHED` | Wandb reports finished |
| `CRASHED` | Wandb reports crashed/failed |
| `KILLED` | Grace period expired, wandb never saw it |
| `REVIEWED` | User marked as handled |

## Viewing Runs

```bash
python -m vfh.run_tracker_viewer              # refresh wandb + show all
python -m vfh.run_tracker_viewer --no-refresh  # skip wandb polling
python -m vfh.run_tracker_viewer --filter-empty # hide runs with 0 checkpoints
python -m vfh.run_tracker_viewer --show-reviewed
```

Inside the viewer (both tracker and catalog modes):

- `[c]` — switch tracker ↔ catalog mode (catalog auto-reloads on switch in)
- `[t]` — toggle lineage tree view (tracker: reads `origin.parent_run_id` from each `run_metadata.json5`; catalog: uses cached `follows`/`preceded_by`)
- `[h]` — filter to runs/entries from the last N hours
- `[f]` — filters (tracker: toggle `filter empty`; catalog: tag/model/dataset sub-menu)
- `[x]` — tracker only: bulk-mark all already-cataloged runs as reviewed
- `[a]` — tracker only: toggle showing `REVIEWED` runs
- `[v]` — build a wandb workspace URL for a selection of runs (0.99 EMA smoothing applied by default)
- `n-m` or `n-m, k-t` — bulk select by range (mark reviewed, set tags/description)
- `[n]` / `[d]` — tracker notes / catalog full descriptions

Active toggles are marked with `*` in the prompt bar. Descriptions default to full (not truncated) in both modes.

## Tracker File

All runs are stored in `logs/VerlRun/tracked_runs.jsonl` (one JSON object per line). This is the single global tracker file — all callers write to the same file.
