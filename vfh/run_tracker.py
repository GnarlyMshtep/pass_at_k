"""VFH Run Tracker — track ongoing and completed training runs.

Library functions for registering, loading, refreshing, and saving tracked runs.
All callers share a single global JSONL file at logs/VerlRun/tracked_runs.jsonl.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import dacite
import pyjson5

from vfh.run_tracker_types import RunState, TrackedRun

# ---------------------------------------------------------------------------
# Global tracker path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRACKER_PATH = _REPO_ROOT / "logs" / "VerlRun" / "tracked_runs.jsonl"
_LOCK_PATH = _REPO_ROOT / "logs" / "VerlRun" / ".tracked_runs.lock"

_GRACE_PERIOD = timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_tracked_run(run: TrackedRun) -> dict[str, Any]:
    """Convert TrackedRun to a JSON-serializable dict."""
    d = dataclasses.asdict(run)
    # Enum → string
    d["state"] = run.state.value
    # datetime → ISO string
    d["registered_at"] = run.registered_at.isoformat()
    d["state_changed_at"] = run.state_changed_at.isoformat()
    d["ended_at"] = run.ended_at.isoformat() if run.ended_at else None
    return d


_STATE_MIGRATIONS: dict[str, str] = {
    "killed": "registered_no_wandb",  # removed in favor of REGISTERED_NO_WANDB
}


def _deserialize_tracked_run(d: dict[str, Any]) -> TrackedRun:
    """Reconstruct TrackedRun from a JSON dict."""
    # Migrate removed states
    if d.get("state") in _STATE_MIGRATIONS:
        d["state"] = _STATE_MIGRATIONS[d["state"]]
    return dacite.from_dict(
        data_class=TrackedRun,
        data=d,
        config=dacite.Config(
            cast=[RunState],
            type_hooks={datetime: datetime.fromisoformat},
        ),
    )


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_tracked_runs() -> list[TrackedRun]:
    """Load all tracked runs from the global JSONL file."""
    if not _TRACKER_PATH.exists():
        return []
    runs: list[TrackedRun] = []
    for line in _TRACKER_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        runs.append(_deserialize_tracked_run(d=json.loads(line)))
    return runs


def save_tracked_runs(runs: list[TrackedRun]) -> None:
    """Overwrite the global JSONL file with the given runs.

    Holds an exclusive lock during the write to prevent register_run appends
    from being clobbered. Also merges in any runs that were appended between
    the caller's load_tracked_runs() and this save (identified by run_dir,
    which is unique per run).
    """
    _TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(_LOCK_PATH, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            # Re-read to pick up any appends since the caller's load
            on_disk = load_tracked_runs()
            known_dirs = {r.run_dir for r in runs}

            # Merge: keep any on-disk entries the caller doesn't know about
            for disk_run in on_disk:
                if disk_run.run_dir not in known_dirs:
                    runs.append(disk_run)

            tmp_path = _TRACKER_PATH.with_suffix(".jsonl.tmp")
            with open(tmp_path, "w") as f:
                for run in runs:
                    json.dump(_serialize_tracked_run(run=run), f, ensure_ascii=False)
                    f.write("\n")
            tmp_path.rename(_TRACKER_PATH)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


def register_run(tracked_run: TrackedRun) -> None:
    """Append a new tracked run to the global tracker file.

    Uses a shared lock file to coordinate with save_tracked_runs().
    If wandb info is missing, forces state to RUNNING_NO_WANDB.
    """
    # Force no-wandb state if wandb info is missing
    if tracked_run.wandb_entity is None or tracked_run.wandb_project is None:
        tracked_run.state = RunState.RUNNING_NO_WANDB

    _TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_serialize_tracked_run(run=tracked_run), ensure_ascii=False) + "\n"

    with open(_LOCK_PATH, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(_TRACKER_PATH, "a") as f:
                f.write(line)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def register_run_from_metadata(
    metadata_path: str,
    n_gpus: int | None = None,
    slurm_job_id: int | None = None,
) -> None:
    """Convenience: register a run from its run_metadata.json5 file.

    Loads metadata, extracts model name / wandb info, and calls register_run().
    Designed to be callable as a one-liner from sbatch scripts.
    If slurm_job_id is not provided, tries $SLURM_JOB_ID from the environment.
    """
    from vfh.run_manager import load_run_metadata

    meta_path = Path(metadata_path)
    run_dir = str(meta_path.parent)
    metadata = load_run_metadata(run_dir=run_dir)

    # Extract short model name from the hydra overrides
    base_model: str | None = None
    for override in metadata.resolved_hydra_overrides:
        if override.startswith("actor_rollout_ref.model.path="):
            full_path = override.split("=", 1)[1]
            base_model = Path(full_path).name
            break

    # Auto-detect SLURM job ID from environment if not provided
    if slurm_job_id is None:
        env_job_id = os.environ.get("SLURM_JOB_ID")
        if env_job_id is not None:
            slurm_job_id = int(env_job_id)

    now = datetime.now(tz=timezone.utc)
    register_run(tracked_run=TrackedRun(
        run_dir=run_dir,
        state=RunState.REGISTERED,
        registered_at=now,
        state_changed_at=now,
        run_id=metadata.run_id,
        description=metadata.description,
        wandb_url=metadata.wandb_url,
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        wandb_project=metadata.project_name,
        base_model=base_model,
        n_gpus=n_gpus,
        slurm_job_id=slurm_job_id,
    ))


# ---------------------------------------------------------------------------
# Unregister
# ---------------------------------------------------------------------------


def unregister_run(run_id: str) -> bool:
    """Remove a run from the tracker by run_id. Returns True if found."""
    runs = load_tracked_runs()
    before = len(runs)
    runs = [r for r in runs if r.run_id != run_id]
    if len(runs) == before:
        return False
    save_tracked_runs(runs=runs)
    return True


# ---------------------------------------------------------------------------
# Refresh — checkpoint/rollout discovery (delegates to catalog extractors)
# ---------------------------------------------------------------------------


def _refresh_steps(run: TrackedRun) -> None:
    """Use the appropriate catalog extractor to discover checkpoint/rollout steps."""
    from vfh.catalog import find_extractor

    run_dir = Path(run.run_dir)
    if not run_dir.is_dir():
        return

    try:
        extractor = find_extractor(run_dir=run_dir)
    except ValueError:
        return

    run.checkpoint_steps = extractor.find_checkpoint_steps(run_dir=run_dir)
    run.rollout_steps = extractor.find_rollout_steps(run_dir=run_dir)


# ---------------------------------------------------------------------------
# Refresh — wandb state polling
# ---------------------------------------------------------------------------


def refresh_run_states(runs: list[TrackedRun]) -> list[TrackedRun]:
    """Poll wandb for state transitions and refresh checkpoint/rollout steps.

    Only queries wandb for runs in REGISTERED or RUNNING state.
    Returns updated list (caller is responsible for saving).
    """
    # Refresh checkpoint/rollout steps for all non-reviewed runs
    for run in runs:
        if run.state == RunState.REVIEWED:
            continue
        _refresh_steps(run=run)

    # Collect runs that need wandb polling
    _POLLABLE_STATES = (RunState.REGISTERED, RunState.REGISTERED_NO_WANDB, RunState.RUNNING)
    needs_poll = [
        r for r in runs
        if r.state in _POLLABLE_STATES
        and r.wandb_entity is not None
        and r.wandb_project is not None
        and r.run_id is not None
    ]

    # Also backfill ended_at for already-ended runs missing it
    needs_backfill = [
        r for r in runs
        if r.state in (RunState.FINISHED, RunState.CRASHED)
        and r.ended_at is None
        and r.wandb_entity is not None
        and r.wandb_project is not None
        and r.run_id is not None
    ]

    if not needs_poll and not needs_backfill:
        return runs

    # Lazy import — wandb is heavy
    import wandb
    api = wandb.Api()

    now = datetime.now(tz=timezone.utc)
    for run in needs_poll:
        assert run.wandb_entity and run.wandb_project and run.run_id  # for type checker
        try:
            wb_run = api.run(f"{run.wandb_entity}/{run.wandb_project}/{run.run_id}")
            wb_state: str = wb_run.state  # "running", "finished", "crashed", "failed"
        except wandb.errors.CommError:
            # Wandb doesn't know about this run yet
            if run.state == RunState.REGISTERED:
                elapsed = now - run.registered_at
                if elapsed > _GRACE_PERIOD:
                    run.state = RunState.REGISTERED_NO_WANDB
                    run.state_changed_at = now
                # else: still in grace period, keep REGISTERED
            # REGISTERED_NO_WANDB stays as-is — keeps polling next refresh
            continue

        old_state = run.state
        if wb_state == "running":
            if run.state in (RunState.REGISTERED, RunState.REGISTERED_NO_WANDB):
                run.state = RunState.RUNNING
                run.state_changed_at = now
        elif wb_state in ("finished", "crashed", "failed"):
            run.state = RunState.FINISHED if wb_state == "finished" else RunState.CRASHED
            run.state_changed_at = now

            run.ended_at = _extract_end_time(wb_run=wb_run)

    # Backfill ended_at for already-ended runs
    for run in needs_backfill:
        assert run.wandb_entity and run.wandb_project and run.run_id
        try:
            wb_run = api.run(f"{run.wandb_entity}/{run.wandb_project}/{run.run_id}")
            run.ended_at = _extract_end_time(wb_run=wb_run)
        except Exception:
            pass  # best-effort

    return runs


def _extract_end_time(wb_run: Any) -> datetime | None:
    """Extract actual end time from a wandb run object (best-effort)."""
    try:
        last_ts = wb_run.summary.get("_timestamp")
        if last_ts is not None:
            return datetime.fromtimestamp(last_ts, tz=timezone.utc)
        if wb_run.heartbeatAt:
            return datetime.fromisoformat(
                wb_run.heartbeatAt.replace("Z", "+00:00")
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# CLI entry point (for one-liner calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Usage: python -m vfh.run_tracker register <metadata_path> [n_gpus]
    if len(sys.argv) >= 3 and sys.argv[1] == "register":
        n_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else None
        register_run_from_metadata(metadata_path=sys.argv[2], n_gpus=n_gpus)
        print(f"Registered run from {sys.argv[2]}")
    else:
        print("Usage: python -m vfh.run_tracker register <metadata_path> [n_gpus]")
        sys.exit(1)
