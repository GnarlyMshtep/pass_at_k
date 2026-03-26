"""Types for the VFH run tracker — tracks ongoing and completed training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RunState(Enum):
    REGISTERED = "registered"            # Just launched, wandb not yet reporting
    REGISTERED_NO_WANDB = "registered_no_wandb"  # Grace period expired, still polling wandb each refresh
    RUNNING = "running"                  # Wandb confirms active
    RUNNING_NO_WANDB = "running_no_wandb"  # No wandb info at registration — user must manually manage
    FINISHED = "finished"                # Wandb reports finished
    CRASHED = "crashed"                  # Wandb reports crashed/failed
    REVIEWED = "reviewed"                # User explicitly marked as handled


@dataclass
class TrackedRun:
    """A single tracked training run."""

    # Required — always provided
    run_dir: str                                        # absolute path to run directory
    state: RunState
    registered_at: datetime                             # when added to tracker
    state_changed_at: datetime                          # last state transition

    # Optional — may not apply in all contexts
    run_id: str | None = None                           # 8-char wandb/custom ID
    description: str | None = None
    wandb_url: str | None = None
    wandb_entity: str | None = None                     # for wandb API queries
    wandb_project: str | None = None                    # for wandb API queries
    base_model: str | None = None                       # short name, e.g. "Qwen3-4B-I"
    n_gpus: int | None = None
    slurm_job_id: int | None = None                     # SLURM_JOB_ID if launched via sbatch
    source_framework: str = "vfh"

    # Timing
    ended_at: datetime | None = None                    # actual end time (from wandb), not detection time

    # User notes
    comments: str | None = None                         # free-form user notes

    # Populated on refresh/poll
    checkpoint_steps: list[int] | None = None           # [40, 80, 120, ...]
    rollout_steps: list[int] | None = None
