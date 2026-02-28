"""VFH (Verl For Humans) — shared dataclasses and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ForkReason(Enum):
    ROOT = "root"
    INTENTIONAL_FORK = "fork"
    CONTINUE = "continue"  # stopped or crashed, resuming


class ValidationMode(Enum):
    FULL = "full"         # run validation, prompt for confirmation
    AUTO_APPROVE = "auto" # run validation, auto-approve (-y)
    SKIP = "skip"         # skip validation entirely (-yy)


class BaseModel(Enum):
    QWEN3_4B_I = "Qwen3-4B-I"
    # add more as needed


# ---------------------------------------------------------------------------
# Orchestrator config (loaded via load_orchestrator_config, not hardcoded)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointDaemonConfig:
    enabled: bool = True
    poll_interval_seconds: int = 300
    clean_after_backup: bool = True  # remove contents after confirmed DVC backup; keep empty dir


@dataclass
class OrchestratorConfig:
    validation_mode: ValidationMode = ValidationMode.FULL
    checkpoint_daemons: list[CheckpointDaemonConfig] = field(
        default_factory=lambda: [CheckpointDaemonConfig()]
    )
    logs_root: str = "logs"
    # wandb_entity: from $WANDB_ENTITY env var at launch time
    # wandb_project: from run config (trainer.project_name)
    # n_gpus: from run config (trainer.n_gpus_per_node)


# ---------------------------------------------------------------------------
# Run origin / lineage
# ---------------------------------------------------------------------------


@dataclass
class RunOrigin:
    fork_reason: ForkReason
    parent_run_id: Optional[str] = None
    parent_run_dir: Optional[str] = None
    parent_checkpoint_step: Optional[int] = None  # global_step we resume/fork from


# ---------------------------------------------------------------------------
# Input configs (what the user provides to the orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class NewRunConfig:
    base_config_path: str                             # path to base JSON5 config
    overrides_path: Optional[str] = None              # path to override JSON5
    extra_hydra_overrides: list[str] = field(default_factory=list)
    description: str = ""
    origin: RunOrigin = field(default_factory=lambda: RunOrigin(fork_reason=ForkReason.ROOT))


@dataclass
class ContinueRunConfig:
    run_dir: str
    overrides_path: Optional[str] = None
    extra_hydra_overrides: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prepared run (intermediate state between prepare and exec/sbatch)
# ---------------------------------------------------------------------------


@dataclass
class PreparedRun:
    """Result of prepare(): everything needed to either exec into verl or generate sbatch."""
    merged_config: dict[str, Any]
    hydra_overrides: list[str]
    run_metadata: RunMetadata


# ---------------------------------------------------------------------------
# Run metadata (persisted as run_metadata.json5 in each run directory)
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    run_id: str
    run_dir: str
    wandb_run_id: str
    wandb_url: str
    project_name: str
    description: str
    origin: RunOrigin
    created_at: str                          # ISO 8601 timestamp
    base_config_path: str
    overrides_path: Optional[str]
    resolved_hydra_overrides: list[str]      # full list passed to verl
    child_run_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Notable runs (stored in notable_runs.jsonl at project root)
# ---------------------------------------------------------------------------


@dataclass
class RunAttributes:
    base_model: BaseModel


@dataclass
class NotableRunEntry:
    """One entry per line in notable_runs.jsonl."""
    paths: list[str]             # ordered DAG of run dirs (root → leaf)
    attributes: RunAttributes
    description: Optional[str] = None
