"""Types for the run catalog system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TagCategory:
    """A category that groups related tags (e.g. Project, Outcome, RunType)."""
    name: str
    description: str


@dataclass
class CatalogTag:
    """A reusable tag for categorizing catalog entries."""
    name: str
    description: str
    category: str = "Uncategorized"  # references TagCategory.name


@dataclass
class CatalogEntry:
    """One cataloged run/checkpoint."""
    run_id: str
    run_dir: str
    description: str                   # user-provided
    base_model: str                    # from resolved_hydra_overrides: actor_rollout_ref.model.path
    reward_config: dict[str, Any]      # keys under custom_reward_function.reward_kwargs.reward_config.*
    train_dataset: str                 # from data.train_files
    checkpoint_range: tuple[int, int] | None  # (min_step, max_step) of present global_step_N dirs
    rollout_range: tuple[int, int] | None  # (min_step, max_step) of {N}.jsonl files in rollouts/train/
    tags: list[str]                    # tag names
    wandb_url: str
    cataloged_at: str                  # ISO 8601
    started_at: str | None = None      # ISO 8601 — when the run was launched (from metadata)
    source_framework: str = "vfh"      # extensible: "tinker", etc.
    follows: list[str] = field(default_factory=list)       # run_ids this run continues/forks from
    preceded_by: list[str] = field(default_factory=list)   # run_ids that continue/fork from this run


@dataclass
class Catalog:
    """The full catalog: entries + tags + tag categories."""
    entries: list[CatalogEntry] = field(default_factory=list)
    tags: list[CatalogTag] = field(default_factory=list)
    tag_categories: list[TagCategory] = field(default_factory=list)
