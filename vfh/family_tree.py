"""Explore the family tree of a VFH/TFH run — parents, children, and descendant counts.

Usage:
    python -m vfh.family_tree --path <run_dir_or_wandb_id>
    python -m vfh.family_tree --path k16vo4tp
    python -m vfh.family_tree --path logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyjson5
import tyro

from vfh.catalog import (
    find_extractor,
    load_catalog,
    resolve_run_dir,
    save_catalog,
)
from vfh.catalog_types import Catalog, CatalogEntry
from vfh.interactive_utils import (
    C,
    UserCancelled,
    colored,
    copy_and_print_path,
    copy_and_print_url,
    input_or_esc,
    trunc,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """Lightweight info about a run, read from metadata on disk."""
    run_id: str
    run_dir: Path
    parent_run_id: str | None
    child_run_ids: list[str]
    fork_reason: str | None          # root, intentional_fork, continue
    description: str                 # from run dir name
    checkpoint_steps: list[int] | None
    rollout_steps: list[int] | None
    base_model: str | None
    is_cataloged: bool


# ---------------------------------------------------------------------------
# Metadata reading
# ---------------------------------------------------------------------------


def _read_run_info(run_dir: Path, cataloged_ids: set[str]) -> RunInfo:
    """Read lightweight run info from run_metadata.json5 + disk scan."""
    meta_file = run_dir / "run_metadata.json5"
    if not meta_file.exists():
        raise ValueError(f"No run_metadata.json5 in {run_dir}")

    with open(meta_file) as f:
        meta = pyjson5.load(f)

    run_id: str = meta["run_id"]
    origin = meta.get("origin", {})
    child_run_ids: list[str] = meta.get("child_run_ids", [])

    # Base model — try extractor-style parsing
    base_model: str | None = None
    overrides: list[str] = meta.get("resolved_hydra_overrides", [])
    for ov in overrides:
        clean = ov.lstrip("+")
        if clean.startswith("actor_rollout_ref.model.path="):
            base_model = clean.split("=", 1)[1]
            break
    # TFH fallback
    if base_model is None:
        resolved_config = meta.get("resolved_config", {})
        base_model = resolved_config.get("model_name")

    # Step discovery — reuse extractor logic
    try:
        extractor = find_extractor(run_dir=run_dir)
        checkpoint_steps = extractor.find_checkpoint_steps(run_dir=run_dir)
        rollout_steps = extractor.find_rollout_steps(run_dir=run_dir)
    except ValueError:
        checkpoint_steps = None
        rollout_steps = None

    return RunInfo(
        run_id=run_id,
        run_dir=run_dir,
        parent_run_id=origin.get("parent_run_id"),
        child_run_ids=child_run_ids,
        fork_reason=origin.get("fork_reason"),
        description=run_dir.name,
        checkpoint_steps=checkpoint_steps,
        rollout_steps=rollout_steps,
        base_model=base_model,
        is_cataloged=run_id in cataloged_ids,
    )


def _try_read_run_info(run_id: str, cataloged_ids: set[str]) -> RunInfo | None:
    """Try to resolve and read a run. Returns None if not found."""
    try:
        run_dir = resolve_run_dir(path_or_id=run_id)
        return _read_run_info(run_dir=run_dir, cataloged_ids=cataloged_ids)
    except (ValueError, FileNotFoundError, KeyError) as e:
        return None


def _count_descendants(run_id: str, _cache: dict[str, int] | None = None) -> int:
    """Count total descendants recursively (BFS). Returns 0 if no children."""
    if _cache is None:
        _cache = {}
    if run_id in _cache:
        return _cache[run_id]

    try:
        run_dir = resolve_run_dir(path_or_id=run_id)
        meta_file = run_dir / "run_metadata.json5"
        if not meta_file.exists():
            _cache[run_id] = 0
            return 0
        with open(meta_file) as f:
            meta = pyjson5.load(f)
        children: list[str] = meta.get("child_run_ids", [])
    except (ValueError, FileNotFoundError):
        _cache[run_id] = 0
        return 0

    total = len(children)
    for child_id in children:
        total += _count_descendants(run_id=child_id, _cache=_cache)
    _cache[run_id] = total
    return total


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _format_steps(steps: list[int] | None) -> str:
    """Format steps as a range string."""
    if not steps:
        return colored("[no steps]", C.RED, C.DIM)
    if len(steps) == 1:
        return colored(f"step {steps[0]}", C.YELLOW)
    return colored(f"steps {steps[0]}\u2192{steps[-1]} ({len(steps)})", C.YELLOW)


def _format_run_info(idx: int, info: RunInfo, descendant_count: int | None = None) -> str:
    """Format a run info line for display."""
    parts: list[str] = []

    parts.append(f"  [{idx}]")
    parts.append(colored(info.run_id, C.BOLD))

    # Cataloged marker
    if info.is_cataloged:
        parts.append(colored("[cataloged]", C.GREEN))

    # Model (short name)
    if info.base_model:
        model_short = info.base_model.rsplit("/", 1)[-1] if "/" in info.base_model else info.base_model
        parts.append(colored(model_short, C.CYAN))

    # Fork reason
    if info.fork_reason and info.fork_reason != "root":
        parts.append(colored(f"({info.fork_reason})", C.DIM))

    # Steps (prefer rollouts)
    parts.append(_format_steps(steps=info.rollout_steps or info.checkpoint_steps))

    # Descendant count
    if descendant_count is not None and descendant_count > 0:
        parts.append(colored(f"+{descendant_count} descendants", C.MAGENTA))

    return "  ".join(parts)


def _show_tree(root: RunInfo, parents: list[RunInfo], children: list[RunInfo],
               descendant_counts: dict[str, int]) -> list[RunInfo]:
    """Display the family tree. Returns the flat selectable list (parents + sorted children)."""
    selectable: list[RunInfo] = []
    idx = 0

    # Parents
    if parents:
        print(f"\n  {colored('PARENTS', C.BOLD, C.BLUE)}")
        for p in parents:
            desc_count = descendant_counts.get(p.run_id)
            print(_format_run_info(idx=idx, info=p, descendant_count=desc_count))
            selectable.append(p)
            idx += 1
        print(colored("  \u2502", C.DIM))
        print(colored("  \u25bc", C.DIM))

    # Root
    print(f"\n  {colored('ROOT', C.BOLD, C.YELLOW)}")
    total_desc = descendant_counts.get(root.run_id, 0)
    root_line = _format_run_info(idx=-1, info=root, descendant_count=total_desc).replace("[-1]", " \u2605 ")
    print(root_line)

    # Children — sort by first step
    if children:
        print(colored("\n  \u25bc", C.DIM))
        print(f"  {colored(f'CHILDREN ({len(children)})', C.BOLD, C.GREEN)}")
        children_sorted = sorted(
            children,
            key=lambda c: (c.rollout_steps or c.checkpoint_steps or [0])[0],
        )
        for child in children_sorted:
            desc_count = descendant_counts.get(child.run_id, 0)
            print(_format_run_info(idx=idx, info=child, descendant_count=desc_count))
            selectable.append(child)
            idx += 1
    else:
        print(colored("\n  (no children)", C.DIM))

    return selectable


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _catalog_run(info: RunInfo, catalog: Catalog, catalog_path: Path) -> None:
    """Launch catalog interactively for a run."""
    print(colored(f"  Launching catalog for {info.run_id}...", C.CYAN))
    cmd = [sys.executable, "-m", "vfh.catalog", "--path", str(info.run_dir)]
    subprocess.run(cmd)
    # Reload catalog after cataloging
    return


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------


@dataclass
class FamilyTreeConfig:
    """Explore the family tree of a training run."""
    path: str  # run dir path OR wandb ID (8-char string)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_tree(path_or_id: str) -> None:
    """Core tree logic, callable both from CLI and recursive [t]ree action."""
    run_dir = resolve_run_dir(path_or_id=path_or_id)

    # Load catalog to check cataloged status
    from vfh.catalog import _get_catalog_path
    catalog_path = _get_catalog_path()
    catalog = load_catalog(catalog_path=catalog_path)
    cataloged_ids = {e.run_id for e in catalog.entries}

    # Read root run info
    print(colored(f"Reading {run_dir.name}...", C.DIM))
    root = _read_run_info(run_dir=run_dir, cataloged_ids=cataloged_ids)

    # Read parents (walk up the chain)
    parents: list[RunInfo] = []
    current_parent_id = root.parent_run_id
    seen: set[str] = {root.run_id}
    while current_parent_id and current_parent_id not in seen:
        seen.add(current_parent_id)
        parent_info = _try_read_run_info(run_id=current_parent_id, cataloged_ids=cataloged_ids)
        if parent_info is None:
            print(colored(f"  Warning: parent {current_parent_id} not found on disk", C.YELLOW))
            break
        parents.append(parent_info)
        current_parent_id = parent_info.parent_run_id
    parents.reverse()  # oldest ancestor first

    # Read immediate children
    children: list[RunInfo] = []
    for child_id in root.child_run_ids:
        child_info = _try_read_run_info(run_id=child_id, cataloged_ids=cataloged_ids)
        if child_info is None:
            print(colored(f"  Warning: child {child_id} not found on disk", C.YELLOW))
            continue
        children.append(child_info)

    # Count descendants for root and each child
    print(colored("Counting descendants...", C.DIM))
    desc_cache: dict[str, int] = {}
    descendant_counts: dict[str, int] = {}
    descendant_counts[root.run_id] = _count_descendants(run_id=root.run_id, _cache=desc_cache)
    for child in children:
        descendant_counts[child.run_id] = _count_descendants(run_id=child.run_id, _cache=desc_cache)
    for parent in parents:
        descendant_counts[parent.run_id] = _count_descendants(run_id=parent.run_id, _cache=desc_cache)

    # Display
    selectable = _show_tree(root=root, parents=parents, children=children, descendant_counts=descendant_counts)

    if not selectable:
        print(colored("\n  No relatives to interact with.", C.DIM))
        return

    # Interactive loop
    while True:
        print(f"\n  {colored('[#] select  [q]uit', C.DIM)}")
        try:
            raw = input_or_esc(prompt="\n> ").strip().lower()
        except UserCancelled:
            break

        if raw in ("q", "quit", ""):
            break

        try:
            idx = int(raw)
        except ValueError:
            print(colored(f"  Unknown command: {raw}", C.RED))
            continue

        if idx < 0 or idx >= len(selectable):
            print(colored(f"  Index out of range (0-{len(selectable) - 1})", C.RED))
            continue

        selected = selectable[idx]

        # Show detail + actions
        print(f"\n  {colored('Run:', C.BOLD)} {colored(selected.run_id, C.BOLD, C.CYAN)}")
        print(f"  {'Dir:':<14} {trunc(text=str(selected.run_dir), max_len=80)}")
        if selected.base_model:
            model_short = selected.base_model.rsplit("/", 1)[-1] if "/" in selected.base_model else selected.base_model
            print(f"  {'Model:':<14} {colored(model_short, C.CYAN)}")
        if selected.fork_reason:
            print(f"  {'Origin:':<14} {selected.fork_reason}")
        print(f"  {'Checkpoints:':<14} {_format_steps(steps=selected.checkpoint_steps)}")
        print(f"  {'Rollouts:':<14} {_format_steps(steps=selected.rollout_steps)}")
        print(f"  {'Children:':<14} {len(selected.child_run_ids)}")
        desc = descendant_counts.get(selected.run_id, 0)
        if desc > 0:
            print(f"  {'Descendants:':<14} {desc}")
        print(f"  {'Cataloged:':<14} {colored('yes', C.GREEN) if selected.is_cataloged else colored('no', C.DIM)}")

        actions = "[w]andb  [p]ath  [c]atalog  [t]ree (explore this run)"
        if selected.is_cataloged:
            actions = "[w]andb  [p]ath  [t]ree (explore this run)"
        print(f"\n  {colored(actions, C.DIM)}")

        try:
            choice = input_or_esc(prompt="  Action: ").strip().lower()
        except UserCancelled:
            continue

        if choice == "p":
            copy_and_print_path(path=str(selected.run_dir))

        elif choice == "w":
            # Try to construct wandb URL from metadata
            try:
                meta_file = selected.run_dir / "run_metadata.json5"
                with open(meta_file) as f:
                    meta = pyjson5.load(f)
                wandb_url = meta.get("wandb_url", "")
                if not wandb_url:
                    overrides = meta.get("resolved_hydra_overrides", [])
                    from vfh.catalog import _parse_hydra_overrides, _DEFAULT_WANDB_ENTITY
                    import os
                    ov_dict = _parse_hydra_overrides(overrides=overrides)
                    project = ov_dict.get("trainer.project_name", "")
                    entity = os.environ.get("WANDB_ENTITY", _DEFAULT_WANDB_ENTITY)
                    if project:
                        wandb_url = f"https://wandb.ai/{entity}/{project}/runs/{selected.run_id}"
                copy_and_print_url(url=wandb_url, label="W&B URL")
            except Exception:
                print(colored("  Could not determine W&B URL.", C.RED))

        elif choice == "c" and not selected.is_cataloged:
            _catalog_run(info=selected, catalog=catalog, catalog_path=catalog_path)
            # Reload catalog
            catalog = load_catalog(catalog_path=catalog_path)
            cataloged_ids = {e.run_id for e in catalog.entries}
            selected.is_cataloged = selected.run_id in cataloged_ids
            # Redisplay tree
            selectable = _show_tree(root=root, parents=parents, children=children, descendant_counts=descendant_counts)

        elif choice == "t":
            # Recurse into this run's family tree
            print(colored(f"\n  === Exploring {selected.run_id} ===", C.BOLD))
            _run_tree(path_or_id=str(selected.run_dir))
            # After returning, redisplay current tree
            selectable = _show_tree(root=root, parents=parents, children=children, descendant_counts=descendant_counts)

        else:
            print(colored(f"  Unknown action: {choice}", C.RED))


def main() -> None:
    config = tyro.cli(FamilyTreeConfig)
    _run_tree(path_or_id=config.path)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, UserCancelled):
        print()
