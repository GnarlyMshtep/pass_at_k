"""Interactive viewer for tracked VFH training runs.

Usage:
    python -m vfh.run_tracker_viewer              # refresh + show all
    python -m vfh.run_tracker_viewer --no-refresh  # skip wandb polling
    python -m vfh.run_tracker_viewer --show-reviewed
    python -m vfh.run_tracker_viewer --filter-empty # hide runs with 0 checkpoints
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tyro

from vfh.catalog import load_catalog, refresh_catalog_lineage, save_catalog
from vfh.catalog_types import Catalog, CatalogEntry, CatalogTag
from vfh.interactive_utils import (
    C,
    UserCancelled,
    colored,
    copy_and_print_path,
    copy_and_print_url,
    copy_to_clipboard,
    input_or_esc,
    trunc,
)
from vfh.run_tracker import (
    load_tracked_runs,
    refresh_run_states,
    save_tracked_runs,
    unregister_run,
)
from vfh.run_tracker_types import RunState, TrackedRun


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------


@dataclass
class ViewerConfig:
    """Interactive viewer for tracked training runs."""
    refresh: bool = True            # poll wandb for state updates on startup
    show_reviewed: bool = False     # include REVIEWED runs in the list
    filter_empty: bool = False      # hide runs with 0 checkpoint steps


@dataclass
class CatalogFilter:
    """Active filters for catalog view. AND across filter types, OR within each multi-select."""
    tags: list[str] = field(default_factory=list)       # entry must have ALL of these tags
    exclude_tags: list[str] = field(default_factory=list)  # entry must have NONE of these tags
    models: list[str] = field(default_factory=list)     # entry model must be one of these (OR)
    datasets: list[str] = field(default_factory=list)   # entry dataset must be one of these (OR)


def _get_catalog_path() -> Path:
    """Catalog file path — same logic as catalog.py."""
    import os
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("RUN_CATALOG_PATH", str(_REPO_ROOT / "logs" / "catalog_data" / "catalog.json")))


# ---------------------------------------------------------------------------
# Color scheme per state
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[RunState, str] = {
    RunState.REGISTERED: C.CYAN,
    RunState.REGISTERED_NO_WANDB: C.YELLOW,
    RunState.RUNNING: C.GREEN,
    RunState.RUNNING_NO_WANDB: C.YELLOW,
    RunState.FINISHED: C.RESET,
    RunState.CRASHED: C.RED,
    RunState.REVIEWED: C.DIM,
}

_STATE_DOTS: dict[RunState, str] = {
    RunState.REGISTERED: colored("●", C.CYAN),
    RunState.REGISTERED_NO_WANDB: colored("?", C.YELLOW),
    RunState.RUNNING: colored("●", C.GREEN),
    RunState.RUNNING_NO_WANDB: colored("●", C.YELLOW),
    RunState.FINISHED: colored("○", C.RESET),
    RunState.CRASHED: colored("●", C.RED),
    RunState.REVIEWED: colored("✓", C.DIM),
}

# Display order for state groups
_STATE_ORDER: list[RunState] = [
    RunState.RUNNING,
    RunState.RUNNING_NO_WANDB,
    RunState.REGISTERED,
    RunState.REGISTERED_NO_WANDB,
    RunState.FINISHED,
    RunState.CRASHED,
    RunState.REVIEWED,
]


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def _relative_time(dt: datetime) -> str:
    """Human-readable relative time like '2h ago', '3d ago'."""
    now = datetime.now(tz=timezone.utc)
    # Handle naive datetimes
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _format_steps(steps: list[int] | None) -> str:
    """Format steps as a range string."""
    if not steps:
        return colored("[no steps]", C.RED, C.DIM)
    if len(steps) == 1:
        return colored(f"{steps[0]}", C.YELLOW)
    return colored(f"{steps[0]}\u2192{steps[-1]}", C.YELLOW)


def _format_run_line(idx: int, run: TrackedRun) -> str:
    """Format a single run as a display line."""
    parts: list[str] = []

    # Index
    parts.append(f" [{idx}]")

    # Run ID
    run_id_str = run.run_id or "?"
    parts.append(colored(run_id_str, C.BOLD))

    # Model
    if run.base_model:
        parts.append(colored(run.base_model, C.CYAN))

    # Description
    desc = run.description or Path(run.run_dir).name
    parts.append(trunc(text=desc, max_len=45))

    # Steps (prefer rollouts as more fine-grained, fall back to checkpoints)
    parts.append(_format_steps(steps=run.rollout_steps or run.checkpoint_steps))

    # GPUs
    if run.n_gpus is not None:
        parts.append(f"{run.n_gpus}gpu")

    # Framework
    _FRAMEWORK_COLORS: dict[str, str] = {"vfh": C.YELLOW, "tfh": C.MAGENTA}
    fw_color = _FRAMEWORK_COLORS.get(run.source_framework, C.DIM)
    parts.append(colored(f"[{run.source_framework}]", fw_color))

    # Time info
    started = colored(f"started {_relative_time(dt=run.registered_at)}", C.DIM)
    if run.state in (RunState.RUNNING, RunState.RUNNING_NO_WANDB, RunState.REGISTERED, RunState.REGISTERED_NO_WANDB):
        parts.append(started)
    else:
        # Show both started and ended time for finished/crashed
        end_dt = run.ended_at or run.state_changed_at
        ended = _relative_time(dt=end_dt)
        parts.append(colored(f"({started}, ended {ended})", C.DIM))

    return "  ".join(parts)


def _display_runs(runs: list[TrackedRun], config: ViewerConfig) -> list[TrackedRun]:
    """Display runs grouped by state. Returns the flat display-order list."""
    display_runs: list[TrackedRun] = []

    for state in _STATE_ORDER:
        if state == RunState.REVIEWED and not config.show_reviewed:
            continue

        group = [r for r in runs if r.state == state]
        if config.filter_empty:
            group = [r for r in group if r.checkpoint_steps]
        group.sort(key=lambda r: r.registered_at, reverse=True)

        if not group:
            continue

        # Header
        dot = _STATE_DOTS.get(state, "")
        state_color = _STATE_COLORS.get(state, C.RESET)
        header = colored(f"{state.value.upper()} ({len(group)})", state_color, C.BOLD)
        print(f"\n {dot} {header}")

        # Runs
        for run in group:
            idx = len(display_runs)
            print(_format_run_line(idx=idx, run=run))
            display_runs.append(run)

    if not display_runs:
        print(colored("\n  No tracked runs.", C.DIM))

    return display_runs


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _show_run_detail(run: TrackedRun) -> None:
    """Show detailed info about a single run."""
    print(f"\n  {colored('Run Details', C.BOLD, C.CYAN)}")
    print(f"  {'ID:':<12} {colored(run.run_id or '?', C.BOLD)}")
    print(f"  {'Dir:':<12} {trunc(text=run.run_dir, max_len=80)}")
    print(f"  {'State:':<12} {run.state.value}")
    print(f"  {'Model:':<12} {run.base_model or '?'}")
    print(f"  {'Framework:':<12} {run.source_framework}")
    if run.n_gpus is not None:
        print(f"  {'GPUs:':<12} {run.n_gpus}")
    if run.slurm_job_id is not None:
        print(f"  {'SLURM Job:':<12} {run.slurm_job_id}")
    if run.checkpoint_steps:
        print(f"  {'Checkpoints:':<12} {run.checkpoint_steps}")
    if run.rollout_steps:
        print(f"  {'Rollouts:':<12} steps {run.rollout_steps[0]}\u2192{run.rollout_steps[-1]} ({len(run.rollout_steps)} files)")
    print(f"  {'Registered:':<12} {run.registered_at.isoformat()}")
    if run.ended_at:
        print(f"  {'Ended:':<12} {run.ended_at.isoformat()}")
    elif run.state_changed_at != run.registered_at:
        print(f"  {'Changed:':<12} {run.state_changed_at.isoformat()}")
    if run.wandb_url:
        print(f"  {'W&B:':<12} {colored(run.wandb_url, C.BLUE)}")
    if run.comments:
        print(f"  {'Comments:':<12} {trunc(text=run.comments, max_len=400)}")


def _action_menu(run: TrackedRun, all_runs: list[TrackedRun]) -> bool:
    """Show action menu for a selected run. Returns True if list needs redisplay."""
    _show_run_detail(run=run)

    actions = "[w]andb  [p]ath  [l]aunch cmd  [n]ote  [r]emove  [m]ark reviewed"
    if run.state in (RunState.FINISHED, RunState.CRASHED, RunState.REGISTERED_NO_WANDB):
        actions += "  [c]atalog"
    print(f"\n  {colored(actions, C.DIM)}")

    try:
        choice = input_or_esc(prompt="  Action: ").strip().lower()
    except UserCancelled:
        return False

    if choice == "w":
        copy_and_print_url(url=run.wandb_url, label="W&B URL")
        return False

    elif choice == "p":
        copy_and_print_path(path=run.run_dir)
        return False

    elif choice == "l":
        launch_file = Path(run.run_dir) / "launching_command.txt"
        if launch_file.exists():
            cmd = launch_file.read_text().strip()
            from vfh.interactive_utils import copy_to_clipboard
            copy_to_clipboard(text=cmd)
            print(f"  {colored(cmd, C.CYAN)}")
        else:
            print(colored("  No launching_command.txt found.", C.RED))
        return False

    elif choice == "n":
        if run.comments:
            print(f"  Current: {trunc(text=run.comments, max_len=400)}")
        try:
            new_comment = input_or_esc(prompt="  Comment (empty to clear): ").strip()
        except UserCancelled:
            return False
        run.comments = new_comment if new_comment else None
        save_tracked_runs(runs=all_runs)
        print(colored("  Comment saved.", C.GREEN))
        # Offer to mark as reviewed
        try:
            mark = input(colored("  Mark as reviewed? [y/N]: ", C.CYAN)).strip().lower()
        except (KeyboardInterrupt, EOFError):
            mark = ""
            print()
        if mark == "y":
            run.state = RunState.REVIEWED
            run.state_changed_at = datetime.now(tz=timezone.utc)
            save_tracked_runs(runs=all_runs)
            print(colored("  Marked as reviewed.", C.GREEN))
        return True

    elif choice == "r":
        try:
            confirm = input_or_esc(prompt=f"  Remove {run.run_id or run.run_dir} from tracker? [y/N]: ")
        except UserCancelled:
            return False
        if confirm.strip().lower() == "y":
            all_runs.remove(run)
            save_tracked_runs(runs=all_runs)
            print(colored("  Removed.", C.GREEN))
            return True
        return False

    elif choice == "m":
        run.state = RunState.REVIEWED
        run.state_changed_at = datetime.now(tz=timezone.utc)
        save_tracked_runs(runs=all_runs)
        print(colored("  Marked as reviewed.", C.GREEN))
        return True

    elif choice == "c":
        # Launch catalog interactively, passing note as prefill if present
        import subprocess
        print(colored("  Launching catalog...", C.CYAN))
        cmd = [sys.executable, "-m", "vfh.catalog", "--path", run.run_dir]
        if run.comments:
            cmd += ["--prefill-description", run.comments]
        subprocess.run(cmd)
        # Offer to mark as reviewed after cataloging
        try:
            mark = input(colored("  Mark as reviewed? [y/N]: ", C.CYAN)).strip().lower()
        except (KeyboardInterrupt, EOFError):
            mark = ""
            print()
        if mark == "y":
            run.state = RunState.REVIEWED
            run.state_changed_at = datetime.now(tz=timezone.utc)
            save_tracked_runs(runs=all_runs)
            print(colored("  Marked as reviewed.", C.GREEN))
            return True
        return False

    else:
        print(colored(f"  Unknown action: {choice}", C.RED))
        return False


# ---------------------------------------------------------------------------
# Catalog display
# ---------------------------------------------------------------------------


def _format_catalog_range(rng: tuple[int, int] | None) -> str:
    """Format a (min, max) range for display."""
    if rng is None:
        return colored("[none]", C.DIM)
    if rng[0] == rng[1]:
        return colored(str(rng[0]), C.YELLOW)
    return colored(f"{rng[0]}\u2192{rng[1]}", C.YELLOW)


def _format_tags(tags: list[str]) -> str:
    """Format tags as colored chips."""
    if not tags:
        return ""
    return " ".join(colored(f"[{t}]", C.MAGENTA) for t in tags)


def _format_catalog_line(idx: int, entry: CatalogEntry) -> str:
    """Format a single catalog entry as a display line (no description — shown separately)."""
    parts: list[str] = []
    parts.append(f" [{idx}]")
    parts.append(colored(entry.run_id, C.BOLD))

    # Model — short name (last path component)
    model_short = entry.base_model.rsplit("/", 1)[-1] if "/" in entry.base_model else entry.base_model
    parts.append(colored(model_short, C.CYAN))

    # Run name (directory basename) in dim
    run_name = Path(entry.run_dir).name
    parts.append(colored(run_name, C.DIM))

    # Checkpoint range
    parts.append(_format_catalog_range(rng=entry.checkpoint_range))

    # Framework
    _FRAMEWORK_COLORS: dict[str, str] = {"vfh": C.YELLOW, "tfh": C.MAGENTA}
    fw_color = _FRAMEWORK_COLORS.get(entry.source_framework, C.DIM)
    parts.append(colored(f"[{entry.source_framework}]", fw_color))

    # Tags
    tag_str = _format_tags(tags=entry.tags)
    if tag_str:
        parts.append(tag_str)

    return "  ".join(parts)


def _format_catalog_description(entry: CatalogEntry, full: bool = False) -> str:
    """Format description as an indented second line in dim text."""
    desc = entry.description or "(no description)"
    if not full:
        desc = trunc(text=desc, max_len=70)
    return colored(f"       ↳ {desc}", C.DIM)


def _apply_catalog_filters(entries: list[CatalogEntry], filt: CatalogFilter) -> list[CatalogEntry]:
    """Apply active filters. AND across filter types, OR within models/datasets."""
    result = entries
    if filt.tags:
        result = [e for e in result if all(t in e.tags for t in filt.tags)]
    if filt.exclude_tags:
        result = [e for e in result if not any(t in e.tags for t in filt.exclude_tags)]
    if filt.models:
        result = [e for e in result if e.base_model in filt.models]
    if filt.datasets:
        result = [e for e in result if e.train_dataset in filt.datasets]
    return result


def _display_catalog(catalog: Catalog, filt: CatalogFilter, show_full_descriptions: bool = False) -> list[CatalogEntry]:
    """Display catalog entries, applying filters. Returns flat display-order list."""
    entries = _apply_catalog_filters(entries=catalog.entries, filt=filt)

    # Sort by cataloged_at descending
    entries.sort(key=lambda e: e.cataloged_at, reverse=True)

    # Show active filters
    active_filters: list[str] = []
    if filt.tags:
        active_filters.append(colored("tags: " + ", ".join(filt.tags), C.MAGENTA))
    if filt.exclude_tags:
        active_filters.append(colored("exclude: " + ", ".join(filt.exclude_tags), C.RED))
    if filt.models:
        short_models = [m.rsplit("/", 1)[-1] if "/" in m else m for m in filt.models]
        active_filters.append(colored("models: " + ", ".join(short_models), C.CYAN))
    if filt.datasets:
        short_datasets = [d.rsplit("/", 1)[-1] if "/" in d else d for d in filt.datasets]
        active_filters.append(colored("datasets: " + ", ".join(short_datasets), C.GREEN))
    if active_filters:
        print(f"\n  {colored('Filters:', C.BOLD)} {' + '.join(active_filters)}")

    # Header
    has_filters = filt.tags or filt.exclude_tags or filt.models or filt.datasets
    count_str = f"{len(entries)}/{len(catalog.entries)}" if has_filters else str(len(entries))
    star = "★"
    print(f"\n {colored(star, C.YELLOW)} {colored(f'CATALOG ({count_str})', C.YELLOW, C.BOLD)}")

    if not entries:
        print(colored("  No catalog entries match filters.", C.DIM))
        return []

    for i, entry in enumerate(entries):
        print(_format_catalog_line(idx=i, entry=entry))
        print(_format_catalog_description(entry=entry, full=show_full_descriptions))

    return entries


# ---------------------------------------------------------------------------
# Catalog tree view
# ---------------------------------------------------------------------------


def _format_tree_entry(entry: CatalogEntry) -> str:
    """Compact one-line summary for tree view (no index — caller adds prefix)."""
    parts: list[str] = []
    parts.append(colored(entry.run_id, C.BOLD))
    model_short = entry.base_model.rsplit("/", 1)[-1] if "/" in entry.base_model else entry.base_model
    parts.append(colored(model_short, C.CYAN))
    parts.append(_format_catalog_range(rng=entry.checkpoint_range))
    tag_str = _format_tags(tags=entry.tags)
    if tag_str:
        parts.append(tag_str)
    return "  ".join(parts)


def _display_catalog_tree(
    catalog: Catalog,
    filt: CatalogFilter,
    show_full_descriptions: bool = False,
) -> list[CatalogEntry]:
    """Display catalog as lineage trees. Returns flat display-order list."""
    filtered = _apply_catalog_filters(entries=catalog.entries, filt=filt)
    if not filtered:
        print(colored("  No catalog entries match filters.", C.DIM))
        return []

    filtered_ids = {e.run_id for e in filtered}
    by_id: dict[str, CatalogEntry] = {e.run_id: e for e in filtered}

    # Show active filters (same as flat view)
    active_filters: list[str] = []
    if filt.tags:
        active_filters.append(colored("tags: " + ", ".join(filt.tags), C.MAGENTA))
    if filt.exclude_tags:
        active_filters.append(colored("exclude: " + ", ".join(filt.exclude_tags), C.RED))
    if filt.models:
        short_models = [m.rsplit("/", 1)[-1] if "/" in m else m for m in filt.models]
        active_filters.append(colored("models: " + ", ".join(short_models), C.CYAN))
    if filt.datasets:
        short_datasets = [d.rsplit("/", 1)[-1] if "/" in d else d for d in filt.datasets]
        active_filters.append(colored("datasets: " + ", ".join(short_datasets), C.GREEN))
    if active_filters:
        print(f"\n  {colored('Filters:', C.BOLD)} {' + '.join(active_filters)}")

    has_filters = filt.tags or filt.exclude_tags or filt.models or filt.datasets
    count_str = f"{len(filtered)}/{len(catalog.entries)}" if has_filters else str(len(filtered))
    print(f"\n {colored('⌥', C.YELLOW)} {colored(f'CATALOG TREE ({count_str})', C.YELLOW, C.BOLD)}")

    # Find roots: entries whose follows are all outside the filtered set
    roots: list[CatalogEntry] = []
    for entry in filtered:
        parents_in_catalog = [f for f in entry.follows if f in filtered_ids]
        if not parents_in_catalog:
            roots.append(entry)

    # Sort roots by cataloged_at ascending (oldest first — trees read top-down chronologically)
    roots.sort(key=lambda e: e.cataloged_at)

    # DFS render
    display_order: list[CatalogEntry] = []
    # Fixed-width index field so tree connectors stay aligned
    idx_width = len(str(len(filtered) - 1)) if filtered else 1

    def _render(entry: CatalogEntry, prefix: str, is_last: bool, is_root: bool) -> None:
        idx = len(display_order)
        display_order.append(entry)

        # Tree connector
        if is_root:
            connector = ""
            child_prefix = "  "
        else:
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")

        idx_str = f" [{idx:>{idx_width}}]"
        line = f"{idx_str} {prefix}{connector}{_format_tree_entry(entry=entry)}"
        print(line)

        # Description line — align ↳ under the tree connector / entry content
        desc = entry.description or "(no description)"
        if not show_full_descriptions:
            desc = trunc(text=desc, max_len=70)
        pad = " " * len(idx_str)
        print(colored(f"{pad} {child_prefix}↳ {desc}", C.DIM))

        # Children: preceded_by entries that are in the filtered set
        children = [by_id[rid] for rid in entry.preceded_by if rid in filtered_ids]
        # Sort children by cataloged_at
        children.sort(key=lambda e: e.cataloged_at)
        for i, child in enumerate(children):
            _render(entry=child, prefix=child_prefix, is_last=(i == len(children) - 1), is_root=False)

    for i, root in enumerate(roots):
        if i > 0:
            print()  # blank line between trees
        _render(entry=root, prefix="", is_last=True, is_root=True)

    return display_order


# ---------------------------------------------------------------------------
# Catalog detail + actions
# ---------------------------------------------------------------------------


def _show_catalog_detail(entry: CatalogEntry) -> None:
    """Show detailed info about a catalog entry."""
    print(f"\n  {colored('Catalog Entry', C.BOLD, C.YELLOW)}")
    print(f"  {'ID:':<14} {colored(entry.run_id, C.BOLD)}")
    print(f"  {'Dir:':<14} {trunc(text=entry.run_dir, max_len=80)}")
    model_short = entry.base_model.rsplit("/", 1)[-1] if "/" in entry.base_model else entry.base_model
    print(f"  {'Model:':<14} {colored(model_short, C.CYAN)}")
    print(f"  {'Description:':<14} {entry.description or '(none)'}")
    print(f"  {'Tags:':<14} {_format_tags(tags=entry.tags) or colored('(none)', C.DIM)}")
    print(f"  {'Checkpoints:':<14} {_format_catalog_range(rng=entry.checkpoint_range)}")
    print(f"  {'Rollouts:':<14} {_format_catalog_range(rng=entry.rollout_range)}")
    print(f"  {'Dataset:':<14} {trunc(text=entry.train_dataset, max_len=60)}")
    print(f"  {'Framework:':<14} {entry.source_framework}")
    if entry.wandb_url:
        print(f"  {'W&B:':<14} {colored(entry.wandb_url, C.BLUE)}")
    if entry.follows:
        print(f"  {'Follows:':<14} {', '.join(entry.follows)}")
    if entry.preceded_by:
        print(f"  {'Preceded by:':<14} {', '.join(entry.preceded_by)}")
    # Reward config summary (skip internal keys starting with _)
    reward_keys = {k: v for k, v in entry.reward_config.items() if not k.startswith("_")}
    if reward_keys:
        summary = ", ".join(f"{k}={v}" for k, v in list(reward_keys.items())[:5])
        print(f"  {'Reward:':<14} {trunc(text=summary, max_len=70)}")
    print(f"  {'Cataloged:':<14} {entry.cataloged_at}")


def _tag_edit_flow(entry: CatalogEntry, catalog: Catalog, catalog_path: Path) -> bool:
    """Interactive tag editing. Returns True if tags changed."""
    all_tags = [t.name for t in catalog.tags]
    current = set(entry.tags)
    print(f"\n  Current tags: {_format_tags(tags=entry.tags) or colored('(none)', C.DIM)}")
    print(f"  {colored('Toggle tags by index, or +new_tag to create. Empty to finish.', C.DIM)}")

    changed = False
    while True:
        # List all tags with toggle state
        for i, tag_name in enumerate(all_tags):
            marker = colored("\u2713", C.GREEN) if tag_name in current else " "
            print(f"    [{i}] {marker} {tag_name}")

        try:
            raw = input_or_esc(prompt="  Tag (index/+name/empty): ").strip()
        except UserCancelled:
            break
        if not raw:
            break

        if raw.startswith("+"):
            new_name = raw[1:].strip()
            if new_name and new_name not in all_tags:
                try:
                    desc = input_or_esc(prompt="  Description: ").strip()
                except UserCancelled:
                    desc = ""
                catalog.tags.append(CatalogTag(name=new_name, description=desc))
                all_tags.append(new_name)
                current.add(new_name)
                changed = True
                print(colored(f"    Created + added tag: {new_name}", C.GREEN))
            elif new_name in all_tags:
                current.add(new_name)
                changed = True
            continue

        try:
            idx = int(raw)
            if 0 <= idx < len(all_tags):
                tag_name = all_tags[idx]
                if tag_name in current:
                    current.discard(tag_name)
                else:
                    current.add(tag_name)
                changed = True
        except ValueError:
            print(colored(f"    Unknown input: {raw}", C.RED))

    if changed:
        entry.tags = sorted(current)
        save_catalog(catalog=catalog, catalog_path=catalog_path)
    return changed


def _catalog_action_menu(entry: CatalogEntry, catalog: Catalog, catalog_path: Path) -> bool:
    """Show action menu for a catalog entry. Returns True if list needs redisplay."""
    _show_catalog_detail(entry=entry)

    actions = "[w]andb  [p]ath  [l]aunch cmd  [e]dit desc  [t]ags  [r]emove"
    print(f"\n  {colored(actions, C.DIM)}")

    try:
        choice = input_or_esc(prompt="  Action: ").strip().lower()
    except UserCancelled:
        return False

    if choice == "w":
        copy_and_print_url(url=entry.wandb_url, label="W&B URL")
        return False

    elif choice == "p":
        copy_and_print_path(path=entry.run_dir)
        return False

    elif choice == "l":
        launch_file = Path(entry.run_dir) / "launching_command.txt"
        if launch_file.exists():
            cmd = launch_file.read_text().strip()
            copy_to_clipboard(text=cmd)
            print(f"  {colored(cmd, C.CYAN)}")
        else:
            print(colored("  No launching_command.txt found.", C.RED))
        return False

    elif choice == "e":
        print(f"  Current: {entry.description or '(none)'}")
        try:
            new_desc = input_or_esc(prompt="  New description: ", prefill=entry.description or "").strip()
        except UserCancelled:
            return False
        if new_desc:
            entry.description = new_desc
            save_catalog(catalog=catalog, catalog_path=catalog_path)
            print(colored("  Description updated.", C.GREEN))
            return True
        return False

    elif choice == "t":
        return _tag_edit_flow(entry=entry, catalog=catalog, catalog_path=catalog_path)

    elif choice == "r":
        try:
            confirm = input_or_esc(prompt=f"  Remove {entry.run_id} from catalog? [y/N]: ")
        except UserCancelled:
            return False
        if confirm.strip().lower() == "y":
            catalog.entries.remove(entry)
            save_catalog(catalog=catalog, catalog_path=catalog_path)
            print(colored("  Removed from catalog.", C.GREEN))
            return True
        return False

    else:
        print(colored(f"  Unknown action: {choice}", C.RED))
        return False


# ---------------------------------------------------------------------------
# Catalog filter flow
# ---------------------------------------------------------------------------


def _filter_tag_flow(catalog: Catalog, filt: CatalogFilter) -> bool:
    """Interactive tag filter toggle — 3-state cycle: off → include → exclude → off."""
    all_tag_names = [t.name for t in catalog.tags]
    if not all_tag_names:
        print(colored("  No tags in catalog.", C.DIM))
        return False

    changed = False
    while True:
        # Count entries per tag (respecting current model + dataset filters)
        tag_counts: dict[str, int] = {}
        for tag_name in all_tag_names:
            count = sum(
                1 for e in catalog.entries
                if tag_name in e.tags
                and (not filt.models or e.base_model in filt.models)
                and (not filt.datasets or e.train_dataset in filt.datasets)
            )
            tag_counts[tag_name] = count

        print(f"\n  {colored('Tags:', C.BOLD)}  {colored('Toggle by index (cycles: off → include → exclude → off). Empty to finish.', C.DIM)}")
        for i, tag_name in enumerate(all_tag_names):
            if tag_name in filt.tags:
                marker = colored("\u2713", C.GREEN)
                name_str = colored(tag_name, C.MAGENTA, C.BOLD)
            elif tag_name in filt.exclude_tags:
                marker = colored("\u2717", C.RED)
                name_str = colored(tag_name, C.RED)
            else:
                marker = " "
                name_str = colored(tag_name, C.MAGENTA)
            count_str = colored(f"({tag_counts[tag_name]})", C.DIM)
            print(f"    [{i}] {marker} {name_str} {count_str}")

        try:
            raw = input_or_esc(prompt="  Tag index: ").strip()
        except UserCancelled:
            break
        if not raw:
            break

        try:
            idx = int(raw)
            if 0 <= idx < len(all_tag_names):
                tag_name = all_tag_names[idx]
                if tag_name in filt.tags:
                    # include → exclude
                    filt.tags.remove(tag_name)
                    filt.exclude_tags.append(tag_name)
                    print(colored(f"  Excluding: {tag_name}", C.RED))
                elif tag_name in filt.exclude_tags:
                    # exclude → off
                    filt.exclude_tags.remove(tag_name)
                    print(colored(f"  Cleared: {tag_name}", C.YELLOW))
                else:
                    # off → include
                    filt.tags.append(tag_name)
                    print(colored(f"  Including: {tag_name}", C.GREEN))
                changed = True
            else:
                print(colored(f"  Index out of range (0-{len(all_tag_names) - 1})", C.RED))
        except ValueError:
            print(colored(f"  Unknown input: {raw}", C.RED))

    return changed


def _filter_model_flow(catalog: Catalog, filt: CatalogFilter) -> bool:
    """Interactive model filter — toggle by index, like tags."""
    # Scope models to entries matching current tag + dataset filters
    scoped_entries = catalog.entries
    if filt.tags:
        scoped_entries = [e for e in scoped_entries if all(t in e.tags for t in filt.tags)]
    if filt.datasets:
        scoped_entries = [e for e in scoped_entries if e.train_dataset in filt.datasets]

    # Build unique models with counts
    model_counts: dict[str, int] = {}
    for e in scoped_entries:
        model_counts[e.base_model] = model_counts.get(e.base_model, 0) + 1
    unique_models = sorted(model_counts.keys())

    if not unique_models:
        print(colored("  No models match current filters.", C.DIM))
        return False

    short_names = [m.rsplit("/", 1)[-1] if "/" in m else m for m in unique_models]
    current_set = set(filt.models)

    changed = False
    while True:
        print(f"\n  {colored('Models:', C.BOLD)}  {colored('Toggle by index, empty to finish.', C.DIM)}")
        for i, (short, full) in enumerate(zip(short_names, unique_models)):
            active = full in current_set
            marker = colored("\u2713", C.GREEN) if active else " "
            name_str = colored(short, C.CYAN, C.BOLD) if active else colored(short, C.CYAN)
            count_str = colored(f"({model_counts[full]})", C.DIM)
            print(f"    [{i}] {marker} {name_str} {count_str}")

        try:
            raw = input_or_esc(prompt="  Model index: ").strip()
        except UserCancelled:
            break
        if not raw:
            break

        try:
            idx = int(raw)
            if 0 <= idx < len(unique_models):
                full_model = unique_models[idx]
                short = short_names[idx]
                if full_model in current_set:
                    current_set.discard(full_model)
                    print(colored(f"  Removed: {short}", C.YELLOW))
                else:
                    current_set.add(full_model)
                    print(colored(f"  Added: {short}", C.GREEN))
                changed = True
            else:
                print(colored(f"  Index out of range (0-{len(unique_models) - 1})", C.RED))
        except ValueError:
            print(colored(f"  Unknown input: {raw}", C.RED))

    if changed:
        filt.models = sorted(current_set)
    return changed


def _filter_dataset_flow(catalog: Catalog, filt: CatalogFilter) -> bool:
    """Interactive dataset filter — toggle by index, like tags/models."""
    # Scope to entries matching current tag + model filters
    scoped_entries = catalog.entries
    if filt.tags:
        scoped_entries = [e for e in scoped_entries if all(t in e.tags for t in filt.tags)]
    if filt.models:
        scoped_entries = [e for e in scoped_entries if e.base_model in filt.models]

    # Build unique datasets with counts
    dataset_counts: dict[str, int] = {}
    for e in scoped_entries:
        dataset_counts[e.train_dataset] = dataset_counts.get(e.train_dataset, 0) + 1
    unique_datasets = sorted(dataset_counts.keys())

    if not unique_datasets:
        print(colored("  No datasets match current filters.", C.DIM))
        return False

    short_names = [d.rsplit("/", 1)[-1] if "/" in d else d for d in unique_datasets]
    current_set = set(filt.datasets)

    changed = False
    while True:
        print(f"\n  {colored('Datasets:', C.BOLD)}  {colored('Toggle by index, empty to finish.', C.DIM)}")
        for i, (short, full) in enumerate(zip(short_names, unique_datasets)):
            active = full in current_set
            marker = colored("\u2713", C.GREEN) if active else " "
            name_str = colored(short, C.GREEN, C.BOLD) if active else colored(short, C.GREEN)
            count_str = colored(f"({dataset_counts[full]})", C.DIM)
            print(f"    [{i}] {marker} {name_str} {count_str}")

        try:
            raw = input_or_esc(prompt="  Dataset index: ").strip()
        except UserCancelled:
            break
        if not raw:
            break

        try:
            idx = int(raw)
            if 0 <= idx < len(unique_datasets):
                full_dataset = unique_datasets[idx]
                short = short_names[idx]
                if full_dataset in current_set:
                    current_set.discard(full_dataset)
                    print(colored(f"  Removed: {short}", C.YELLOW))
                else:
                    current_set.add(full_dataset)
                    print(colored(f"  Added: {short}", C.GREEN))
                changed = True
            else:
                print(colored(f"  Index out of range (0-{len(unique_datasets) - 1})", C.RED))
        except ValueError:
            print(colored(f"  Unknown input: {raw}", C.RED))

    if changed:
        filt.datasets = sorted(current_set)
    return changed


def _filter_menu(catalog: Catalog, filt: CatalogFilter) -> bool:
    """Show filter sub-menu. Returns True if filters changed."""
    print(f"\n  {colored('Filter by:', C.BOLD)}  [t]ag  [m]odel  [d]ataset  [c]lear all")
    try:
        choice = input_or_esc(prompt="  Filter: ").strip().lower()
    except UserCancelled:
        return False

    if choice == "t":
        return _filter_tag_flow(catalog=catalog, filt=filt)

    elif choice == "m":
        return _filter_model_flow(catalog=catalog, filt=filt)

    elif choice == "d":
        return _filter_dataset_flow(catalog=catalog, filt=filt)

    elif choice == "c":
        if filt.tags or filt.exclude_tags or filt.models or filt.datasets:
            filt.tags.clear()
            filt.exclude_tags.clear()
            filt.models.clear()
            filt.datasets.clear()
            print(colored("  All filters cleared.", C.YELLOW))
            return True
        print(colored("  No active filters.", C.DIM))
        return False

    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _do_refresh(runs: list[TrackedRun]) -> list[TrackedRun]:
    """Poll wandb, refresh disk state, report transitions, and save."""
    print(colored("Refreshing run states...", C.DIM))
    old_states = {id(r): r.state for r in runs}
    runs = refresh_run_states(runs=runs)

    # Report transitions
    transitions: list[str] = []
    for run in runs:
        old = old_states.get(id(run))
        if old is not None and old != run.state:
            transitions.append(
                f"  {run.run_id or '?'}: {old.value} \u2192 {run.state.value}"
            )
    if transitions:
        print(colored("State changes:", C.BOLD))
        for t in transitions:
            print(t)
    else:
        print(colored("  No state changes.", C.DIM))

    save_tracked_runs(runs=runs)
    return runs


def main() -> None:
    config = tyro.cli(ViewerConfig)

    runs = load_tracked_runs()

    if config.refresh and runs:
        runs = _do_refresh(runs=runs)

    # Catalog state
    viewing_catalog = False
    catalog_filter = CatalogFilter()
    catalog_path = _get_catalog_path()
    catalog: Catalog | None = None  # lazy-loaded on first [c]
    show_full_descriptions = False
    tree_view = False

    # Interactive loop
    while True:
        if viewing_catalog:
            if catalog is None:
                catalog = load_catalog(catalog_path=catalog_path)
                refresh_catalog_lineage(catalog=catalog)
            if tree_view:
                display_catalog = _display_catalog_tree(catalog=catalog, filt=catalog_filter, show_full_descriptions=show_full_descriptions)
            else:
                display_catalog = _display_catalog(catalog=catalog, filt=catalog_filter, show_full_descriptions=show_full_descriptions)

            filter_hint = ""
            if catalog_filter.tags or catalog_filter.exclude_tags or catalog_filter.models or catalog_filter.datasets:
                filter_hint = " *"
            desc_label = colored("[d]esc*", C.DIM) if show_full_descriptions else colored("[d]esc", C.DIM)
            tree_label = colored("[t]ree*", C.DIM) if tree_view else colored("[t]ree", C.DIM)
            print(f"\n  {colored(f'[#] select  [c]tracked  [r]eload  [q]uit  [f]ilter{filter_hint}  ', C.DIM)}{desc_label}  {tree_label}")

            try:
                raw = input_or_esc(prompt="\n> ").strip().lower()
            except UserCancelled:
                break

            if raw in ("q", "quit", ""):
                break

            if raw == "c":
                viewing_catalog = False
                print(colored("  Switched to tracked runs.", C.CYAN))
                continue

            if raw == "r":
                catalog = load_catalog(catalog_path=catalog_path)
                refresh_catalog_lineage(catalog=catalog)
                print(colored("  Catalog reloaded.", C.GREEN))
                continue

            if raw == "f":
                _filter_menu(catalog=catalog, filt=catalog_filter)
                continue

            if raw == "d":
                show_full_descriptions = not show_full_descriptions
                label = "ON" if show_full_descriptions else "OFF"
                print(colored(f"  Full descriptions: {label}", C.YELLOW))
                continue

            if raw == "t":
                tree_view = not tree_view
                label = "ON" if tree_view else "OFF"
                print(colored(f"  Tree view: {label}", C.YELLOW))
                continue

            # Numeric selection
            try:
                idx = int(raw)
            except ValueError:
                print(colored(f"  Unknown command: {raw}", C.RED))
                continue

            if 0 <= idx < len(display_catalog):
                _catalog_action_menu(entry=display_catalog[idx], catalog=catalog, catalog_path=catalog_path)
            else:
                max_idx = len(display_catalog) - 1 if display_catalog else 0
                print(colored(f"  Index out of range (0-{max_idx})", C.RED))

        else:
            # Tracker view (original)
            display_list = _display_runs(runs=runs, config=config)
            if not display_list and not runs:
                # No runs at all — offer to switch to catalog
                print(colored("  No tracked runs. Press [c] to view catalog.", C.DIM))

            print(f"\n  {colored('[#] select  [c]atalog  [r]efresh  [q]uit  [f]ilter empty  [a]ll (show reviewed)', C.DIM)}")

            try:
                raw = input_or_esc(prompt="\n> ").strip().lower()
            except UserCancelled:
                break

            if raw in ("q", "quit", ""):
                break

            if raw == "c":
                viewing_catalog = True
                print(colored("  Switched to catalog view.", C.YELLOW))
                continue

            if raw == "f":
                config.filter_empty = not config.filter_empty
                label = "ON" if config.filter_empty else "OFF"
                print(colored(f"  Filter empty: {label}", C.YELLOW))
                continue

            if raw == "a":
                config.show_reviewed = not config.show_reviewed
                label = "ON" if config.show_reviewed else "OFF"
                print(colored(f"  Show reviewed: {label}", C.YELLOW))
                continue

            if raw == "r":
                runs = load_tracked_runs()
                runs = _do_refresh(runs=runs)
                continue

            # Try numeric selection
            try:
                idx = int(raw)
            except ValueError:
                print(colored(f"  Unknown command: {raw}", C.RED))
                continue

            if 0 <= idx < len(display_list):
                _action_menu(run=display_list[idx], all_runs=runs)
            else:
                max_idx = len(display_list) - 1 if display_list else 0
                print(colored(f"  Index out of range (0-{max_idx})", C.RED))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, UserCancelled):
        print()
