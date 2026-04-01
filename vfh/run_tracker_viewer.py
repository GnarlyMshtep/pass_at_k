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

from vfh.catalog import load_catalog, save_catalog
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
    """Active filters for catalog view. AND logic: entry must match ALL active filters."""
    tags: list[str] = field(default_factory=list)       # entry must have ALL of these tags
    model: str | None = None                            # exact model name (selected via fuzzy)


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
# Fuzzy search helpers
# ---------------------------------------------------------------------------


def _fuzzy_search(query: str, items: list[str], limit: int = 10) -> list[tuple[str, float]]:
    """Fuzzy search strings. Returns (item, score) pairs above threshold."""
    if not query or not items:
        return [(item, 100.0) for item in items[:limit]]
    try:
        from rapidfuzz import fuzz
        scored = [(item, fuzz.partial_ratio(query.lower(), item.lower())) for item in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(item, score) for item, score in scored[:limit] if score > 40]
    except ImportError:
        # Fallback: substring match
        lower_q = query.lower()
        return [(item, 100.0) for item in items if lower_q in item.lower()][:limit]


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
    """Format a single catalog entry as a display line."""
    parts: list[str] = []
    parts.append(f" [{idx}]")
    parts.append(colored(entry.run_id, C.BOLD))

    # Model — short name (last path component)
    model_short = entry.base_model.rsplit("/", 1)[-1] if "/" in entry.base_model else entry.base_model
    parts.append(colored(model_short, C.CYAN))

    # Description
    parts.append(trunc(text=entry.description or "(no description)", max_len=45))

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


def _apply_catalog_filters(entries: list[CatalogEntry], filt: CatalogFilter) -> list[CatalogEntry]:
    """Apply active filters (AND logic)."""
    result = entries
    if filt.tags:
        result = [e for e in result if all(t in e.tags for t in filt.tags)]
    if filt.model is not None:
        result = [e for e in result if e.base_model == filt.model]
    return result


def _display_catalog(catalog: Catalog, filt: CatalogFilter) -> list[CatalogEntry]:
    """Display catalog entries, applying filters. Returns flat display-order list."""
    entries = _apply_catalog_filters(entries=catalog.entries, filt=filt)

    # Sort by cataloged_at descending
    entries.sort(key=lambda e: e.cataloged_at, reverse=True)

    # Show active filters
    active_filters: list[str] = []
    if filt.tags:
        active_filters.append(colored("tags: " + ", ".join(filt.tags), C.MAGENTA))
    if filt.model is not None:
        model_short = filt.model.rsplit("/", 1)[-1] if "/" in filt.model else filt.model
        active_filters.append(colored("model: " + model_short, C.CYAN))
    if active_filters:
        print(f"\n  {colored('Filters:', C.BOLD)} {' + '.join(active_filters)}")

    # Header
    count_str = f"{len(entries)}/{len(catalog.entries)}" if (filt.tags or filt.model) else str(len(entries))
    star = "★"
    print(f"\n {colored(star, C.YELLOW)} {colored(f'CATALOG ({count_str})', C.YELLOW, C.BOLD)}")

    if not entries:
        print(colored("  No catalog entries match filters.", C.DIM))
        return []

    for i, entry in enumerate(entries):
        print(_format_catalog_line(idx=i, entry=entry))

    return entries


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
                catalog.tags.append(CatalogTag(name=new_name, description=""))
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
    """Interactive tag filter toggle — show all tags, toggle on/off by index. Returns True if changed."""
    all_tag_names = [t.name for t in catalog.tags]
    if not all_tag_names:
        print(colored("  No tags in catalog.", C.DIM))
        return False

    changed = False
    while True:
        # Count entries per tag (respecting current model filter)
        tag_counts: dict[str, int] = {}
        for tag_name in all_tag_names:
            count = sum(
                1 for e in catalog.entries
                if tag_name in e.tags and (filt.model is None or e.base_model == filt.model)
            )
            tag_counts[tag_name] = count

        print(f"\n  {colored('Tags:', C.BOLD)}  {colored('Toggle by index, empty to finish.', C.DIM)}")
        for i, tag_name in enumerate(all_tag_names):
            active = tag_name in filt.tags
            marker = colored("\u2713", C.GREEN) if active else " "
            name_str = colored(tag_name, C.MAGENTA, C.BOLD) if active else colored(tag_name, C.MAGENTA)
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
                    filt.tags.remove(tag_name)
                    print(colored(f"  Removed: {tag_name}", C.YELLOW))
                else:
                    filt.tags.append(tag_name)
                    print(colored(f"  Added: {tag_name}", C.GREEN))
                changed = True
            else:
                print(colored(f"  Index out of range (0-{len(all_tag_names) - 1})", C.RED))
        except ValueError:
            print(colored(f"  Unknown input: {raw}", C.RED))

    return changed


def _filter_model_flow(catalog: Catalog, filt: CatalogFilter) -> bool:
    """Interactive model filter — show models (scoped to active tag filters), fuzzy search + select."""
    # Scope models to entries matching current tag filters
    scoped_entries = catalog.entries
    if filt.tags:
        scoped_entries = [e for e in scoped_entries if all(t in e.tags for t in filt.tags)]

    # Build unique full-path → short-name mapping, with entry counts
    model_counts: dict[str, int] = {}
    for e in scoped_entries:
        model_counts[e.base_model] = model_counts.get(e.base_model, 0) + 1
    unique_models = sorted(model_counts.keys())

    if not unique_models:
        print(colored("  No models match current tag filters.", C.DIM))
        return False

    short_names = [m.rsplit("/", 1)[-1] if "/" in m else m for m in unique_models]
    current_short = filt.model.rsplit("/", 1)[-1] if filt.model and "/" in filt.model else filt.model

    print(f"\n  {colored('Models:', C.BOLD)}  {colored('Search or empty to show all.', C.DIM)}")
    try:
        query = input_or_esc(prompt="  Model search: ").strip()
    except UserCancelled:
        return False

    matches = _fuzzy_search(query=query, items=short_names)
    if not matches:
        print(colored("  No matching models.", C.DIM))
        return False

    for i, (name, _score) in enumerate(matches):
        active = name == current_short
        marker = colored("\u2713", C.GREEN) if active else " "
        name_str = colored(name, C.CYAN, C.BOLD) if active else colored(name, C.CYAN)
        full = unique_models[short_names.index(name)]
        count_str = colored(f"({model_counts[full]})", C.DIM)
        print(f"    [{i}] {marker} {name_str} {count_str}")

    try:
        raw = input_or_esc(prompt="  Model index: ").strip()
    except UserCancelled:
        return False
    if not raw:
        return False

    try:
        idx = int(raw)
        if 0 <= idx < len(matches):
            selected_short = matches[idx][0]
            full_model = unique_models[short_names.index(selected_short)]
            if filt.model == full_model:
                filt.model = None
                print(colored(f"  Removed model filter", C.YELLOW))
            else:
                filt.model = full_model
                print(colored(f"  Filtering by: {selected_short}", C.GREEN))
            return True
    except ValueError:
        pass
    return False


def _filter_menu(catalog: Catalog, filt: CatalogFilter) -> bool:
    """Show filter sub-menu. Returns True if filters changed."""
    print(f"\n  {colored('Filter by:', C.BOLD)}  [t]ag  [m]odel  [c]lear all")
    try:
        choice = input_or_esc(prompt="  Filter: ").strip().lower()
    except UserCancelled:
        return False

    if choice == "t":
        return _filter_tag_flow(catalog=catalog, filt=filt)

    elif choice == "m":
        return _filter_model_flow(catalog=catalog, filt=filt)

    elif choice == "c":
        if filt.tags or filt.model:
            filt.tags.clear()
            filt.model = None
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

    # Interactive loop
    while True:
        if viewing_catalog:
            if catalog is None:
                catalog = load_catalog(catalog_path=catalog_path)
            display_catalog = _display_catalog(catalog=catalog, filt=catalog_filter)

            filter_hint = ""
            if catalog_filter.tags or catalog_filter.model:
                filter_hint = " *"
            print(f"\n  {colored(f'[#] select  [c]tracked  [r]eload  [q]uit  [f]ilter{filter_hint}', C.DIM)}")

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
                print(colored("  Catalog reloaded.", C.GREEN))
                continue

            if raw == "f":
                _filter_menu(catalog=catalog, filt=catalog_filter)
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
