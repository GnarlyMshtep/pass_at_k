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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Allow running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tyro

from vfh.catalog import (
    _display_tags_grouped,
    _handle_plus_command,
    _tags_in_display_order,
    load_catalog,
    refresh_catalog_lineage,
    save_catalog,
)
from vfh.catalog_types import Catalog, CatalogEntry, CatalogTag, TagCategory
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
from vfh.wandb_view import (
    MixedProjectError,
    WandbRunRef,
    create_wandb_view_url,
    parse_wandb_url,
)


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


def _catalog_filter_path() -> Path:
    """Path to persisted catalog filter state — sibling of catalog.json."""
    return _get_catalog_path().parent / "catalog_filter.json"


def _save_catalog_filter(filt: CatalogFilter) -> None:
    """Persist current catalog filter to disk."""
    import json as _json
    path = _catalog_filter_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump({"tags": filt.tags, "exclude_tags": filt.exclude_tags,
                     "models": filt.models, "datasets": filt.datasets}, f, indent=2)


def _load_catalog_filter() -> CatalogFilter:
    """Load persisted catalog filter from disk, or return empty filter."""
    import json as _json
    path = _catalog_filter_path()
    if not path.exists():
        return CatalogFilter()
    try:
        with open(path) as f:
            data = _json.load(f)
        return CatalogFilter(
            tags=data.get("tags", []),
            exclude_tags=data.get("exclude_tags", []),
            models=data.get("models", []),
            datasets=data.get("datasets", []),
        )
    except Exception:
        return CatalogFilter()


def _get_catalog_path() -> Path:
    """Catalog file path — same logic as catalog.py."""
    import os
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("RUN_CATALOG_PATH", str(_REPO_ROOT / "logs" / "catalog_data" / "catalog.json")))


# Lineage cache for tracker tree view. Maps run_dir → (parent_run_id, child_run_ids).
# Cleared on [r]efresh so newly-appended child_run_ids are picked up.
_LINEAGE_CACHE: dict[str, tuple[str | None, list[str]]] = {}


def _resolve_run_dir(run_dir: str) -> Path:
    """Repo-relative run_dirs (VFH) vs absolute (TFH) — normalize."""
    p = Path(run_dir)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / run_dir


def _read_lineage(run_dir: str) -> tuple[str | None, list[str]]:
    """Read (parent_run_id, child_run_ids) from run_metadata.json5. Cached."""
    if run_dir in _LINEAGE_CACHE:
        return _LINEAGE_CACHE[run_dir]
    result: tuple[str | None, list[str]] = (None, [])
    meta_path = _resolve_run_dir(run_dir=run_dir) / "run_metadata.json5"
    if meta_path.exists():
        try:
            import pyjson5
            with open(meta_path) as f:
                meta = pyjson5.load(f)
            parent = (meta.get("origin") or {}).get("parent_run_id")
            children = meta.get("child_run_ids") or []
            result = (parent, list(children))
        except Exception:
            pass  # fall back to no lineage
    _LINEAGE_CACHE[run_dir] = result
    return result


def _clear_lineage_cache() -> None:
    _LINEAGE_CACHE.clear()


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


def _format_run_content(run: TrackedRun) -> str:
    """Format a run as a display line *without* the index prefix.

    Split out so tree-view rendering can prepend its own tree connectors
    instead of an index-only prefix.
    """
    parts: list[str] = []

    # Run ID
    run_id_str = run.run_id or "?"
    parts.append(colored(run_id_str, C.BOLD))

    # Model
    if run.base_model:
        parts.append(colored(run.base_model, C.CYAN))

    # Description — full (not truncated)
    desc = run.description or Path(run.run_dir).name
    parts.append(desc)

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


def _format_run_line(idx: int, run: TrackedRun) -> str:
    """Format a single run as a display line, with [idx] prefix."""
    return f" [{idx}]  {_format_run_content(run=run)}"


def _get_run_note(run: TrackedRun) -> str | None:
    """Get a run's note: prefer comments from tracker, fall back to NOTE.md on disk."""
    if run.comments:
        return run.comments
    note_path = Path(run.run_dir) / "NOTE.md"
    if note_path.exists():
        try:
            return note_path.read_text().strip()
        except OSError:
            pass
    return None


def _display_runs(runs: list[TrackedRun], config: ViewerConfig, show_notes: bool = False,
                   hours_filter: float | None = None) -> list[TrackedRun]:
    """Display runs grouped by state. Returns the flat display-order list."""
    display_runs: list[TrackedRun] = []
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours_filter) if hours_filter else None

    for state in _STATE_ORDER:
        if state == RunState.REVIEWED and not config.show_reviewed:
            continue

        group = [r for r in runs if r.state == state]
        if config.filter_empty:
            group = [r for r in group if r.checkpoint_steps]
        if cutoff:
            group = [r for r in group if r.registered_at >= cutoff]
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
            if show_notes:
                note = _get_run_note(run=run)
                if note:
                    note_lines = note.splitlines() or [""]
                    print(colored(f"        ↳ {note_lines[0]}", C.DIM))
                    for extra_line in note_lines[1:]:
                        print(colored(f"          {extra_line}", C.DIM))
            display_runs.append(run)

    if not display_runs:
        print(colored("\n  No tracked runs.", C.DIM))

    return display_runs


def _display_runs_tree(runs: list[TrackedRun], config: ViewerConfig,
                       show_notes: bool = False,
                       hours_filter: float | None = None) -> list[TrackedRun]:
    """Display tracked runs as lineage trees via parent/child links from run_metadata.json5.

    Shares filter semantics with `_display_runs` (state, filter_empty, hours).
    Runs whose parent_run_id is NOT in the filtered set become roots; chains
    of tracked runs nest underneath.
    """
    # Filter (same as _display_runs)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours_filter) if hours_filter else None
    filtered: list[TrackedRun] = []
    for r in runs:
        if r.state == RunState.REVIEWED and not config.show_reviewed:
            continue
        if config.filter_empty and not r.checkpoint_steps:
            continue
        if cutoff and r.registered_at < cutoff:
            continue
        filtered.append(r)

    if not filtered:
        print(colored("\n  No tracked runs match filters.", C.DIM))
        return []

    by_id: dict[str, TrackedRun] = {r.run_id: r for r in filtered if r.run_id}
    filtered_ids = set(by_id.keys())

    # Build children map restricted to filtered set
    children_map: dict[str, list[str]] = {rid: [] for rid in filtered_ids}
    has_parent_in_set: dict[str, bool] = {rid: False for rid in filtered_ids}
    for rid, run in by_id.items():
        parent, _children = _read_lineage(run_dir=run.run_dir)
        if parent and parent in filtered_ids:
            children_map[parent].append(rid)
            has_parent_in_set[rid] = True

    roots = [by_id[rid] for rid, has_p in has_parent_in_set.items() if not has_p]
    # Oldest first (trees read top-down chronologically, matching catalog tree)
    roots.sort(key=lambda r: r.registered_at)

    # Also include runs with no run_id (rare, TFH oddities) as bare roots
    orphans = [r for r in filtered if not r.run_id]

    total = len(filtered)
    print(f"\n {colored('⌥', C.CYAN)} {colored(f'TRACKED RUNS TREE ({total})', C.CYAN, C.BOLD)}")

    display_order: list[TrackedRun] = []
    idx_width = len(str(total - 1)) if total else 1

    def _render(run: TrackedRun, prefix: str, is_last: bool, is_root: bool) -> None:
        idx = len(display_order)
        display_order.append(run)

        if is_root:
            connector = ""
            child_prefix = "  "
        else:
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")

        idx_str = f" [{idx:>{idx_width}}]"
        print(f"{idx_str} {prefix}{connector}{_format_run_content(run=run)}")

        if show_notes:
            note = _get_run_note(run=run)
            if note:
                pad = " " * len(idx_str)
                for nline in note.splitlines():
                    print(colored(f"{pad} {child_prefix}↳ {nline}", C.DIM))

        kids = [by_id[cid] for cid in children_map.get(run.run_id or "", []) if cid in by_id]
        kids.sort(key=lambda r: r.registered_at)
        for i, child in enumerate(kids):
            _render(run=child, prefix=child_prefix, is_last=(i == len(kids) - 1), is_root=False)

    for i, root in enumerate(roots):
        if i > 0:
            print()
        _render(run=root, prefix="", is_last=True, is_root=True)

    for r in orphans:
        print()
        idx = len(display_order)
        display_order.append(r)
        idx_str = f" [{idx:>{idx_width}}]"
        print(f"{idx_str} {_format_run_content(run=r)}")

    return display_order


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _show_run_detail(run: TrackedRun) -> None:
    """Show detailed info about a single run."""
    print(f"\n  {colored('Run Details', C.BOLD, C.CYAN)}")
    print(f"  {'ID:':<12} {colored(run.run_id or '?', C.BOLD)}")
    print(f"  {'Dir:':<12} {run.run_dir}")
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


def _mark_relatives_reviewed_after_catalog(
    newly_cataloged_ids: set[str],
    all_runs: list[TrackedRun],
) -> bool:
    """Offer to bulk-mark tracker entries for newly-cataloged relatives as reviewed.

    Called after the `[c]atalog` action returns. The catalog's `_discover_lineage`
    flow may have cataloged ancestors/descendants of the focused run; when any of
    those runs are also in the tracker (and not already REVIEWED), this offers a
    single prompt to mark them all at once — saving the user from walking the
    tracker list and marking each one by hand.

    Returns True if any runs were marked (caller should redisplay list).
    """
    if not newly_cataloged_ids:
        return False

    candidates = [
        r for r in all_runs
        if r.run_id is not None
        and r.run_id in newly_cataloged_ids
        and r.state != RunState.REVIEWED
    ]
    if not candidates:
        return False

    print(
        f"\n  {colored(f'{len(candidates)} newly-cataloged relative(s) are also tracked:', C.CYAN)}"
    )
    for r in candidates:
        desc = r.description or "(no description)"
        print(f"    - {r.run_id}  {trunc(text=desc, max_len=60)}")
    try:
        mark = input(colored("  Mark all as reviewed? [y/N]: ", C.CYAN)).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if mark != "y":
        return False

    now = datetime.now(tz=timezone.utc)
    for r in candidates:
        r.state = RunState.REVIEWED
        r.state_changed_at = now
    save_tracked_runs(runs=all_runs)
    print(colored(f"  Marked {len(candidates)} relative(s) as reviewed.", C.GREEN))
    return True


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
        # Launch catalog interactively, passing note as prefill if present.
        # Snapshot catalog ids before + after to detect runs added via the
        # catalog's lineage-discovery flow — we'll offer to bulk-mark those
        # as reviewed too if they're in the tracker.
        import subprocess
        catalog_path = _get_catalog_path()
        try:
            before_ids = {
                e.run_id for e in load_catalog(catalog_path=catalog_path).entries
            }
        except Exception:
            before_ids = set()

        print(colored("  Launching catalog...", C.CYAN))
        cmd = [sys.executable, "-m", "vfh.catalog", "--path", run.run_dir]
        if run.comments:
            cmd += ["--prefill-description", run.comments]
        subprocess.run(cmd)

        try:
            after_ids = {
                e.run_id for e in load_catalog(catalog_path=catalog_path).entries
            }
        except Exception:
            after_ids = before_ids
        newly_cataloged_relative_ids = (after_ids - before_ids) - {run.run_id}

        # Offer to mark THIS run as reviewed
        root_marked = False
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
            root_marked = True

        # Offer to bulk-mark newly-cataloged relatives (ancestors/descendants)
        # that are also tracked runs and not already reviewed.
        relatives_marked = _mark_relatives_reviewed_after_catalog(
            newly_cataloged_ids=newly_cataloged_relative_ids,
            all_runs=all_runs,
        )

        return root_marked or relatives_marked

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


def _format_entry_date(entry: CatalogEntry) -> str:
    """Format started_at as MM/DD. Shows ??/?? if unknown — no fallback to cataloged_at."""
    if not entry.started_at:
        return colored("??/??", C.DIM)
    try:
        dt = datetime.fromisoformat(entry.started_at)
        return colored(dt.strftime("%m/%d"), C.DIM)
    except (ValueError, TypeError):
        return colored("??/??", C.DIM)


def _format_catalog_line(idx: int, entry: CatalogEntry) -> str:
    """Format a single catalog entry as a display line (no description — shown separately)."""
    parts: list[str] = []
    parts.append(f" [{idx}]")
    parts.append(colored(entry.run_id, C.BOLD))
    parts.append(_format_entry_date(entry=entry))

    # Run name (directory basename) in dim — between wandb id and model
    run_name = Path(entry.run_dir).name
    parts.append(colored(run_name, C.DIM))

    # Model — short name (last path component)
    model_short = entry.base_model.rsplit("/", 1)[-1] if "/" in entry.base_model else entry.base_model
    parts.append(colored(model_short, C.CYAN))

    # Rollout range
    parts.append(_format_catalog_range(rng=entry.rollout_range))

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


def _apply_catalog_filters(entries: list[CatalogEntry], filt: CatalogFilter,
                           hours_filter: float | None = None) -> list[CatalogEntry]:
    """Apply active filters. AND across filter types, OR within models/datasets."""
    result = entries
    if hours_filter is not None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours_filter)
        result = [e for e in result if datetime.fromisoformat(e.cataloged_at) >= cutoff]
    if filt.tags:
        result = [e for e in result if all(t in e.tags for t in filt.tags)]
    if filt.exclude_tags:
        result = [e for e in result if not any(t in e.tags for t in filt.exclude_tags)]
    if filt.models:
        result = [e for e in result if e.base_model in filt.models]
    if filt.datasets:
        result = [e for e in result if e.train_dataset in filt.datasets]
    return result


def _display_catalog(catalog: Catalog, filt: CatalogFilter, show_full_descriptions: bool = False,
                     hours_filter: float | None = None) -> list[CatalogEntry]:
    """Display catalog entries, applying filters. Returns flat display-order list."""
    entries = _apply_catalog_filters(entries=catalog.entries, filt=filt, hours_filter=hours_filter)

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
    if hours_filter is not None:
        active_filters.append(colored(f"hours: {hours_filter:g}", C.YELLOW))
    if active_filters:
        print(f"\n  {colored('Filters:', C.BOLD)} {' + '.join(active_filters)}")

    # Header
    has_filters = filt.tags or filt.exclude_tags or filt.models or filt.datasets or hours_filter is not None
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
    parts.append(_format_entry_date(entry=entry))
    # Run name (directory basename) in dim — between wandb id and model
    run_name = Path(entry.run_dir).name
    parts.append(colored(run_name, C.DIM))
    model_short = entry.base_model.rsplit("/", 1)[-1] if "/" in entry.base_model else entry.base_model
    parts.append(colored(model_short, C.CYAN))
    parts.append(_format_catalog_range(rng=entry.rollout_range))
    tag_str = _format_tags(tags=entry.tags)
    if tag_str:
        parts.append(tag_str)
    return "  ".join(parts)


def _display_catalog_tree(
    catalog: Catalog,
    filt: CatalogFilter,
    show_full_descriptions: bool = False,
    hours_filter: float | None = None,
) -> list[CatalogEntry]:
    """Display catalog as lineage trees. Returns flat display-order list."""
    filtered = _apply_catalog_filters(entries=catalog.entries, filt=filt, hours_filter=hours_filter)
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
    if hours_filter is not None:
        active_filters.append(colored(f"hours: {hours_filter:g}", C.YELLOW))
    if active_filters:
        print(f"\n  {colored('Filters:', C.BOLD)} {' + '.join(active_filters)}")

    has_filters = filt.tags or filt.exclude_tags or filt.models or filt.datasets or hours_filter is not None
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
    print(f"  {'Dir:':<14} {entry.run_dir}")
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
    current = set(entry.tags)
    # We use a mutable list wrapper so _handle_plus_command can append to it
    selected_proxy: list[str] = list(current)
    print(f"\n  Current tags: {_format_tags(tags=entry.tags) or colored('(none)', C.DIM)}")
    print(f"  {colored('Toggle by index. +/Category/Tag to create. Empty to finish.', C.DIM)}")

    changed = False
    display_order: list[CatalogTag] = []
    while True:
        # Display tags grouped by category with toggle state + sequential indices
        display_order = _tags_in_display_order(catalog=catalog)
        tag_to_idx = {tag.name: i for i, tag in enumerate(display_order)}

        cat_names = [c.name for c in catalog.tag_categories]
        by_category: dict[str, list[CatalogTag]] = {c: [] for c in cat_names}
        by_category["Uncategorized"] = []
        for tag in catalog.tags:
            bucket = tag.category if tag.category in by_category else "Uncategorized"
            by_category[bucket].append(tag)

        for cat_name, tags in by_category.items():
            if not tags:
                continue
            print(f"    {colored(cat_name + ':', C.BOLD)}")
            for tag in tags:
                idx = tag_to_idx[tag.name]
                marker = colored("\u2713", C.GREEN) if tag.name in current else " "
                print(f"      [{idx}] {marker} {colored(tag.name, C.MAGENTA)}")

        try:
            raw = input_or_esc(prompt="  Tag (index/+name/empty): ").strip()
        except UserCancelled:
            break
        if not raw:
            break

        if raw.startswith("+"):
            old_len = len(catalog.tags)
            _handle_plus_command(
                query=raw,
                catalog=catalog,
                selected_tags=selected_proxy,
                colored_fn=colored,
                input_fn=input_or_esc,
                cancel_cls=UserCancelled,
            )
            # Sync selected_proxy back to current
            current = set(selected_proxy)
            if len(catalog.tags) != old_len or current != set(entry.tags):
                changed = True
            continue

        try:
            idx = int(raw)
            if 0 <= idx < len(display_order):
                tag_name = display_order[idx].name
                if tag_name in current:
                    current.discard(tag_name)
                    if tag_name in selected_proxy:
                        selected_proxy.remove(tag_name)
                else:
                    current.add(tag_name)
                    if tag_name not in selected_proxy:
                        selected_proxy.append(tag_name)
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
    if not catalog.tags:
        print(colored("  No tags in catalog.", C.DIM))
        return False

    changed = False
    while True:
        # Count entries per tag (respecting current model + dataset filters)
        tag_counts: dict[str, int] = {}
        for tag in catalog.tags:
            count = sum(
                1 for e in catalog.entries
                if tag.name in e.tags
                and (not filt.models or e.base_model in filt.models)
                and (not filt.datasets or e.train_dataset in filt.datasets)
            )
            tag_counts[tag.name] = count

        print(f"\n  {colored('Tags:', C.BOLD)}  {colored('Toggle by index (cycles: off → include → exclude → off). Empty to finish.', C.DIM)}")

        # Group by category with sequential display indices
        display_order = _tags_in_display_order(catalog=catalog)
        tag_to_idx = {tag.name: i for i, tag in enumerate(display_order)}

        cat_names = [c.name for c in catalog.tag_categories]
        by_category: dict[str, list[CatalogTag]] = {c: [] for c in cat_names}
        by_category["Uncategorized"] = []
        for tag in catalog.tags:
            bucket = tag.category if tag.category in by_category else "Uncategorized"
            by_category[bucket].append(tag)

        for cat_name, tags in by_category.items():
            if not tags:
                continue
            print(f"    {colored(cat_name + ':', C.BOLD)}")
            for tag in tags:
                idx = tag_to_idx[tag.name]
                if tag.name in filt.tags:
                    marker = colored("\u2713", C.GREEN)
                    name_str = colored(tag.name, C.MAGENTA, C.BOLD)
                elif tag.name in filt.exclude_tags:
                    marker = colored("\u2717", C.RED)
                    name_str = colored(tag.name, C.RED)
                else:
                    marker = " "
                    name_str = colored(tag.name, C.MAGENTA)
                count_str = colored(f"({tag_counts[tag.name]})", C.DIM)
                print(f"      [{idx}] {marker} {name_str} {count_str}")

        try:
            raw = input_or_esc(prompt="  Tag index: ").strip()
        except UserCancelled:
            break
        if not raw:
            break

        try:
            idx = int(raw)
            if 0 <= idx < len(display_order):
                tag_name = display_order[idx].name
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
                print(colored(f"  Index out of range (0-{len(display_order) - 1})", C.RED))
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
# [v]iew — build a wandb workspace URL for a selection of runs
# ---------------------------------------------------------------------------


@dataclass
class _ResolvedRun:
    """Result of a run_id lookup across tracker + catalog."""
    run_id: str
    ref: WandbRunRef
    source: str      # "tracker" or "catalog"
    label: str       # short human-readable name for the echo line


def _resolve_run_id(
    *,
    run_id: str,
    tracked_runs: list[TrackedRun],
    catalog: Catalog | None,
    catalog_path: Path,
) -> tuple[_ResolvedRun | None, Catalog | None]:
    """Look up ``run_id`` in tracker, then catalog.

    Returns ``(resolved_or_None, catalog)``. The catalog is lazy-loaded on first
    access and returned so callers can cache it between invocations.
    """
    # Tracker first
    for run in tracked_runs:
        if run.run_id and run.run_id == run_id:
            if not (run.wandb_entity and run.wandb_project):
                return None, catalog
            ref = WandbRunRef(
                run_id=run_id,
                entity=run.wandb_entity,
                project=run.wandb_project,
            )
            label = run.description or Path(run.run_dir).name
            return _ResolvedRun(
                run_id=run_id, ref=ref, source="tracker", label=label,
            ), catalog

    # Catalog second (lazy load)
    if catalog is None:
        catalog = load_catalog(catalog_path=catalog_path)
    for entry in catalog.entries:
        if entry.run_id == run_id:
            parsed = parse_wandb_url(url=entry.wandb_url) if entry.wandb_url else None
            if parsed is None:
                return None, catalog
            entity, project, _ = parsed
            ref = WandbRunRef(run_id=run_id, entity=entity, project=project)
            label = entry.description or Path(entry.run_dir).name
            return _ResolvedRun(
                run_id=run_id, ref=ref, source="catalog", label=label,
            ), catalog

    return None, catalog


def _view_action(
    *,
    tracked_runs: list[TrackedRun],
    catalog: Catalog | None,
    catalog_path: Path,
    display_list: list | None = None,
) -> Catalog | None:
    """Interactively assemble a list of runs by run_id or display index and create a wandb view.

    ``display_list`` is the current on-screen list (TrackedRun or CatalogEntry).
    If provided, numeric inputs are resolved as display indices first.

    Returns the (possibly newly-loaded) catalog so the caller can cache it.
    """
    print(colored("  Enter run_ids or display [N] indices (one per line, empty to finish).", C.DIM))
    print(colored("  Each input is searched in: display list → tracker → catalog.", C.DIM))

    selected: list[_ResolvedRun] = []
    seen_ids: set[str] = set()

    while True:
        try:
            raw = input_or_esc(
                prompt=f"  run_id/index ({len(selected)} selected): ",
            ).strip()
        except UserCancelled:
            return catalog
        if not raw:
            break

        # Try as display index first
        run_id = raw
        if display_list is not None and raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(display_list):
                item = display_list[idx]
                run_id = item.run_id
                print(colored(f"    [{idx}] → {run_id}", C.DIM))

        if run_id in seen_ids:
            print(colored(f"    {run_id}: already added", C.YELLOW))
            continue

        resolved, catalog = _resolve_run_id(
            run_id=run_id,
            tracked_runs=tracked_runs,
            catalog=catalog,
            catalog_path=catalog_path,
        )
        if resolved is None:
            print(colored(
                f"    {run_id}: not found in tracker or catalog (or missing wandb info)",
                C.RED,
            ))
            continue

        selected.append(resolved)
        seen_ids.add(run_id)
        print(colored(
            f"    {run_id}: found in {resolved.source} — {resolved.label}",
            C.GREEN,
        ))

    if not selected:
        print(colored("  No runs selected.", C.DIM))
        return catalog

    # Recap
    print(colored(f"\n  Selected {len(selected)} run(s):", C.BOLD))
    for i, r in enumerate(selected, start=1):
        print(
            f"    {i}. {colored(r.run_id, C.BOLD)} "
            f"[{colored(r.source, C.DIM)}] {r.label}"
        )

    # Name prompt
    try:
        name_raw = input_or_esc(prompt="  View name (empty = auto): ").strip()
    except UserCancelled:
        return catalog
    view_name = name_raw or None

    print(colored(
        f"  Creating wandb view for {len(selected)} run(s)...", C.CYAN,
    ))
    try:
        url = create_wandb_view_url(
            refs=[r.ref for r in selected], view_name=view_name,
        )
    except MixedProjectError as exc:
        print(colored(f"  {exc}", C.RED))
        return catalog
    except ImportError as exc:
        print(colored(f"  {exc}", C.RED))
        return catalog
    except Exception as exc:
        print(colored(f"  Failed to create view: {exc}", C.RED))
        return catalog

    copy_and_print_url(url=url, label="W&B view URL")
    return catalog


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
    catalog_filter = _load_catalog_filter()
    catalog_path = _get_catalog_path()
    catalog: Catalog | None = None  # lazy-loaded on first [c]
    show_full_descriptions = True  # catalog: default to full descriptions
    show_notes = False
    hours_filter: float | None = None
    catalog_hours_filter: float | None = None
    tree_view = False              # catalog tree view
    tracker_tree_view = False      # tracker tree view

    # Interactive loop
    while True:
        if viewing_catalog:
            if catalog is None:
                catalog = load_catalog(catalog_path=catalog_path)
                refresh_catalog_lineage(catalog=catalog)
            if tree_view:
                display_catalog = _display_catalog_tree(catalog=catalog, filt=catalog_filter, show_full_descriptions=show_full_descriptions, hours_filter=catalog_hours_filter)
            else:
                display_catalog = _display_catalog(catalog=catalog, filt=catalog_filter, show_full_descriptions=show_full_descriptions, hours_filter=catalog_hours_filter)

            filter_hint = ""
            if catalog_filter.tags or catalog_filter.exclude_tags or catalog_filter.models or catalog_filter.datasets:
                filter_hint = " *"
            desc_label = colored("[d]esc*", C.DIM) if show_full_descriptions else colored("[d]esc", C.DIM)
            tree_label = colored("[t]ree*", C.DIM) if tree_view else colored("[t]ree", C.DIM)
            hours_label = colored(f"[h]ours({int(catalog_hours_filter)})*", C.DIM) if catalog_hours_filter else colored("[h]ours", C.DIM)
            print(f"\n  {colored(f'[#] select  [c]tracked  [r]eload  [q]uit  [f]ilter{filter_hint}  [v]iew  ', C.DIM)}{desc_label}  {tree_label}  {hours_label}")

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
                if _filter_menu(catalog=catalog, filt=catalog_filter):
                    _save_catalog_filter(filt=catalog_filter)
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

            if raw == "h":
                if catalog_hours_filter is not None:
                    catalog_hours_filter = None
                    print(colored("  Hours filter: OFF", C.YELLOW))
                else:
                    try:
                        hrs_input = input_or_esc(prompt="  Show entries cataloged in the last N hours (e.g. 24): ").strip()
                    except UserCancelled:
                        continue
                    try:
                        catalog_hours_filter = float(hrs_input)
                        print(colored(f"  Showing entries from the last {catalog_hours_filter:g} hours.", C.YELLOW))
                    except ValueError:
                        print(colored(f"  Invalid number: {hrs_input}", C.RED))
                continue

            if raw == "v":
                catalog = _view_action(
                    tracked_runs=runs,
                    catalog=catalog,
                    catalog_path=catalog_path,
                    display_list=display_catalog,
                )
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
            # Tracker view (flat grouped-by-state OR lineage tree)
            if tracker_tree_view:
                display_list = _display_runs_tree(runs=runs, config=config, show_notes=show_notes, hours_filter=hours_filter)
            else:
                display_list = _display_runs(runs=runs, config=config, show_notes=show_notes, hours_filter=hours_filter)
            if not display_list and not runs:
                # No runs at all — offer to switch to catalog
                print(colored("  No tracked runs. Press [c] to view catalog.", C.DIM))

            filter_empty_label = colored("[f]ilter empty*", C.DIM) if config.filter_empty else colored("[f]ilter empty", C.DIM)
            all_label = colored("[a]ll (show reviewed)*", C.DIM) if config.show_reviewed else colored("[a]ll (show reviewed)", C.DIM)
            notes_label = colored("[n]otes*", C.DIM) if show_notes else colored("[n]otes", C.DIM)
            hours_label = colored(f"[h]ours({int(hours_filter)})*", C.DIM) if hours_filter else colored("[h]ours", C.DIM)
            tree_label = colored("[t]ree*", C.DIM) if tracker_tree_view else colored("[t]ree", C.DIM)
            print(f"\n  {colored('[#] select  [c]atalog  [r]efresh  [q]uit  ', C.DIM)}{filter_empty_label}  {all_label}  {colored('[v]iew  ', C.DIM)}{notes_label}  {hours_label}  {tree_label}")

            try:
                raw = input_or_esc(prompt="\n> ").strip().lower()
            except UserCancelled:
                break

            if raw in ("q", "quit", ""):
                break

            if raw == "c":
                viewing_catalog = True
                catalog = None  # force reload — tracker's [c]atalog action may have modified catalog.json
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

            if raw == "n":
                show_notes = not show_notes
                label = "ON" if show_notes else "OFF"
                print(colored(f"  Show notes: {label}", C.YELLOW))
                continue

            if raw == "h":
                if hours_filter is not None:
                    hours_filter = None
                    print(colored("  Hours filter: OFF", C.YELLOW))
                else:
                    try:
                        hrs_input = input_or_esc(prompt="  Show runs from the last N hours (e.g. 24): ").strip()
                    except UserCancelled:
                        continue
                    try:
                        hours_filter = float(hrs_input)
                        print(colored(f"  Showing runs from the last {hours_filter:g} hours.", C.YELLOW))
                    except ValueError:
                        print(colored(f"  Invalid number: {hrs_input}", C.RED))
                continue

            if raw == "r":
                runs = load_tracked_runs()
                runs = _do_refresh(runs=runs)
                _clear_lineage_cache()  # child_run_ids on disk may have changed
                continue

            if raw == "t":
                tracker_tree_view = not tracker_tree_view
                label = "ON" if tracker_tree_view else "OFF"
                print(colored(f"  Tree view: {label}", C.YELLOW))
                continue

            if raw == "v":
                catalog = _view_action(
                    tracked_runs=runs,
                    catalog=catalog,
                    catalog_path=catalog_path,
                    display_list=display_list,
                )
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
