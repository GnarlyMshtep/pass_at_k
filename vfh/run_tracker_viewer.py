"""Interactive viewer for tracked VFH training runs.

Usage:
    python -m vfh.run_tracker_viewer              # refresh + show all
    python -m vfh.run_tracker_viewer --no-refresh  # skip wandb polling
    python -m vfh.run_tracker_viewer --show-reviewed
    python -m vfh.run_tracker_viewer --filter-empty # hide runs with 0 checkpoints
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tyro

from vfh.interactive_utils import (
    C,
    UserCancelled,
    colored,
    copy_and_print_path,
    copy_and_print_url,
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

    # Interactive loop
    while True:
        display_list = _display_runs(runs=runs, config=config)
        if not display_list:
            break

        print(f"\n  {colored('[#] select  [r]efresh  [q]uit  [f]ilter empty  [a]ll (show reviewed)', C.DIM)}")

        try:
            raw = input_or_esc(prompt="\n> ").strip().lower()
        except UserCancelled:
            break

        if raw in ("q", "quit", ""):
            break

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
            needs_redisplay = _action_menu(run=display_list[idx], all_runs=runs)
            # Always redisplay after an action (state may have changed)
        else:
            print(colored(f"  Index out of range (0-{len(display_list) - 1})", C.RED))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, UserCancelled):
        print()
