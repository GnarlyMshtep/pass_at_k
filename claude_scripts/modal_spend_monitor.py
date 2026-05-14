"""Modal spend monitor.

Polls `modal billing report` and triggers a STOP signal if cumulative spend
exceeds a threshold. Writes a sentinel file the main workflow can check.

Usage:
    # one-shot check:
    python claude_scripts/modal_spend_monitor.py --threshold 200 --range "this month"

    # polling loop (every 60s), writes stop file if exceeded:
    python claude_scripts/modal_spend_monitor.py --threshold 200 --poll 60

    # test mode: use a tiny threshold to verify the trigger works
    python claude_scripts/modal_spend_monitor.py --threshold 0.25 --poll 10 --max-polls 20

Exit codes:
    0 = under threshold
    42 = threshold exceeded (the "STOP" signal)

Stop file: tmp/modal_stop_signal.txt — written when threshold exceeded.
Any other process can `test -f tmp/modal_stop_signal.txt` to check.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

STOP_FILE = Path(__file__).resolve().parent.parent / "tmp" / "modal_stop_signal.txt"
LOG_FILE = Path(__file__).resolve().parent.parent / "tmp" / "modal_spend_log.jsonl"


@dataclass
class Config:
    threshold: float = 200.0
    """USD threshold — trigger STOP if cumulative spend >= this."""

    range: str = "this month"
    """Billing range: 'today', 'this month', 'last month', etc."""

    poll: int = 0
    """Poll interval in seconds. 0 = one-shot."""

    max_polls: int = 0
    """Max polls before exiting (0 = forever)."""

    modal_bin: str = ""
    """Path to modal CLI. Auto-detected if empty."""

    kill_apps: bool = False
    """If set, actively `modal app stop` all ephemeral+running apps when threshold exceeded."""

    watch_app: str = ""
    """If set, also watch this app ID. If it transitions to 'stopped' unexpectedly, write a crash file."""


def get_modal_bin(override: str) -> str:
    if override:
        return override
    p = shutil.which("modal")
    if p:
        return p
    home_local = Path.home() / ".local/bin/modal"
    if home_local.exists():
        return str(home_local)
    raise RuntimeError("Could not find `modal` CLI.")


def query_spend(modal_bin: str, date_range: str) -> float:
    """Returns total USD spend for the given range."""
    result = subprocess.run(
        [modal_bin, "billing", "report", "--for", date_range, "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    entries = json.loads(result.stdout)
    total = sum(float(e["Cost"]) for e in entries)
    return total


def write_stop(total: float, threshold: float, date_range: str) -> None:
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(
        f"STOP\ntotal={total:.4f}\nthreshold={threshold:.2f}\nrange={date_range}\n"
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )


def append_log(total: float, threshold: float) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total": total,
                    "threshold": threshold,
                    "exceeded": total >= threshold,
                }
            )
            + "\n"
        )


def _check_watched_app(modal_bin: str, app_id: str) -> None:
    """If the watched app is stopped, write a crash file."""
    try:
        result = subprocess.run(
            [modal_bin, "app", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        for a in json.loads(result.stdout):
            if a.get("App ID") == app_id:
                state = a.get("State", "")
                if state == "stopped":
                    crash_file = STOP_FILE.parent / f"modal_app_{app_id}_stopped.txt"
                    crash_file.write_text(
                        f"APP STOPPED\napp_id={app_id}\nstate={state}\n"
                        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    print(f"🚨 Watched app {app_id} is STOPPED", flush=True)
                return
    except Exception as e:
        print(f"[watch_app] error: {e}", flush=True)


def check_once(cfg: Config, modal_bin: str) -> bool:
    """Returns True if threshold exceeded."""
    total = query_spend(modal_bin, cfg.range)
    append_log(total, cfg.threshold)
    exceeded = total >= cfg.threshold
    if cfg.watch_app:
        _check_watched_app(modal_bin, cfg.watch_app)
    ts = time.strftime("%H:%M:%S")
    status = "🚨 EXCEEDED" if exceeded else "OK"
    print(
        f"[{ts}] Modal spend ({cfg.range}): ${total:.4f} / ${cfg.threshold:.2f}  {status}",
        flush=True,
    )
    if exceeded:
        write_stop(total, cfg.threshold, cfg.range)
        print(f"Wrote stop signal to {STOP_FILE}", flush=True)
        if cfg.kill_apps:
            _kill_running_apps(modal_bin)
    return exceeded


def _kill_running_apps(modal_bin: str) -> None:
    """List running/ephemeral apps and stop them."""
    try:
        result = subprocess.run(
            [modal_bin, "app", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        apps = json.loads(result.stdout)
        for a in apps:
            state = a.get("State", "")
            if "ephemeral" in state or state == "running":
                app_id = a["App ID"]
                print(f"🚨 Stopping app {app_id} ({state})", flush=True)
                subprocess.run([modal_bin, "app", "stop", app_id], check=False)
    except Exception as e:
        print(f"[kill_apps] error: {e}", flush=True)


def main(cfg: Config) -> None:
    modal_bin = get_modal_bin(cfg.modal_bin)
    # Clear stale stop file at the start of a fresh run
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    if cfg.poll <= 0:
        exceeded = check_once(cfg, modal_bin)
        sys.exit(42 if exceeded else 0)

    polls = 0
    while True:
        exceeded = check_once(cfg, modal_bin)
        if exceeded:
            sys.exit(42)
        polls += 1
        if cfg.max_polls and polls >= cfg.max_polls:
            print(f"Hit max_polls={cfg.max_polls}, exiting without trigger.", flush=True)
            sys.exit(0)
        time.sleep(cfg.poll)


if __name__ == "__main__":
    main(tyro.cli(Config))
