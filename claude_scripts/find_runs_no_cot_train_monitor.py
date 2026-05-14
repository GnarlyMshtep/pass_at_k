"""Filter tracked runs for:
  - base_model == gpt-oss-120b
  - renderer_name == gpt_oss_no_analysis ("no_cot_renderer")
  - at least one TRAIN phase has skip_monitor=false
  - len(rollout_steps) >= 2
"""
from __future__ import annotations

import json
from pathlib import Path

TRACKED_RUNS = Path("/shared/matan/code/pass_at_k/logs/VerlRun/tracked_runs.jsonl")


def load_tinker_config(run_dir: str) -> dict | None:
    cfg_path = Path(run_dir) / "config.json"
    if not cfg_path.exists():
        return None
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return None


def train_has_skip_monitor_false(cfg: dict) -> bool:
    """True if ANY train phase has skip_monitor=false."""
    db = cfg.get("dataset_builder", {})
    phases_json = db.get("phases_json")
    if not phases_json:
        return False
    try:
        phases = json.loads(phases_json)
    except Exception:
        return False
    for ph in phases:
        rk = ph.get("reward_kwargs", {}) or {}
        rc = rk.get("reward_config", {}) or {}
        if rc.get("skip_monitor") is False:
            return True
    return False


def main() -> None:
    matches: list[dict] = []

    with open(TRACKED_RUNS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)

            # 1. base model
            if run.get("base_model") != "gpt-oss-120b":
                continue

            # 2. rollout steps >= 2
            rollout_steps = run.get("rollout_steps") or []
            if len(rollout_steps) < 2:
                continue

            # Open config
            cfg = load_tinker_config(run["run_dir"])
            if cfg is None:
                continue

            # 3. no_cot_renderer
            if cfg.get("renderer_name") != "gpt_oss_no_analysis":
                continue

            # 4. train skip_monitor=false (at least one phase)
            if not train_has_skip_monitor_false(cfg):
                continue

            matches.append(run)

    print(f"Matched {len(matches)} runs:\n")
    for r in matches:
        n_rollouts = len(r.get("rollout_steps") or [])
        print(f"  {r['run_id']:>10}  {r.get('state'):<16}  rollouts={n_rollouts:<4}  {r['description']}")
        print(f"             wandb: {r.get('wandb_url')}")
        print(f"             dir:   {r['run_dir']}")
        print()


if __name__ == "__main__":
    main()
