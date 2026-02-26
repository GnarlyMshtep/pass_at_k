"""VFH Orchestrator — main entry point for launching verl training runs.

Usage:
    python -m vfh.orchestrator new --base-config base.json5 --desc "baseline" -y
    python -m vfh.orchestrator new --base-config base.json5 --overrides o.json5 --desc "fork" \\
        --fork-from logs/VerlRun/02/21/run_dir/ --fork-step 320
    python -m vfh.orchestrator continue --run-dir logs/VerlRun/02/21/run_dir/ -y
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import dacite
import pyjson5

from vfh.config_resolver import resolve_config
from vfh.run_manager import (
    create_run_dir,
    create_run_subdirs,
    load_run_metadata,
    register_child_run,
)
from vfh.vfh_types import (
    CheckpointDaemonConfig,
    ContinueRunConfig,
    ForkReason,
    NewRunConfig,
    OrchestratorConfig,
    RunOrigin,
    ValidationMode,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch(
    orch_config: OrchestratorConfig,
    run_config: NewRunConfig | ContinueRunConfig,
    requires_openrouter: bool = False,
    use_dummy_verl: bool = False,
) -> None:
    """Resolve config, create run dir, validate env, and launch verl."""

    if isinstance(run_config, ContinueRunConfig):
        merged_config, hydra_overrides, run_metadata = _prepare_continue(
            orch_config=orch_config,
            run_config=run_config,
        )
    else:
        merged_config, hydra_overrides, run_metadata = _prepare_new(
            orch_config=orch_config,
            run_config=run_config,
        )

    print(f"\n=== VFH Run ===")
    print(f"  run_id:    {run_metadata.run_id}")
    print(f"  run_dir:   {run_metadata.run_dir}")
    print(f"  wandb_url: {run_metadata.wandb_url}")
    print(f"  overrides: {len(hydra_overrides)} Hydra keys")
    print()

    # --- Validation ---
    if orch_config.validation_mode != ValidationMode.SKIP:
        _run_validation(
            merged_config=merged_config,
            checkpoints_path=str(Path(run_metadata.run_dir) / "checkpoints"),
            rollouts_path=str(Path(run_metadata.run_dir) / "rollouts"),
            requires_openrouter=requires_openrouter,
            warn_only=(orch_config.validation_mode == ValidationMode.AUTO_APPROVE),
        )
        if orch_config.validation_mode == ValidationMode.FULL:
            answer = input("Validation passed. Continue? [Y/n] ").strip().lower()
            if answer not in ("", "y", "yes"):
                print("Aborted by user.")
                sys.exit(1)
    else:
        print("Skipping validation (-yy)")

    # --- Create subdirs (after validation, so validate_env doesn't see empty dirs) ---
    create_run_subdirs(run_dir=run_metadata.run_dir)

    # --- Spawn checkpoint daemon(s) BEFORE exec ---
    # os.execvp preserves our PID, so daemons will watch verl correctly.
    for daemon_cfg in orch_config.checkpoint_daemons:
        if not daemon_cfg.enabled:
            continue
        _spawn_checkpoint_daemon(
            run_dir=run_metadata.run_dir,
            verl_pid=os.getpid(),
            daemon_cfg=daemon_cfg,
        )

    # --- Set up output tee (verl output → terminal + file) ---
    verl_log = Path(run_metadata.run_dir) / "verl_output.log"
    _setup_output_tee(log_path=verl_log)

    # --- exec into verl (replaces this process) ---
    verl_module = "vfh.dummy_verl" if use_dummy_verl else "verl.trainer.main_ppo"
    cmd = ["python3", "-m", verl_module] + hydra_overrides
    print(f"\nLaunching (exec): {verl_module}")
    print(f"  Full command ({len(cmd)} args)")
    print(f"  Output tee: {verl_log}")
    sys.stdout.flush()
    os.execvp("python3", cmd)


# ---------------------------------------------------------------------------
# Output tee
# ---------------------------------------------------------------------------


def _setup_output_tee(log_path: Path) -> None:
    """Fork a tee child so stdout/stderr go to both terminal and a log file.

    After this call, the parent's stdout/stderr are redirected to a pipe.
    A forked child reads from the pipe and writes to both the original
    terminal FDs and the log file.  When the parent (verl via os.execvp)
    exits, the pipe closes and the tee child exits automatically.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Save original terminal FDs before we redirect
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)

    r_fd, w_fd = os.pipe()
    tee_pid = os.fork()

    if tee_pid == 0:
        # --- Tee child process ---
        os.close(w_fd)
        log_f = open(log_path, "wb")
        try:
            while True:
                data = os.read(r_fd, 8192)
                if not data:
                    break
                os.write(orig_stdout_fd, data)
                log_f.write(data)
                log_f.flush()
        finally:
            log_f.close()
            os.close(r_fd)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)
        os._exit(0)

    # --- Parent: redirect stdout/stderr to the pipe ---
    os.close(r_fd)
    os.dup2(w_fd, 1)  # stdout → pipe
    os.dup2(w_fd, 2)  # stderr → pipe
    os.close(w_fd)
    os.close(orig_stdout_fd)
    os.close(orig_stderr_fd)


# ---------------------------------------------------------------------------
# Daemon spawning
# ---------------------------------------------------------------------------


def _spawn_checkpoint_daemon(
    run_dir: str,
    verl_pid: int,
    daemon_cfg: CheckpointDaemonConfig,
) -> subprocess.Popen:
    """Spawn the checkpoint daemon as a subprocess."""
    cmd = [
        "python3", "-m", "vfh.checkpoint_daemon",
        "--run-dir", run_dir,
        "--verl-pid", str(verl_pid),
        "--poll-interval", str(daemon_cfg.poll_interval_seconds),
    ]
    if daemon_cfg.clean_after_backup:
        cmd.append("--clean-after-backup")

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Checkpoint daemon PID: {proc.pid}")
    return proc


# ---------------------------------------------------------------------------
# Prepare new run
# ---------------------------------------------------------------------------


def _prepare_new(
    orch_config: OrchestratorConfig,
    run_config: NewRunConfig,
) -> tuple[dict[str, Any], list[str], Any]:
    """Resolve config, create run dir, inject VFH overrides."""
    merged_config, hydra_overrides = resolve_config(
        base_config_path=run_config.base_config_path,
        overrides_path=run_config.overrides_path,
        extra_hydra_overrides=run_config.extra_hydra_overrides,
    )

    project_name = _get_nested(d=merged_config, keys=["trainer", "project_name"])
    experiment_name = _get_nested(
        d=merged_config, keys=["trainer", "experiment_name"], default="vfh_run"
    )

    run_metadata = create_run_dir(
        logs_root=orch_config.logs_root,
        description=run_config.description or experiment_name,
        origin=run_config.origin,
        project_name=project_name,
        base_config_path=run_config.base_config_path,
        overrides_path=run_config.overrides_path,
        resolved_hydra_overrides=hydra_overrides,
    )

    # Inject VFH-managed paths into Hydra overrides
    run_dir = Path(run_metadata.run_dir)
    hydra_overrides.extend([
        f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.rollout_data_dir={run_dir / 'rollouts' / 'train'}",
        f"trainer.validation_data_dir={run_dir / 'rollouts' / 'val'}",
        f"+trainer.wandb_run_id={run_metadata.wandb_run_id}",
    ])

    # If forking, set resume path
    if run_config.origin.fork_reason in (ForkReason.INTENTIONAL_FORK, ForkReason.CONTINUE):
        parent_dir = run_config.origin.parent_run_dir
        parent_step = run_config.origin.parent_checkpoint_step
        if parent_dir and parent_step is not None:
            resume_path = str(
                Path(parent_dir) / "checkpoints" / f"global_step_{parent_step}"
            )
            hydra_overrides.extend([
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={resume_path}",
            ])
            # Register as child in parent metadata
            register_child_run(
                parent_run_dir=parent_dir,
                child_run_id=run_metadata.run_id,
            )

    return merged_config, hydra_overrides, run_metadata


# ---------------------------------------------------------------------------
# Prepare continue (crashed/stopped run)
# ---------------------------------------------------------------------------


def _prepare_continue(
    orch_config: OrchestratorConfig,
    run_config: ContinueRunConfig,
) -> tuple[dict[str, Any], list[str], Any]:
    """Load parent metadata, create new child run, set resume from latest checkpoint."""
    parent_metadata = load_run_metadata(run_dir=run_config.run_dir)
    latest_step = _find_latest_checkpoint_step(
        checkpoints_dir=str(Path(run_config.run_dir) / "checkpoints")
    )
    if latest_step is None:
        raise RuntimeError(
            f"No checkpoints found in {run_config.run_dir}/checkpoints/ — nothing to continue from."
        )

    origin = RunOrigin(
        fork_reason=ForkReason.CONTINUE,
        parent_run_id=parent_metadata.run_id,
        parent_run_dir=parent_metadata.run_dir,
        parent_checkpoint_step=latest_step,
    )

    # Build a NewRunConfig from parent + any new overrides
    new_run_config = NewRunConfig(
        base_config_path=parent_metadata.base_config_path,
        overrides_path=run_config.overrides_path or parent_metadata.overrides_path,
        extra_hydra_overrides=run_config.extra_hydra_overrides,
        description=f"cont_{parent_metadata.description}",
        origin=origin,
    )

    return _prepare_new(orch_config=orch_config, run_config=new_run_config)


def _find_latest_checkpoint_step(checkpoints_dir: str) -> Optional[int]:
    """Find the latest global_step_N in a checkpoints directory."""
    tracker_file = Path(checkpoints_dir) / "latest_checkpointed_iteration.txt"
    if tracker_file.exists():
        content = tracker_file.read_text().strip()
        if content:
            return int(content)

    # Fallback: scan for global_step_N directories
    ckpt_path = Path(checkpoints_dir)
    if not ckpt_path.exists():
        return None
    steps: list[int] = []
    for d in ckpt_path.iterdir():
        if d.is_dir() and d.name.startswith("global_step_"):
            try:
                steps.append(int(d.name.split("_")[-1]))
            except ValueError:
                pass
    return max(steps) if steps else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _run_validation(
    merged_config: dict[str, Any],
    checkpoints_path: str,
    rollouts_path: str,
    requires_openrouter: bool,
    warn_only: bool = False,
) -> None:
    """Build and run validate_env.py CLI args from the merged config.

    Args:
        warn_only: If True (-y mode), log validation failures as warnings
            instead of raising. If False (default/FULL), raise on failure.
    """
    get = lambda *keys, **kw: _get_nested(d=merged_config, keys=list(keys), **kw)

    # Extract train/test paths — could be list or string
    train_files = get("data", "train_files")
    test_files = get("data", "val_files")
    train_path = train_files[0] if isinstance(train_files, list) else train_files
    test_path = test_files[0] if isinstance(test_files, list) else test_files

    cmd: list[str] = [
        "python3", "validate_env.py",
        "--train-path", str(train_path),
        "--test-path", str(test_path),
        "--model-path", str(get("actor_rollout_ref", "model", "path")),
        "--reward-path", str(get("custom_reward_function", "path")),
        "--reward-name", str(get("custom_reward_function", "name")),
        "--n-gpu", str(get("trainer", "n_gpus_per_node")),
        "--batch-size", str(get("data", "train_batch_size")),
        "--mini-batch-size", str(get("actor_rollout_ref", "actor", "ppo_mini_batch_size")),
        "--n-rollout", str(get("actor_rollout_ref", "rollout", "n")),
        "--micro-batch-size-per-gpu", str(get("actor_rollout_ref", "actor", "ppo_micro_batch_size_per_gpu")),
        "--adv-estimator", str(get("algorithm", "adv_estimator")),
        "--norm-by-std", str(get("algorithm", "norm_adv_by_std_in_grpo")),
        "--proj-name", str(get("trainer", "project_name")),
        "--exp-name", str(get("trainer", "experiment_name", default="vfh_run")),
        "--linear-warmup-steps", str(get("actor_rollout_ref", "actor", "optim", "lr_warmup_steps")),
        "--checkpoints-path", checkpoints_path,
        "--rollouts-path", rollouts_path,
    ]

    # Optional: CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        cmd.extend(["--cuda-visible-devices", cuda_devices])

    # Optional flags
    if requires_openrouter:
        cmd.append("--requires-openrouter")

    # Optional: reward_kwargs
    reward_kwargs = get("custom_reward_function", "reward_kwargs", default=None)
    if reward_kwargs is not None:
        cmd.extend(["--reward-kwargs", json.dumps(reward_kwargs)])

    print("Running validation...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        if warn_only:
            print(f"\033[1;33m⚠ WARNING: Validation failed (exit code {result.returncode}) "
                  f"— continuing anyway (-y mode)\033[0m")
        else:
            raise RuntimeError(f"validate_env.py failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(
    d: dict[str, Any],
    keys: list[str],
    default: Any = KeyError,
) -> Any:
    """Get a nested value from a dict using a list of keys."""
    current = d
    for k in keys:
        # Strip leading "+" from keys (JSON5 new-field markers)
        bare_k = k.lstrip("+")
        if isinstance(current, dict):
            # Try both bare and +-prefixed keys
            if bare_k in current:
                current = current[bare_k]
            elif f"+{bare_k}" in current:
                current = current[f"+{bare_k}"]
            elif default is not KeyError:
                return default
            else:
                raise KeyError(f"Key '{bare_k}' not found at path {'.'.join(keys)}")
        else:
            if default is not KeyError:
                return default
            raise KeyError(f"Cannot descend into non-dict at path {'.'.join(keys)}")
    return current


# ---------------------------------------------------------------------------
# Orchestrator config loading (abstracted for future flexibility)
# ---------------------------------------------------------------------------


def load_orchestrator_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> OrchestratorConfig:
    """Load orchestrator config from JSON5 file + CLI overrides.

    Currently supports JSON5 file with CLI overrides on top.
    Loading logic is isolated here so the source format can change later.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        with open(config_path) as f:
            data = pyjson5.load(f)

    if cli_overrides:
        data.update(cli_overrides)

    if not data:
        return OrchestratorConfig()

    return dacite.from_dict(
        data_class=OrchestratorConfig,
        data=data,
        config=dacite.Config(cast=[ValidationMode]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vfh.orchestrator",
        description="""
VFH (Verl For Humans) — orchestrate verl training runs.

Config system:
  Every run is driven by two JSON5 files:
    --base-config   "template" overrides always applied (e.g. batch sizes,
                    architecture flags, data paths, save freq). Keep model
                    and reward OUT of here — put those in --overrides.
    --overrides     Run-specific changes layered on top (model path, reward
                    function, experiment name, etc.).  Optional.

  Both files are deep-merged and flattened into Hydra CLI overrides that
  are passed to verl.trainer.main_ppo.  Keys prefixed with "+" in the JSON5
  (e.g. "+rollout_dump_freq") become "+key=value" Hydra overrides for fields
  that don't exist in the base ppo_trainer.yaml schema.

Run directory:
  Each launch creates a new directory:
    logs/VerlRun/{MM}/{DD}/{desc}_{HH}_{MM}_{run_id}/
  containing checkpoints/, rollouts/train/, rollouts/val/,
  run_metadata.json5, and daemon_logs/.

Checkpoint daemon:
  A background process watches the verl PID.  It runs dvc add + dvc push
  periodically, confirms backup via dvc status --cloud, and optionally
  removes checkpoint contents after backup (--clean-after-backup in orch
  config).  It does a final backup pass when verl exits, then self-terminates.

Environment:
  CUDA_VISIBLE_DEVICES   which GPUs to use (required for multi-GPU)
  WANDB_ENTITY           your W&B entity (required)

Examples:
  # Fresh run
  CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_ENTITY=myorg \\
      python -m vfh.orchestrator new \\
          --base-config vfh/configs/default_APPS_code.json5 \\
          --overrides   vfh/configs/benign_qwen3_8b.json5 \\
          --desc        "benign_baseline" -y

  # Continue crashed run (auto-finds latest checkpoint)
  python -m vfh.orchestrator continue \\
      --run-dir logs/VerlRun/02/21/benign_baseline_14_30_a1b2c3d4/ -y

  # Fork from a specific checkpoint
  python -m vfh.orchestrator new \\
      --base-config vfh/configs/default_APPS_code.json5 \\
      --overrides   vfh/configs/my_fork.json5 \\
      --fork-from   logs/VerlRun/02/21/benign_baseline_14_30_a1b2c3d4/ \\
      --fork-step   320 --desc "fork_new_reward"

  # Dry-run with dummy verl (no GPUs, fast)
  python -m vfh.orchestrator new --base-config ... --dummy -yy \\
      --extra-overrides "+dummy.total_steps=6" "+dummy.sleep_per_step=0.1"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- new ---
    new_parser = subparsers.add_parser(
        "new",
        help="Start a new training run (or fork from an existing one)",
        description="""
Start a new verl training run.

Config resolution order:
  1. base-config JSON5  (always applied)
  2. overrides JSON5    (layered on top, optional)
  3. --extra-overrides  (raw Hydra strings, appended last)

The orchestrator then injects:
  trainer.default_local_dir  → run_dir/checkpoints/
  trainer.rollout_data_dir   → run_dir/rollouts/train/
  trainer.validation_data_dir→ run_dir/rollouts/val/
  +trainer.wandb_run_id      → pre-generated 8-char ID (same as run dir suffix)

Forking:
  --fork-from sets resume_mode=resume_path pointing at the specified
  checkpoint step (or the latest if --fork-step is omitted).  A new run
  directory is created and linked to the parent in run_metadata.json5.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    new_parser.add_argument(
        "--base-config", required=True, metavar="PATH",
        help="Path to base JSON5 config (batch sizes, arch flags, data, save freq, etc.)",
    )
    new_parser.add_argument(
        "--overrides", default=None, metavar="PATH",
        help="Path to run-specific JSON5 overrides (model, reward, experiment name, etc.)",
    )
    new_parser.add_argument(
        "--desc", default="", metavar="TEXT",
        help="Short description used as the run directory prefix (e.g. 'benign_baseline')",
    )
    new_parser.add_argument(
        "--fork-from", default=None, metavar="RUN_DIR",
        help="Run dir to fork from.  Creates child run with resume_mode=resume_path.",
    )
    new_parser.add_argument(
        "--fork-step", type=int, default=None, metavar="N",
        help="global_step_N to fork from (default: latest checkpoint in --fork-from)",
    )
    new_parser.add_argument(
        "--requires-openrouter", action="store_true",
        help="Pass --requires-openrouter to validate_env.py (needed for model-based rewards)",
    )
    new_parser.add_argument(
        "--dummy", action="store_true",
        help="Use vfh.dummy_verl instead of real verl (no GPUs needed, fast)",
    )
    new_parser.add_argument(
        "--extra-overrides", nargs="*", default=[], metavar="KEY=VAL",
        help="Extra raw Hydra overrides appended after JSON5 resolution "
             "(e.g. trainer.total_epochs=50  or  +dummy.total_steps=6)",
    )

    # --- continue ---
    cont_parser = subparsers.add_parser(
        "continue",
        help="Continue a stopped or crashed run from its latest checkpoint",
        description="""
Continue a stopped or crashed verl run.

Finds the latest checkpoint in run-dir/checkpoints/ (via
latest_checkpointed_iteration.txt or by scanning global_step_N dirs),
creates a NEW run directory linked to the original as a child, and launches
verl with resume_mode=resume_path.

Optionally apply new overrides on top of the original run's config.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cont_parser.add_argument(
        "--run-dir", required=True, metavar="PATH",
        help="Path to the run directory to continue (must contain run_metadata.json5)",
    )
    cont_parser.add_argument(
        "--overrides", default=None, metavar="PATH",
        help="Optional JSON5 overrides to apply on top of the original config",
    )
    cont_parser.add_argument(
        "--requires-openrouter", action="store_true",
        help="Pass --requires-openrouter to validate_env.py",
    )
    cont_parser.add_argument(
        "--dummy", action="store_true",
        help="Use vfh.dummy_verl instead of real verl",
    )
    cont_parser.add_argument(
        "--extra-overrides", nargs="*", default=[], metavar="KEY=VAL",
        help="Extra raw Hydra overrides appended last",
    )

    # --- shared flags ---
    for p in [new_parser, cont_parser]:
        p.add_argument(
            "--orch-config", default=None, metavar="PATH",
            help="Path to orchestrator JSON5 config (checkpoint daemon settings, logs_root, etc.). "
                 "Defaults to built-in defaults if omitted.",
        )
        p.add_argument(
            "-y", action="store_true",
            help="Auto-approve after validation (skip the 'Continue? [Y/n]' prompt)",
        )
        p.add_argument(
            "-yy", action="store_true",
            help="Skip validation entirely (don't run validate_env.py)",
        )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Determine validation mode
    if args.yy:
        val_mode = ValidationMode.SKIP
    elif args.y:
        val_mode = ValidationMode.AUTO_APPROVE
    else:
        val_mode = ValidationMode.FULL

    orch_config = load_orchestrator_config(
        config_path=args.orch_config,
        cli_overrides={"validation_mode": val_mode.value},
    )

    if args.command == "new":
        origin = RunOrigin(fork_reason=ForkReason.ROOT)
        if args.fork_from:
            parent_meta = load_run_metadata(run_dir=args.fork_from)
            fork_step = args.fork_step
            if fork_step is None:
                raise ValueError(
                    "--fork-step is required when using --fork-from. "
                    "Specify the global_step_N to fork from explicitly."
                )
            origin = RunOrigin(
                fork_reason=ForkReason.INTENTIONAL_FORK,
                parent_run_id=parent_meta.run_id,
                parent_run_dir=parent_meta.run_dir,
                parent_checkpoint_step=fork_step,
            )

        run_config = NewRunConfig(
            base_config_path=args.base_config,
            overrides_path=args.overrides,
            extra_hydra_overrides=args.extra_overrides,
            description=args.desc,
            origin=origin,
        )

    elif args.command == "continue":
        run_config = ContinueRunConfig(
            run_dir=args.run_dir,
            overrides_path=args.overrides,
            extra_hydra_overrides=args.extra_overrides,
        )
    else:
        parser.error(f"Unknown command: {args.command}")

    launch(
        orch_config=orch_config,
        run_config=run_config,
        requires_openrouter=getattr(args, "requires_openrouter", False),
        use_dummy_verl=args.dummy,
    )


if __name__ == "__main__":
    main()
