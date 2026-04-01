"""VFH Orchestrator — main entry point for launching verl training runs.

Usage:
    python -m vfh.orchestrator new --base-config base.json5 --desc "baseline" -y
    python -m vfh.orchestrator new --base-config base.json5 --overrides o.json5 --desc "fork" \\
        --fork-from logs/VerlRun/02/21/run_dir/ --fork-step 320
    python -m vfh.orchestrator continue --run-dir logs/VerlRun/02/21/run_dir/ -y

Sbatch mode (generate SLURM script instead of launching directly):
    python -m vfh.orchestrator new --base-config base.json5 --desc "baseline" \\
        --sbatch --time 08:00:00 -y
    python -m vfh.orchestrator new --base-config base.json5 --sbatch --time 08:00:00 \\
        --dont-auto-sbatch -y

Run a previously prepared sbatch run:
    python -m vfh.orchestrator run-prepared --run-dir logs/VerlRun/02/28/run_dir/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
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
    validate_run_metadata,
    write_run_metadata,
)
from vfh.vfh_types import (
    CheckpointDaemonConfig,
    ContinueRunConfig,
    ForkReason,
    NewRunConfig,
    OrchestratorConfig,
    PreparedRun,
    RunOrigin,
    ValidationMode,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def prepare(
    orch_config: OrchestratorConfig,
    run_config: NewRunConfig | ContinueRunConfig,
    requires_openrouter: bool = False,
) -> PreparedRun:
    """Phase 1: config resolution, run dir creation, validation.

    Returns everything needed for exec_prepared() or sbatch generation.
    No process-level side effects (no fork, no exec, no daemon spawn).
    """
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
        resume_from = _extract_override_value(
            overrides=hydra_overrides, key="trainer.resume_from_path",
        )
        _run_validation(
            merged_config=merged_config,
            checkpoints_path=str(Path(run_metadata.run_dir) / "checkpoints"),
            rollouts_path=str(Path(run_metadata.run_dir) / "rollouts"),
            requires_openrouter=requires_openrouter,
            warn_only=(orch_config.validation_mode == ValidationMode.AUTO_APPROVE),
            resume_from_path=resume_from,
        )
        if orch_config.validation_mode == ValidationMode.FULL:
            answer = input("Validation passed. Continue? [Y/n] ").strip().lower()
            if answer not in ("", "y", "yes"):
                print("Aborted by user.")
                sys.exit(1)
    else:
        print("Skipping validation (-yy)")

    return PreparedRun(
        merged_config=merged_config,
        hydra_overrides=hydra_overrides,
        run_metadata=run_metadata,
    )


def exec_prepared(
    prepared: PreparedRun,
    orch_config: OrchestratorConfig,
    use_dummy_verl: bool = False,
) -> None:
    """Phase 2: create subdirs, spawn daemons, setup tee, execvp into verl.

    This function never returns (os.execvp replaces the process).
    """
    run_dir = prepared.run_metadata.run_dir

    # --- Unset ROCR_VISIBLE_DEVICES (SLURM sets it, conflicts with CUDA_VISIBLE_DEVICES) ---
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    # --- Disable uvloop in Ray workers (causes transport cleanup crashes with httpx) ---
    os.environ["RAY_USE_UVLOOP"] = "0"

    # --- Create subdirs (after validation, so validate_env doesn't see empty dirs) ---
    create_run_subdirs(run_dir=run_dir)

    # --- Spawn checkpoint daemon(s) BEFORE exec ---
    # os.execvp preserves our PID, so daemons will watch verl correctly.
    for daemon_cfg in orch_config.checkpoint_daemons:
        if not daemon_cfg.enabled:
            continue
        _spawn_checkpoint_daemon(
            run_dir=run_dir,
            verl_pid=os.getpid(),
            daemon_cfg=daemon_cfg,
        )

    # --- Set up output tee (verl output → terminal + file) ---
    verl_log = Path(run_dir) / "verl_output.log"
    _setup_output_tee(log_path=verl_log)

    # --- exec into verl (replaces this process) ---
    verl_module = "vfh.dummy_verl" if use_dummy_verl else "verl.trainer.main_ppo"
    cmd = ["python3", "-m", verl_module] + prepared.hydra_overrides
    print(f"\nLaunching (exec): {verl_module}")
    print(f"  Full command ({len(cmd)} args)")
    print(f"  Output tee: {verl_log}")
    sys.stdout.flush()
    os.execvp("python3", cmd)


def launch(
    orch_config: OrchestratorConfig,
    run_config: NewRunConfig | ContinueRunConfig,
    requires_openrouter: bool = False,
    use_dummy_verl: bool = False,
) -> None:
    """Resolve config, create run dir, validate env, and launch verl.

    Thin wrapper around prepare() + exec_prepared().
    """
    prepared = prepare(
        orch_config=orch_config,
        run_config=run_config,
        requires_openrouter=requires_openrouter,
    )
    exec_prepared(
        prepared=prepared,
        orch_config=orch_config,
        use_dummy_verl=use_dummy_verl,
    )


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

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
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

    # wandb group: use root ancestor's run_id so all continuations share a group
    wandb_group = _find_root_run_id(origin=run_config.origin)
    if wandb_group == "__SELF__":
        wandb_group = run_metadata.wandb_run_id

    hydra_overrides.extend([
        f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.rollout_data_dir={run_dir / 'rollouts' / 'train'}",
        f"trainer.validation_data_dir={run_dir / 'rollouts' / 'val'}",
        f"+trainer.wandb_run_id={run_metadata.wandb_run_id}",
        f"+trainer.wandb_group={wandb_group}",
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

            # Validate world_size matches n_gpus_per_node.
            # Use hydra_overrides (not merged_config) since --extra-overrides
            # can change n_gpus_per_node without updating the merged dict.
            n_gpus_str = _extract_override_value(
                overrides=hydra_overrides, key="trainer.n_gpus_per_node",
            )
            n_gpus = int(n_gpus_str) if n_gpus_str else int(
                _get_nested(d=merged_config, keys=["trainer", "n_gpus_per_node"])
            )
            _validate_resume_checkpoint(
                ckpt_dir=Path(resume_path),
                n_gpus=n_gpus,
            )

            # Detect dataset change: compare resolved data.train_files between
            # parent and child. If different, the saved dataloader sampler state
            # won't match the new dataset → tell verl to skip restoring it.
            parent_metadata = load_run_metadata(run_dir=parent_dir)
            parent_train = _extract_override_value(
                overrides=parent_metadata.resolved_hydra_overrides,
                key="data.train_files",
            )
            child_train = _extract_override_value(
                overrides=hydra_overrides,
                key="data.train_files",
            )
            if parent_train and child_train and parent_train != child_train:
                hydra_overrides.append("+trainer.restart_dataloader=true")
                print(
                    f"\n  \033[1;34m⚠️ [VFH] Dataset change detected during fork:\033[0m\n"
                    f"    parent data.train_files: {parent_train}\n"
                    f"    child  data.train_files: {child_train}\n"
                    f"    \033[1;34m→ injecting +trainer.restart_dataloader=true\033[0m\n"
                    f"    Dataloader will fast-forward to fork step (from checkpoint).\n"
                    f"    Use +trainer.override_dataloader_start_step=N to override.\n"
                )
            elif not parent_train:
                print(
                    f"\n  \033[1;33m⚠️ [VFH] Warning: could not find data.train_files in parent's "
                    f"resolved overrides — cannot auto-detect dataset change.\033[0m\n"
                    f"    If you changed the dataset, add --extra-overrides "
                    f"'+trainer.restart_dataloader=true' manually.\n"
                )
            # Register as child in parent metadata
            register_child_run(
                parent_run_dir=parent_dir,
                child_run_id=run_metadata.run_id,
            )

    # Write metadata to disk now that ALL overrides are finalized
    write_run_metadata(run_dir=run_metadata.run_dir, metadata=run_metadata)

    # Sanity-check: catch bugs at prepare-time, not just run-time
    validate_run_metadata(metadata=run_metadata)

    return merged_config, hydra_overrides, run_metadata


def _find_root_run_id(origin: RunOrigin) -> str:
    """Walk up the parent chain to find the root ancestor's run_id.

    Used to set wandb group so all runs in a lineage share one group.
    Returns the root's run_id, or the parent's run_id if the chain
    can't be fully traversed (missing metadata on disk).
    """
    if origin.fork_reason == ForkReason.ROOT or origin.parent_run_dir is None:
        # This IS the root — caller should use the current run's own ID.
        # But we don't have it here, so return a sentinel that the caller
        # replaces with the run's own wandb_run_id.
        return "__SELF__"

    # Walk up
    current_dir = origin.parent_run_dir
    max_depth = 20
    for _ in range(max_depth):
        try:
            parent_meta = load_run_metadata(run_dir=current_dir)
        except Exception:
            break
        if parent_meta.origin.fork_reason == ForkReason.ROOT:
            return parent_meta.run_id
        if parent_meta.origin.parent_run_dir is None:
            return parent_meta.run_id
        current_dir = parent_meta.origin.parent_run_dir

    # Couldn't reach root — use the immediate parent
    try:
        return load_run_metadata(run_dir=origin.parent_run_dir).run_id
    except Exception:
        return "__SELF__"


def _save_code_diff(run_dir: Path) -> None:
    """Save a git diff (HEAD vs working tree) to code.diff in the run directory.

    Captures uncommitted changes so the exact code state is reproducible.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        diff = result.stdout
        if diff:
            (run_dir / "code.diff").write_text(diff)
    except Exception:
        pass  # best-effort — don't block run launch


# ---------------------------------------------------------------------------
# Checkpoint validation (continue / fork)
# ---------------------------------------------------------------------------


def _validate_resume_checkpoint(
    ckpt_dir: Path,
    n_gpus: int | None = None,
) -> None:
    """Validate that a checkpoint directory is ready for resume/fork.

    Checks (in order):
    1. Directory exists on disk (not just a .dvc pointer).
    2. actor/ subdirectory exists.
    3. actor/ contains model weight files (model_world_size_N_rank_*.pt).
    4. If n_gpus is provided, checkpoint world_size matches n_gpus.

    Raises ValueError with a clear message (including DVC pull hints) on failure.
    """
    dvc_file = ckpt_dir.parent / f"{ckpt_dir.name}.dvc"
    dvc_hint = (
        f"\n  Checkpoint may be in DVC — try:\n"
        f"    dvc pull {dvc_file}"
    ) if dvc_file.exists() else ""

    # 1. Directory exists
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise ValueError(
            f"Checkpoint directory does not exist: {ckpt_dir}{dvc_hint}"
        )

    # 2. actor/ subdirectory exists
    actor_dir = ckpt_dir / "actor"
    if not actor_dir.exists() or not actor_dir.is_dir():
        raise ValueError(
            f"Checkpoint has no actor/ subdirectory: {ckpt_dir}{dvc_hint}"
        )

    # 3. actor/ has weight files
    weight_pattern = re.compile(r"model_world_size_(\d+)_rank_\d+\.pt")
    world_sizes: list[int] = []
    for f in actor_dir.iterdir():
        m = weight_pattern.match(f.name)
        if m:
            world_sizes.append(int(m.group(1)))

    if not world_sizes:
        raise ValueError(
            f"Checkpoint actor/ has no model weight files "
            f"(expected model_world_size_N_rank_*.pt): {actor_dir}{dvc_hint}"
        )

    ckpt_world_size = world_sizes[0]
    n_ranks = len(world_sizes)

    # 4. World size matches n_gpus
    if n_gpus is not None:
        if ckpt_world_size != n_gpus:
            raise ValueError(
                f"Checkpoint world_size ({ckpt_world_size}) does not match "
                f"n_gpus_per_node ({n_gpus}). FSDP checkpoint loading will fail.\n"
                f"  Either change n_gpus_per_node to {ckpt_world_size} or use a "
                f"checkpoint saved with world_size={n_gpus}.\n"
                f"  Checkpoint: {ckpt_dir}"
            )
    # Success — no output here; visible validation output comes from validate_env.py


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

    # Validate checkpoint files exist (not just DVC pointers).
    # World_size check is deferred to _prepare_new where we have the resolved config.
    ckpt_dir = Path(run_config.run_dir) / "checkpoints" / f"global_step_{latest_step}"
    _validate_resume_checkpoint(ckpt_dir=ckpt_dir)

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
    resume_from_path: Optional[str] = None,
) -> None:
    """Build and run validate_env.py CLI args from the merged config.

    Args:
        warn_only: If True (-y mode), log validation failures as warnings
            instead of raising. If False (default/FULL), raise on failure.
        resume_from_path: If set, validate the checkpoint dir for resume
            (actor/ exists, weight files present, world_size matches n_gpu).
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

    # Pass full merged config for dataset requirements checking
    cmd.extend(["--merged-config-json", json.dumps(merged_config)])

    # Resume checkpoint validation (continue/fork)
    if resume_from_path:
        cmd.extend(["--resume-from-path", resume_from_path])

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


def _extract_override_value(overrides: list[str], key: str) -> Optional[str]:
    """Extract the value for a given key from a list of Hydra overrides.

    Searches for 'key=value' or '+key=value'. Returns the last match (since
    later overrides take precedence), or None if not found.
    """
    result: Optional[str] = None
    for entry in overrides:
        if "=" not in entry:
            continue
        k, v = entry.split("=", 1)
        # Strip leading "+" from key for comparison
        if k.lstrip("+") == key.lstrip("+"):
            result = v
    return result


def _hydra_overrides_to_nested_dict(overrides: list[str]) -> dict[str, Any]:
    """Parse Hydra override strings ('a.b.c=val') back into a nested dict.

    Values are kept as strings (callers use _get_nested which returns Any).
    Handles +key=val (new-field) syntax by stripping the leading +.
    """
    result: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        key = key.lstrip("+")
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Try to parse as int/float/bool for common cases
        if value.isdigit():
            current[parts[-1]] = int(value)
        elif value.lower() in ("true", "false"):
            current[parts[-1]] = value.lower() == "true"
        else:
            try:
                current[parts[-1]] = float(value)
            except ValueError:
                current[parts[-1]] = value
    return result


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
# Sbatch script generation
# ---------------------------------------------------------------------------


_SLURM_NODE_MAP: dict[int, str] = {
    1: "bleak-mushroom-dove",
    2: "better-ginkgo-dragonfly",
}
_SLURM_NODE_CPUS = 160
_SLURM_NODE_MEM_MB = 1_548_120


def _generate_sbatch(
    prepared: PreparedRun,
    time_limit: str,
    orch_config_path: Optional[str],
    use_dummy_verl: bool,
    requires_openrouter: bool,
    node_num: Optional[int] = None,
    divide_resources_by: int = 1,
) -> Path:
    """Generate an sbatch script for a prepared run.

    Returns the path to the generated sbatch_job.sh.
    """
    run_dir = Path(prepared.run_metadata.run_dir)
    meta = prepared.run_metadata

    # Infer GPUs from merged config
    n_gpus: int = _get_nested(
        d=prepared.merged_config,
        keys=["trainer", "n_gpus_per_node"],
    )

    # Experiment name for job name
    experiment_name = _get_nested(
        d=prepared.merged_config,
        keys=["trainer", "experiment_name"],
        default="vfh_run",
    )
    job_name = f"{experiment_name}_{meta.run_id}"

    # Create sbatch output directory
    sbatch_log_dir = run_dir / "daemon_logs" / "sbatch"
    sbatch_log_dir.mkdir(parents=True, exist_ok=True)

    # Build the run-prepared command
    run_prepared_parts = [
        "python -m vfh.orchestrator run-prepared",
        f"    --run-dir {run_dir.resolve()}",
    ]
    if orch_config_path:
        run_prepared_parts.append(f"    --orch-config {Path(orch_config_path).resolve()}")
    if use_dummy_verl:
        run_prepared_parts.append("    --dummy")
    run_prepared_cmd = " \\\n".join(run_prepared_parts)

    # Dump full environment to a sourceable file (Ray/NCCL/etc. need many vars)
    _SKIP_ENV_PREFIXES = ("SLURM_", "SBATCH_")
    _SKIP_ENV_EXACT = {
        "HOSTNAME", "PWD", "OLDPWD", "SHLVL", "_", "TERM_SESSION_ID",
        "ROCR_VISIBLE_DEVICES",  # conflicts with CUDA_VISIBLE_DEVICES
    }
    env_file = sbatch_log_dir / "env.sh"
    with open(env_file, "w") as f:
        f.write("# Environment captured at sbatch creation time\n")
        for key, value in sorted(os.environ.items()):
            if key in _SKIP_ENV_EXACT:
                continue
            if any(key.startswith(p) for p in _SKIP_ENV_PREFIXES):
                continue
            # Escape single quotes in values for safe shell export
            escaped = value.replace("'", "'\\''")
            f.write(f"export {key}='{escaped}'\n")

    cwd = Path.cwd().resolve()

    nodelist_line = ""
    if node_num is not None:
        node_name = _SLURM_NODE_MAP[node_num]
        nodelist_line = f"\n#SBATCH --nodelist={node_name}"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={n_gpus}
#SBATCH --cpus-per-task={_SLURM_NODE_CPUS // divide_resources_by}
#SBATCH --mem={_SLURM_NODE_MEM_MB // divide_resources_by // 1024}G
#SBATCH --time={time_limit}
#SBATCH --output={sbatch_log_dir.resolve()}/run.out
#SBATCH --error={sbatch_log_dir.resolve()}/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu{nodelist_line}

# --- Environment (captured at sbatch creation time) ---
source {env_file.resolve()}

cd {cwd}
eval "$(conda shell.bash hook)"
conda activate hope

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='{run_dir.resolve()}/run_metadata.json5', n_gpus={n_gpus})" || echo "WARNING: run tracker registration failed"

{run_prepared_cmd}
"""

    script_path = run_dir / "sbatch_job.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    print(f"  sbatch script: {script_path}")
    return script_path


def _append_sbatch_jsonl(
    logs_root: str,
    prepared: PreparedRun,
    sbatch_script_path: Path,
    time_limit: str,
    n_gpus: int,
    launch_command: str,
) -> None:
    """Append an entry to both sbatch_runs.jsonl and sbatch_runs_editable.jsonl."""
    meta = prepared.run_metadata
    entry = {
        "run_id": meta.run_id,
        "description": meta.description,
        "run_dir": meta.run_dir,
        "sbatch_script": str(sbatch_script_path.resolve()),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "slurm_time_limit": time_limit,
        "n_gpus": n_gpus,
        "wandb_url": meta.wandb_url,
        "launch_command": launch_command,
    }
    line = json.dumps(obj=entry) + "\n"

    verlrun_dir = Path(logs_root) / "VerlRun"
    verlrun_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("sbatch_runs.jsonl", "sbatch_runs_editable.jsonl"):
        filepath = verlrun_dir / filename
        with open(filepath, "a") as f:
            f.write(line)


def _validate_time_format(time_str: str) -> None:
    """Validate HH:MM:SS format."""
    if not re.match(r"^\d{1,2}:\d{2}:\d{2}$", time_str):
        raise ValueError(
            f"--time must be in HH:MM:SS format, got: {time_str!r}"
        )


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
        "--no-openrouter", action="store_true",
        help="Skip OpenRouter credit check (default: check is enabled)",
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
    new_parser.add_argument(
        "--note", default=None, metavar="TEXT",
        help="Free-form note written to NOTE.md in the run directory",
    )
    new_parser.add_argument(
        "--print-config", action="store_true",
        help="Print the merged config (base + overrides) with override fields highlighted, then exit.",
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
        "--no-openrouter", action="store_true",
        help="Skip OpenRouter credit check (default: check is enabled)",
    )
    cont_parser.add_argument(
        "--dummy", action="store_true",
        help="Use vfh.dummy_verl instead of real verl",
    )
    cont_parser.add_argument(
        "--extra-overrides", nargs="*", default=[], metavar="KEY=VAL",
        help="Extra raw Hydra overrides appended last",
    )
    cont_parser.add_argument(
        "--note", default=None, metavar="TEXT",
        help="Free-form note written to NOTE.md in the run directory",
    )

    # --- run-prepared (SLURM-time execution of a pre-prepared run) ---
    prep_parser = subparsers.add_parser(
        "run-prepared",
        help="Execute a previously prepared run (used by sbatch scripts)",
        description="""
Execute a run that was already prepared (config resolved, run dir created,
validation done) by a prior 'new --sbatch' or 'continue --sbatch' invocation.

Reads hydra overrides from run_metadata.json5 in the run directory.
Skips config resolution and validation.  Creates subdirs, spawns checkpoint
daemons, sets up output tee, and os.execvp into verl.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    prep_parser.add_argument(
        "--run-dir", required=True, metavar="PATH",
        help="Path to the prepared run directory (must contain run_metadata.json5)",
    )
    prep_parser.add_argument(
        "--dummy", action="store_true",
        help="Use vfh.dummy_verl instead of real verl",
    )
    prep_parser.add_argument(
        "--orch-config", default=None, metavar="PATH",
        help="Path to orchestrator JSON5 config (checkpoint daemon settings, etc.). "
             "Defaults to built-in defaults if omitted.",
    )
    prep_parser.add_argument(
        "--enable-checkpoint-daemon", action="store_true",
        help="Enable the checkpoint backup daemon (disabled by default).",
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
        # --- sbatch flags ---
        p.add_argument(
            "--sbatch", action="store_true",
            help="Generate an sbatch script instead of launching directly. "
                 "Requires --time.",
        )
        p.add_argument(
            "--time", default=None, metavar="HH:MM:SS",
            help="SLURM time limit (required with --sbatch). Format: HH:MM:SS.",
        )
        p.add_argument(
            "--dont-auto-sbatch", action="store_true",
            help="Generate the sbatch script but don't submit it automatically.",
        )
        p.add_argument(
            "--node", type=int, default=None, choices=[1, 2], metavar="N",
            help="SLURM node to run on: 1=bleak-mushroom-dove, 2=better-ginkgo-dragonfly. "
                 "If omitted, SLURM picks automatically.",
        )
        p.add_argument(
            "--divide-resources-by", type=int, default=None, metavar="N",
            help="Divide node CPUs and memory by N for SLURM resource requests "
                 f"(node has {_SLURM_NODE_CPUS} CPUs, {_SLURM_NODE_MEM_MB // 1024}G mem). "
                 "Required with --sbatch. E.g. --divide-resources-by 2 for two jobs per node.",
        )
        p.add_argument(
            "--enable-checkpoint-daemon", action="store_true",
            help="Enable the checkpoint backup daemon (disabled by default). "
                 "Spawns a background process that runs dvc add/push on new checkpoints.",
        )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- run-prepared: special path (no validation, no config resolution) ---
    if args.command == "run-prepared":
        orch_config = load_orchestrator_config(config_path=args.orch_config)
        if args.enable_checkpoint_daemon:
            for daemon_cfg in orch_config.checkpoint_daemons:
                daemon_cfg.enabled = True
        run_metadata = load_run_metadata(run_dir=args.run_dir)
        validate_run_metadata(metadata=run_metadata)
        # Reconstruct merged_config from hydra overrides so downstream code
        # (e.g. register_run, _generate_sbatch) can look up values like n_gpus_per_node.
        merged_config = _hydra_overrides_to_nested_dict(run_metadata.resolved_hydra_overrides)
        prepared = PreparedRun(
            merged_config=merged_config,
            hydra_overrides=run_metadata.resolved_hydra_overrides,
            run_metadata=run_metadata,
        )
        exec_prepared(
            prepared=prepared,
            orch_config=orch_config,
            use_dummy_verl=args.dummy,
        )
        return  # exec_prepared never returns, but for clarity

    # --- new / continue: standard path ---

    # Determine validation mode
    if args.yy:
        val_mode = ValidationMode.SKIP
    elif args.y:
        val_mode = ValidationMode.AUTO_APPROVE
    else:
        val_mode = ValidationMode.FULL

    # Validate sbatch flags
    if args.sbatch:
        if not args.time:
            parser.error("--time is required when using --sbatch (format: HH:MM:SS)")
        if not args.divide_resources_by:
            parser.error("--divide-resources-by is required when using --sbatch")
    if args.time:
        _validate_time_format(time_str=args.time)

    orch_config = load_orchestrator_config(
        config_path=args.orch_config,
        cli_overrides={"validation_mode": val_mode.value},
    )

    # Wire --enable-checkpoint-daemon flag
    if args.enable_checkpoint_daemon:
        for daemon_cfg in orch_config.checkpoint_daemons:
            daemon_cfg.enabled = True

    if args.command == "new" and getattr(args, "print_config", False):
        from vfh.config_resolver import resolve_config, _load_json5
        merged, hydra_overrides = resolve_config(
            base_config_path=args.base_config,
            overrides_path=args.overrides,
            extra_hydra_overrides=args.extra_overrides,
        )
        # Determine which keys came from overrides
        override_keys: set[str] = set()
        if args.overrides:
            overrides_raw = _load_json5(path=args.overrides)
            def _collect_keys(d: dict, prefix: str = "") -> None:
                for k, v in d.items():
                    bare = k.lstrip("+")
                    full = f"{prefix}.{bare}" if prefix else bare
                    override_keys.add(full)
                    if isinstance(v, dict):
                        _collect_keys(d=v, prefix=full)
            _collect_keys(d=overrides_raw)

        # Pretty-print with override highlighting
        CYAN = "\033[36m"
        YELLOW = "\033[1;33m"
        NC = "\033[0m"

        def _print_config(d: dict, indent: int = 0, path: str = "") -> None:
            for k, v in d.items():
                bare = k.lstrip("+")
                full_path = f"{path}.{bare}" if path else bare
                is_override = full_path in override_keys
                marker = f"{YELLOW}[override]{NC} " if is_override else ""
                prefix = "  " * indent
                if isinstance(v, dict):
                    print(f"{prefix}{marker}{CYAN}{k}{NC}:")
                    _print_config(d=v, indent=indent + 1, path=full_path)
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"{prefix}{marker}{CYAN}{k}{NC}:")
                    for i, item in enumerate(v):
                        print(f"{prefix}  [{i}]:")
                        _print_config(d=item, indent=indent + 2, path=full_path)
                else:
                    print(f"{prefix}{marker}{CYAN}{k}{NC}: {v}")

        print(f"\n{'='*60}")
        print(f"Merged Config (base: {args.base_config})")
        if args.overrides:
            print(f"  + overrides: {args.overrides}")
        print(f"{'='*60}\n")
        _print_config(d=merged)
        print(f"\n{'='*60}")
        print(f"Hydra overrides: {len(hydra_overrides)} keys")
        print(f"{'='*60}")
        sys.exit(0)

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
            # Validate checkpoint exists and has weight files (DVC-aware).
            # World_size check is deferred to _prepare_new where we have the resolved config.
            ckpt_dir = Path(args.fork_from) / "checkpoints" / f"global_step_{fork_step}"
            _validate_resume_checkpoint(ckpt_dir=ckpt_dir)
            origin = RunOrigin(
                fork_reason=ForkReason.INTENTIONAL_FORK,
                parent_run_id=parent_meta.run_id,
                parent_run_dir=parent_meta.run_dir,
                parent_checkpoint_step=fork_step,
            )

        run_config: NewRunConfig | ContinueRunConfig = NewRunConfig(
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

    requires_openrouter = not getattr(args, "no_openrouter", False)
    note: str | None = getattr(args, "note", None)

    if args.sbatch:
        # --- Sbatch mode: prepare + generate script ---
        prepared = prepare(
            orch_config=orch_config,
            run_config=run_config,
            requires_openrouter=requires_openrouter,
        )

        # Save launching command, code diff, and optional note
        run_dir_path = Path(prepared.run_metadata.run_dir)
        (run_dir_path / "launching_command.txt").write_text(
            "python -m vfh.orchestrator " + " ".join(sys.argv[1:]) + "\n"
        )
        _save_code_diff(run_dir=run_dir_path)
        if note:
            (run_dir_path / "NOTE.md").write_text(note + "\n")
            print(f"  Note saved: {run_dir_path / 'NOTE.md'}")

        script_path = _generate_sbatch(
            prepared=prepared,
            time_limit=args.time,
            orch_config_path=args.orch_config,
            use_dummy_verl=args.dummy,
            requires_openrouter=requires_openrouter,
            node_num=args.node,
            divide_resources_by=args.divide_resources_by,
        )

        n_gpus: int = _get_nested(
            d=prepared.merged_config,
            keys=["trainer", "n_gpus_per_node"],
        )
        _append_sbatch_jsonl(
            logs_root=orch_config.logs_root,
            prepared=prepared,
            sbatch_script_path=script_path,
            time_limit=args.time,
            n_gpus=n_gpus,
            launch_command=" ".join(sys.argv),
        )

        if not args.dont_auto_sbatch:
            print(f"\nSubmitting: sbatch {script_path}")
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  {result.stdout.strip()}")
            else:
                print(f"  sbatch failed (exit {result.returncode}): {result.stderr.strip()}")
                sys.exit(1)
        else:
            print(f"\n--dont-auto-sbatch: script generated but not submitted.")
            print(f"  Run manually: sbatch {script_path}")
    else:
        # --- Direct launch mode ---
        # Always prepare separately so we can write launching_command.txt + NOTE.md
        prepared = prepare(
            orch_config=orch_config,
            run_config=run_config,
            requires_openrouter=requires_openrouter,
        )
        run_dir_path = Path(prepared.run_metadata.run_dir)
        (run_dir_path / "launching_command.txt").write_text(
            "python -m vfh.orchestrator " + " ".join(sys.argv[1:]) + "\n"
        )
        _save_code_diff(run_dir=run_dir_path)
        if note:
            (run_dir_path / "NOTE.md").write_text(note + "\n")
            print(f"  Note saved: {run_dir_path / 'NOTE.md'}")
        # Register with run tracker (direct launch — not sbatch, which registers in the script)
        try:
            from vfh.run_tracker import register_run_from_metadata
            n_gpus = int(_get_nested(
                d=prepared.merged_config,
                keys=["trainer", "n_gpus_per_node"],
            ))
            register_run_from_metadata(
                metadata_path=str(run_dir_path / "run_metadata.json5"),
                n_gpus=n_gpus,
            )
        except Exception as e:
            print(f"  WARNING: Failed to register run with tracker: {e}")
        exec_prepared(
            prepared=prepared,
            orch_config=orch_config,
            use_dummy_verl=args.dummy,
        )


if __name__ == "__main__":
    main()
