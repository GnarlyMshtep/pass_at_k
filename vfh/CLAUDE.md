# VFH (Verl For Humans)

## Purpose
Replace hand-written bash scripts (120+ Hydra CLI overrides each) with a Python orchestrator that manages the full lifecycle of verl RL training runs: config resolution, run directory creation, validation, launch, checkpoint backup, and run lineage tracking.

## Architecture

### Config system
- **Two-layer JSON5 configs**: a `base-config` (general hyperparams — batch sizes, save freq, architecture flags) + an `overrides` file (per-run specifics — model, reward, experiment name). Deep-merged then flattened to Hydra override strings.
- Keys prefixed with `+` in JSON5 become `+key=value` Hydra overrides (for fields not in verl's `ppo_trainer.yaml` schema). The `+` propagates to all nested children.
- `$ENV_VAR` in string values are expanded via `os.path.expandvars`.
- Config files live in `vfh/configs/`.

### Run directories
Each `python -m vfh.orchestrator new` creates:
```
logs/VerlRun/{MM}/{DD}/{desc}_{HH}_{mm}_{run_id}/
    run_metadata.json5      # full provenance (pretty-printed via json.dump indent=2)
    verl_output.log         # tee'd copy of verl's stdout/stderr
    checkpoints/            # verl writes here (trainer.default_local_dir)
    rollouts/train/ val/    # verl writes here
    daemon_logs/            # checkpoint daemon log
```
The 8-char `run_id` (from `wandb.util.generate_id()`) doubles as the wandb run ID so every run dir is 1:1 with a wandb run.

### Run lineage (DAG)
Runs link to parents via `RunOrigin` in metadata. `ForkReason`: ROOT (fresh), INTENTIONAL_FORK (user chose to fork from a checkpoint), CONTINUE (crashed/stopped). Parent metadata gets `child_run_ids` appended. This forms a DAG traversable by the (not-yet-built) tree_traverser.

### Process & I/O architecture
```
Terminal (you run: python -m vfh.orchestrator new ...)
  │
  ├─ orchestrator spawns daemon: Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
  │    • daemon output goes ONLY to daemon_logs/checkpoint_daemon.log
  │    • nothing reaches terminal
  │
  ├─ orchestrator sets up tee: fork() a child that reads a pipe,
  │   writes to both terminal + {run_dir}/verl_output.log
  │
  └─ os.execvp verl (stdout/stderr redirected to pipe)
       • verl output → pipe → tee child → terminal + file
       • PID preserved, daemon watches correctly
```

### Checkpoint daemon
Background process spawned by the orchestrator (stdout/stderr silenced via DEVNULL, `start_new_session=True` so it runs in its own process group — prevents SLURM from waiting on it when verl exits). Watches the verl PID (via `os.kill(pid, 0)`). Two-phase backup pass each poll:
1. **Phase 1 — Backup**: discovers new complete checkpoints, runs `dvc add` + `dvc push` **per global_step** (each step gets its own `.dvc` file), verifies via `dvc status --cloud`
2. **Phase 2 — Clean**: iterates ALL previously backed-up steps, cleans any that are no longer the latest (removes contents, keeps empty dir + `.cleaned_by_daemon` marker)

DVC subprocess timeouts: `dvc add` 30min, `dvc push` 30min, `dvc gc` 10min. On timeout, logs warning and retries next poll.

After push, runs `dvc gc --not-in-remote -w -f -v` to clean local cache. Does a final backup pass when verl exits, then self-terminates. All timestamps in US Eastern (America/New_York).

### wandb ID injection
Small patch to verl: `Tracking.__init__` accepts `wandb_run_id` → passes `id=..., resume="allow"` to `wandb.init()`. `ray_trainer.py` threads it from config via `self.config.trainer.get("wandb_run_id", None)`.

### dummy_verl
Uses the same `@hydra.main(config_path=..., config_name="ppo_trainer")` entry point as real verl. Accepts identical config + CLI overrides. Instead of training, writes fake `global_step_N/` dirs at `save_freq` intervals. Supports `+dummy.crash_after_step`, `+dummy.sleep_per_step`, `+dummy.total_steps`. Used to test the full orchestrator pipeline without GPUs.

## Design choices
- **pyjson5 for reading configs** (not YAML) — supports comments and trailing commas. dacite for deserialization back to dataclasses. **json.dump for writing** run_metadata (pyjson5.dump ignores indent).
- **`os.execvp` launch** — orchestrator replaces itself with verl via `os.execvp` (not `subprocess.Popen`). Preserves PID, so daemons (spawned before exec) correctly watch the verl process. Verl runs as if launched directly from the shell — no parent process, clean asyncio state.
- **Output tee** — before `os.execvp`, forks a tee child process. Verl's stdout/stderr go to both the terminal and `{run_dir}/verl_output.log`.
- **Subdirs created after validation** — `checkpoints/`, `rollouts/`, `daemon_logs/` are created *after* validate_env runs, so validation never sees freshly-created empty dirs.
- **Validation modes**: no flag = run + prompt; `-y` (AUTO_APPROVE) = run + warn on failure, continue anyway; `-yy` (SKIP) = skip entirely.
- **validate_env auto-removes empty output dirs** — if `checkpoints/` or `rollouts/` exist but have no `global_step_N` dirs / `.jsonl` files, they're silently removed (not errored on).
- **`--fork-from` requires `--fork-step`** — no implicit "latest checkpoint" default; must specify the step explicitly.
- **`continue` creates a child run dir** — uses `resume_from_path` pointing to parent checkpoint, auto-picks latest step.
- **`logs/VerlRun/` un-gitignored** — `.gitignore` has `!logs/VerlRun/` so DVC can create `.dvc` files there.
- **Per-step DVC tracking** — each `global_step_N` gets its own `.dvc` file (`global_step_N.dvc`), enabling independent pull/push/status per step. Old whole-dir `checkpoints.dvc` is auto-removed at daemon startup.
- **No sparsification logic** — just backup all checkpoints and optionally clean after confirmed backup. Simpler than frequency-based sparsification.
- **Checkpoint daemon cache cleanup** — runs `dvc gc --not-in-remote -w -f -v` once after all step pushes complete.
- **Orchestrator config abstracted** — `load_orchestrator_config()` loads from JSON5 file + CLI overrides. Source format can change later.
- **wandb entity from `$WANDB_ENTITY` env var**, project from run config (`trainer.project_name`), not orchestrator config.
- **No asyncio semaphore in code execution** — removed module-level semaphore from `code_execution_utils.py`. Only 10-20 test cases per sample; overhead not worth the complexity.
- **pathspec pinned <1.0.0** — DVC 3.66.1 needs `_DIR_MARK` from pathspec which was removed in 1.0.0. `black` wants >=1.0.0 but that's cosmetic.

## Key files
| File | Purpose |
|------|---------|
| `vfh/vfh_types.py` | All dataclasses and enums |
| `vfh/config_resolver.py` | JSON5 merge → Hydra override list |
| `vfh/run_manager.py` | Create/load run dirs, DAG linking |
| `vfh/orchestrator.py` | Main entry + argparse CLI (new/continue/fork), output tee, daemon spawning |
| `vfh/checkpoint_daemon.py` | Background backup + cleanup process (two-phase: backup then clean) |
| `vfh/dummy_verl.py` | Fake verl for testing |
| `vfh/configs/` | JSON5 config templates |
| `vfh/direct_launch_test.sh` | Direct verl launch (bypasses orchestrator) for isolation testing |
| `verl/utils/tracking.py` | Patched: accepts `wandb_run_id` param |
| `verl/trainer/ppo/ray_trainer.py` | Patched: threads `wandb_run_id` to Tracking |

### Metadata sanity check
`validate_run_metadata()` in `run_manager.py` checks `resolved_hydra_overrides` before verl launches. Called both at prepare-time (in `_prepare_new`) and at run-time (in `run-prepared`). Checks: VFH-managed paths (`default_local_dir`, `rollout_data_dir`, `validation_data_dir`) exist and point inside `run_dir`; `wandb_run_id` matches metadata; fork/continue runs have `resume_mode=resume_path` + valid `resume_from_path`; root runs with `resume_path` get a warning. Raises `ValueError` listing all issues.

### Sbatch integration
`--sbatch --time HH:MM:SS` on `new` or `continue` generates a SLURM sbatch script instead of launching directly. Two-phase design:
- **Phase 1 (creation time)**: `prepare()` resolves config, creates run dir, runs validation. `_generate_sbatch()` writes `{run_dir}/sbatch_job.sh` with SLURM directives (`--gpus` inferred from `trainer.n_gpus_per_node`, `--job-name` from experiment name + run_id, mail notifications to mshtepel@andrew.cmu.edu). Appends to `logs/VerlRun/sbatch_runs.jsonl` (append-only history) and `sbatch_runs_editable.jsonl` (editable checklist).
- **Phase 2 (SLURM run time)**: sbatch script calls `python -m vfh.orchestrator run-prepared --run-dir <path>` which reads `resolved_hydra_overrides` from `run_metadata.json5`, creates subdirs, spawns checkpoint daemon, sets up tee, and `os.execvp` into verl. Skips config resolution and validation.

Flags: `--dont-auto-sbatch` generates the script without submitting. By default, `sbatch` is called automatically. SLURM output goes to `{run_dir}/daemon_logs/sbatch/run.out` and `run.err`. Environment variables (`WANDB_ENTITY`, `OPENROUTER_API_KEY`, `CUDA_VISIBLE_DEVICES`) are captured at creation time and baked into the script. The script activates the `hope` conda environment.

## Not yet built
- **tree_traverser** (M6) — DAG navigation, wandb URL generation, run filtering, notable_runs.jsonl. Needs discussion on interactive UX.
- **Base daemon class** — refactor if more daemons are added.
