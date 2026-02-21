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
    run_metadata.json5      # full provenance: origin, wandb url, resolved overrides, child links
    checkpoints/            # verl writes here (trainer.default_local_dir)
    rollouts/train/ val/    # verl writes here
    daemon_logs/            # checkpoint daemon log
```
The 8-char `run_id` (from `wandb.util.generate_id()`) doubles as the wandb run ID so every run dir is 1:1 with a wandb run.

### Run lineage (DAG)
Runs link to parents via `RunOrigin` in metadata. `ForkReason`: ROOT (fresh), INTENTIONAL_FORK (user chose to fork from a checkpoint), CONTINUE (crashed/stopped). Parent metadata gets `child_run_ids` appended. This forms a DAG traversable by the (not-yet-built) tree_traverser.

### Checkpoint daemon
Background process spawned by the orchestrator. Watches the verl PID (via `os.kill(pid, 0)`). Periodically runs `dvc add` + `dvc push` on the checkpoints dir, verifies backup via `dvc status --cloud`. Optionally cleans backed-up checkpoint contents (keeps empty dir as marker). Does a final backup pass when verl exits, then self-terminates. Logs everything to `daemon_logs/checkpoint_daemon.log`.

### wandb ID injection
Small patch to verl: `Tracking.__init__` accepts `wandb_run_id` → passes `id=..., resume="allow"` to `wandb.init()`. `ray_trainer.py` threads it from config via `self.config.trainer.get("wandb_run_id", None)`.

### dummy_verl
Uses the same `@hydra.main(config_path=..., config_name="ppo_trainer")` entry point as real verl. Accepts identical config + CLI overrides. Instead of training, writes fake `global_step_N/` dirs at `save_freq` intervals. Supports `+dummy.crash_after_step`, `+dummy.sleep_per_step`, `+dummy.total_steps`. Used to test the full orchestrator pipeline without GPUs.

## Design choices
- **pyjson5 for configs** (not YAML) — supports comments and trailing commas. dacite for deserialization back to dataclasses.
- **Subprocess launch** — verl is called via `subprocess.Popen`, not imported. Keeps VFH decoupled; the daemon watches the PID.
- **`logs/VerlRun/` un-gitignored** — `.gitignore` has `!logs/VerlRun/` so DVC can create `.dvc` files there.
- **No sparsification logic** — just backup all checkpoints and optionally clean after confirmed backup. Simpler than frequency-based sparsification.
- **Orchestrator config abstracted** — `load_orchestrator_config()` loads from JSON5 file + CLI overrides. Source format can change later.
- **wandb entity from `$WANDB_ENTITY` env var**, project from run config (`trainer.project_name`), not orchestrator config.

## Key files
| File | Purpose |
|------|---------|
| `vfh/vfh_types.py` | All dataclasses and enums |
| `vfh/config_resolver.py` | JSON5 merge → Hydra override list |
| `vfh/run_manager.py` | Create/load run dirs, DAG linking |
| `vfh/orchestrator.py` | Main entry + argparse CLI (new/continue/fork) |
| `vfh/checkpoint_daemon.py` | Background backup + cleanup process |
| `vfh/dummy_verl.py` | Fake verl for testing |
| `vfh/configs/` | JSON5 config templates |
| `verl/utils/tracking.py` | Patched: accepts `wandb_run_id` param |
| `verl/trainer/ppo/ray_trainer.py` | Patched: threads `wandb_run_id` to Tracking |

## Not yet built
- **tree_traverser** (M6) — DAG navigation, wandb URL generation, run filtering, notable_runs.jsonl. Needs discussion on interactive UX.
- **Base daemon class** — refactor if more daemons are added.
- **validate_env.py fix** — auto-remove run dirs that have no meaningful data (no `global_step_N`, no `.jsonl` rollouts) instead of erroring.
