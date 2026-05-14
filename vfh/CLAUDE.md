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
    launching_command.txt   # the full CLI command that created this run
    code.diff               # git diff HEAD at launch time (uncommitted changes)
    NOTE.md                 # optional free-form note (--note "...")
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

### wandb ID injection & run grouping
Small patch to verl: `Tracking.__init__` accepts `wandb_run_id` → passes `id=..., resume="allow"` to `wandb.init()`, and `wandb_group` → passes `group=...` to `wandb.init()`. `ray_trainer.py` threads both from config via `self.config.trainer.get(...)`.

**Wandb grouping for lineage**: All runs in a lineage (root + continuations + forks) share the same `wandb_group` = the root ancestor's run_id. This makes them visually grouped in the wandb UI. The orchestrator walks up the parent chain via `_find_root_run_id()` to determine the group. Root runs use their own run_id as the group (so future continuations match).

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
| `vfh/checkpoint_daemon.py` | Background backup + cleanup process (disabled by default) |
| `vfh/dvc_backup/` | DVC backup package (batched, round-trip verified) |
| `vfh/catalog.py` | Interactive run catalog with tags, lineage, fuzzy search |
| `vfh/catalog_types.py` | Dataclasses for catalog (CatalogEntry, CatalogTag, Catalog) |
| `vfh/run_tracker.py` | Run tracker library: register, load, save, refresh (wandb polling). See [`RUN_TRACKING.md`](RUN_TRACKING.md) |
| `vfh/run_tracker_types.py` | RunState enum + TrackedRun dataclass |
| `vfh/run_tracker_viewer.py` | Interactive viewer for tracked runs + catalog (toggle with [c]) |
| `vfh/wandb_view.py` | Build a saved wandb workspace URL for a selected set of runs (used by `[v]iew` in the viewer) |
| `vfh/family_tree.py` | Interactive family tree explorer for run lineage |
| `vfh/interactive_utils.py` | Shared interactive helpers (ANSI colors, clipboard via OSC 52, cancel handling) |
| `vfh/RUN_TRACKING.md` | **Minimal reference for registering/viewing runs — point other Claude instances here** |
| `vfh/dummy_verl.py` | Fake verl for testing |
| `vfh/configs/` | JSON5 config templates |
| `vfh/direct_launch_test.sh` | Direct verl launch (bypasses orchestrator) for isolation testing |
| `verl/utils/tracking.py` | Patched: accepts `wandb_run_id` param |
| `verl/trainer/ppo/ray_trainer.py` | Patched: threads `wandb_run_id` to Tracking |

### Metadata sanity check
`validate_run_metadata()` in `run_manager.py` checks `resolved_hydra_overrides` before verl launches. Called both at prepare-time (in `_prepare_new`) and at run-time (in `run-prepared`). Checks: VFH-managed paths (`default_local_dir`, `rollout_data_dir`, `validation_data_dir`) exist and point inside `run_dir`; `wandb_run_id` matches metadata; fork/continue runs have `resume_mode=resume_path` + valid `resume_from_path`; root runs with `resume_path` get a warning. Raises `ValueError` listing all issues.

### Config inspection
`--print-config` on `new` resolves the merged config (base + overrides + extra-overrides), prints it with `[override]` markers highlighting fields that came from the overrides file, then exits without creating a run. Useful for reviewing the full config before launching.

```
python -m vfh.orchestrator new --base-config ... --overrides ... --print-config
```

### Sbatch integration
`--slurm.sbatch --slurm.time HH:MM:SS --slurm.divide-resources-by N` on `new` or `continue` generates a SLURM sbatch script instead of launching directly. `divide-resources-by` splits node CPUs/memory (e.g. 2 when running two 4-GPU jobs on an 8-GPU node). Two-phase design:
- **Phase 1 (creation time)**: `prepare()` resolves config, creates run dir, runs validation. `_generate_sbatch()` writes `{run_dir}/sbatch_job.sh` with SLURM directives (`--gpus` inferred from `trainer.n_gpus_per_node`, `--job-name` from experiment name + run_id, mail notifications to mshtepel@andrew.cmu.edu). Appends to `logs/VerlRun/sbatch_runs.jsonl` (append-only history) and `sbatch_runs_editable.jsonl` (editable checklist).
- **Phase 2 (SLURM run time)**: sbatch script registers with the run tracker via a one-liner (`register_run_from_metadata`), then calls `python -m vfh.orchestrator run-prepared --run-dir <path>` which reads `resolved_hydra_overrides` from `run_metadata.json5` (reconstructed into `merged_config` via `_hydra_overrides_to_nested_dict`), creates subdirs, spawns checkpoint daemon, sets up tee, and `os.execvp` into verl. Skips config resolution and validation.

**Always run validation first.** Before launching with `-y` or `-yy`, first do a dry run without any `-y` flag (use `--slurm.dont-auto-sbatch` to avoid auto-submitting) to see what `validate_env.py` catches. Only skip validation after confirming the errors are expected (e.g. OpenRouter credits check on a non-monitor phase). This prevents silent misconfigs from reaching SLURM.

Flags: `--slurm.dont-auto-sbatch` generates the script without submitting. By default, `sbatch` is called automatically. SLURM output goes to `{run_dir}/daemon_logs/sbatch/run.out` and `run.err`. Environment variables (`WANDB_ENTITY`, `OPENROUTER_API_KEY`, `CUDA_VISIBLE_DEVICES`) are captured at creation time and baked into the script.

**Python environment**: Use the pip venv at `/shared/matan/code/pass_at_k/.venv/` (not the `hope` conda env). The venv has `datasets==4.5.0` which fixes parquet metadata incompatibilities that crash training with `hope`'s `datasets==3.2.0`. Activate with `source .venv/bin/activate`. The sbatch script should use this venv instead of conda.

### Checkpoint daemon (disabled by default)
`CheckpointDaemonConfig.enabled` defaults to `False`. Use `--enable-checkpoint-daemon` on `new`, `continue`, or `run-prepared` to opt in. The daemon was failing frequently and is replaced by the standalone DVC backup script for manual backups.

### DVC backup package (`vfh/dvc_backup/`)
Package for bulk-backing up checkpoints and rollouts via DVC with round-trip verification. Not part of the orchestrator — run manually.

```
python -m vfh.dvc_backup                                    # discover + backup (errors if not enough space)
python -m vfh.dvc_backup --space-budget-gb 250 --yes        # batched mode, 250GB per batch, auto-approve
python -m vfh.dvc_backup --space-budget-gb 100 --max-batches 2  # test 2 batches only
python -m vfh.dvc_backup --path logs/VerlRun/.../run_dir    # single run dir
python -m vfh.dvc_backup --skip-roundtrip-verify            # push-only, no pull-back verification
python -m vfh.dvc_backup --dry-run                          # show what would be done
python -m vfh.dvc_backup --verbose                          # debug output
```

**Package structure:**
| Module | Purpose |
|--------|---------|
| `config.py` | `DvcBackupConfig` dataclass (tyro CLI), timeouts, constants |
| `types.py` | `BackupTarget`, `RunSummary` |
| `discovery.py` | Find runs + targets (checkpoints: `global_step_N` dirs, rollouts: whole `rollouts/` dir) |
| `dvc_ops.py` | DVC/git wrappers: batch add, batch push, fast cache clear, stale lock handling |
| `verification.py` | Round-trip verify: move aside → pull from S3 → hash compare → accept/reject |
| `presentation.py` | Summary display, `human_size()` |
| `pipeline.py` | Main orchestration: space check, batching loop, freed-space rollover |
| `backup_log.py` | Append-only log at `logs/VerlRun/dvc_backup_logs.txt` |
| `test_setup.py` | Create fake VerlRun for testing |

**Batch pipeline flow** (per batch):
1. `dvc add` all targets in one call (handles legacy `.dvc` cleanup + `git rm --cached` automatically)
2. `git commit` the `.dvc` files
3. `dvc push` all targets in one call (DVC parallelizes S3 uploads via `jobs=64`)
4. Clear DVC cache (fast `rm -rf`, not `dvc gc`)
5. Round-trip verify each target: move original → `dvc pull` from S3 → hash compare → delete both copies if match, restore original if mismatch
6. Clear cache again, freed space rolls into next batch's budget

**Space safety:** 2.5x multiplier — `dvc add` doubles space (original + cache), plus verification headroom. If no `--space-budget-gb`, errors with suggested budget. Freed space from verified+deleted targets accumulates across batches.

**DVC tracking convention:** Checkpoints tracked per-step (`checkpoints/global_step_N.dvc`). Rollouts tracked as whole dir (`rollouts.dvc` at run dir level, covering `rollouts/train/` + `rollouts/val/`). DVC auto-generates per-run `.gitignore`.

**Recovery:** if interrupted and re-run, targets with existing `.dvc` files are detected and pushed directly (skip add). Stale DVC lock files auto-cleared.

**Logging:** Every invocation appends to `logs/VerlRun/dvc_backup_logs.txt` with timestamps.

**Parallel verification:** Per batch, verification moves all originals aside (serial rename), then `dvc pull`s all `.dvc` files in one call (DVC's `jobs=64` parallelizes S3 fetches), then joblib-parallelizes the `compare_dirs` hash comparisons, then finalizes accept/reject per-target. See `dvc_pull_batch` in `dvc_ops.py` and `roundtrip_verify_batch` in `verification.py`.

**Reuse contract — do not break:** tinker-cookbook's `tinker_cookbook/tfh/dvc_backup.py` imports this package via `sys.path` (pass_at_k is not pip-installable as `vfh` — `pyproject.toml` declares `name = "verl"`). The public contract is `vfh.dvc_backup.pipeline.run_backup(config, strategy)` + `vfh.dvc_backup.discovery.DiscoveryStrategy` (with concrete `VFHDiscovery` / `TFHDiscovery`). Any refactor of these signatures must preserve backward compat or update the TFH caller in lock-step.

**Path safety:** all DVC/git commands via `subprocess.run(list_form)` — no shell escaping issues.

### Run catalog (`vfh/catalog.py` + `vfh/catalog_types.py`)
Interactive script for cataloging notable runs with metadata, tags, and lineage.

```
python -m vfh.catalog --path <run_dir_or_wandb_id>
python -m vfh.catalog --path 186oohgn        # resolve by wandb ID
```

**Extractor pattern:** `RunExtractor` ABC with `VFHExtractor` implementation. Reads `run_metadata.json5`, parses `resolved_hydra_overrides` for base model (`actor_rollout_ref.model.path`), reward config (`custom_reward_function.reward_kwargs.reward_config.*`), train dataset (`data.train_files`). Scans for checkpoint range from `global_step_N` dirs + `.dvc` files, and rollout range from `{N}.jsonl` files in `rollouts/train/`. Both ranges are `tuple[int,int] | None` — runs with no checkpoints or rollouts are handled gracefully.

**W&B URL:** extracted from `run_metadata.json5` if present, otherwise constructed from `trainer.project_name` (from hydra overrides) + `$WANDB_ENTITY` (env var, defaults to `matan-shtepel-carnegie-mellon-university`). `[w]` option in interactive flow prints the URL and attempts clipboard copy via `pbcopy`/`xclip`/`xsel`.

**Lineage:** BFS traversal of the run DAG up to depth 20 in both directions (ancestors via `follows`, descendants via `preceded_by`). Traverses through already-cataloged runs and empty runs (no checkpoints) to find the full lineage. Interactive one-at-a-time flow: select an index to view metadata, then `[s]ame desc / [n]ew desc / [w]andb url / [d]on't add`. Already-cataloged runs show as `[cataloged]` and can be `[e]dited`.

**Tags:** fuzzy search via `rapidfuzz`, multi-select, create new with `+tag_name`.

**Cancel/escape:** Type `esc` or Ctrl+C at any prompt to go back. Cancelling tag selection aborts the entire cataloging flow (not just tags).

**Catalog file:** JSON at `$RUN_CATALOG_PATH` (default: `logs/catalog_data/catalog.json`, inside the repo). Stores `{"entries": [...], "tags": [...]}`.

## DVC gotchas (learned the hard way)

- **`dvc status --cloud` is ambiguous after cache clear.** It reports `deleted:` both when (a) the file IS on remote but local cache was gc'd, and (b) the file was NEVER pushed and cache was gc'd. Only trust `"in sync"` (confirmed synced) and `"new:"` (definitely needs push). `"deleted:"` is ambiguous — but `dvc push` is safe as a fallback (see next point).
- **`dvc push` on already-pushed data is cheap (~2-5s).** It queries S3 with `object_exists` per OID (hash check, not re-upload). If all blobs exist on remote, returns instantly with "Everything is up to date." This makes `dvc push` a safe fallback for the ambiguous "deleted" status — worst case it's a quick no-op.
- **`dvc status --cloud` must be checked BEFORE deleting local data or clearing cache.** Once data + cache are gone, the signal is unreliable. The backup script's ordering is: add → push → verify ("in sync") → delete → gc.
- **Old whole-directory `.dvc` files block per-step adds.** If a `checkpoints.dvc` tracks the entire `checkpoints/` dir and its `.dir` manifest was gc'd, `dvc add` on any `global_step_N` inside will fail with `"could not read '<hash>.dir"`. Fix: `dvc remove checkpoints.dvc` (or just `rm` it if dvc remove fails). Two instances were fixed by hand (2026-03-20); shouldn't recur since the daemon is disabled.
- **Stale DVC lock files.** `.dvc/tmp/lock` persists after crashes/timeouts. The backup script auto-detects and clears these (checks if holding PID is alive). If a real DVC process is running, it warns instead.
- **`dvc pull` has no `--dry-run` flag.** Can't cheaply check "is this on remote?" without actually downloading. `dvc fetch` downloads to cache only (no checkout) — lighter but still downloads. `dvc status --cloud` is the only non-downloading check, with the ambiguity caveat above.
- **`git add -f` was previously needed** because root `.gitignore` had `logs/` which swallowed `.dvc` files. Fixed by replacing with `logs/*` + `!logs/VerlRun/`. No more force-adds needed.
- **`logs/.gitignore`** exists and ignores `OriginalQ4BIRunVal` and `ProcMonitEval`. Other dirs under `logs/` (GeminiEval, rl_checkpoint_eval) are caught by the `logs/*` rule.
- **`.dir` manifest loss (2026-03-28 incident).** `dvc remove` on a `.dvc` file deletes its `.dir` manifest from both local cache AND S3 remote. This happened when old whole-directory `.dvc` files were removed during migration to per-step tracking — 81 `.dvc` files across 26 runs became unreachable (individual data blobs exist on S3 by hash but can't be reconstructed without the manifest). `dvc gc` warns about these but can't clean them. `dvc pull` fails. Fix: deleted all 81 broken `.dvc` files (commit `9fea4a74`). **Lesson: never use `dvc remove` — just `rm` the `.dvc` file if needed.**
- **`dvc gc --not-in-remote` is slow (~5-6 min)** because it queries S3 for every cached hash. The backup script uses `rm -rf .dvc/cache/files/md5/*` instead (instant) when all data has been verified on remote. Only use `dvc gc` when you need selective cleanup.

### Run tracker (dashboard)
Tracks ongoing and completed runs. Viewer polls wandb API for state transitions.

- **Registration**: `register_run()` or `register_run_from_metadata()` in `vfh/run_tracker.py`. See [`RUN_TRACKING.md`](RUN_TRACKING.md) for the minimal API.
  - **Sbatch runs**: registered once via a one-liner injected into the sbatch script (runs at SLURM launch time, not at script generation time). `exec_prepared()` does NOT register.
  - **Direct launch**: registered in the main CLI handler right before `exec_prepared()`.
- **States**: REGISTERED → RUNNING → FINISHED/CRASHED. REGISTERED_NO_WANDB if wandb doesn't respond after grace period (keeps polling). RUNNING_NO_WANDB for runs registered without wandb info. REVIEWED = user-handled.
- **Viewer**: `python -m vfh.run_tracker_viewer` — ANSI-colored, grouped by state, sorted by most recent. Two modes toggled with `[c]`:
  - **Tracker mode** (default): shows tracked runs grouped by state. Actions: [w]andb, [p]ath, [l]aunch cmd, [n]ote (+ offers mark reviewed), [r]emove, [m]ark reviewed, [c]atalog. [f] toggles filter-empty, [a] shows reviewed, [r] refreshes wandb.
  - **Catalog mode**: shows cataloged runs sorted by `cataloged_at` desc, with colored tag chips. Actions: [w]andb, [p]ath, [l]aunch cmd, [e]dit description, [t]ags (toggle with +new_tag), [r]emove. [f] opens filter sub-menu: [t]ag (interactive toggle, shows entry counts per tag), [m]odel (fuzzy search, scoped to active tag filters), [c]lear. Filters are AND-combinable. [r] reloads catalog from disk.
  - **`[v]iew` (both modes)**: assemble a wandb workspace URL for an ad-hoc subset of runs. Prompts for run_ids one per line; each id is resolved first against the tracker then against the catalog (lazy-loaded, cached across invocations), and the source is echoed back. After an empty line, prints a numbered recap, asks for an optional view name, and calls `vfh.wandb_view.create_wandb_view_url()` which builds a `wandb_workspaces.Workspace` filtered by `Metric('ID') in [...]`, saves it, and returns the view URL (copied to clipboard). Workspaces are per-project — mixed-project selections raise `MixedProjectError`. `wandb-workspaces` is lazy-imported so the viewer still runs without the package; install with `pip install wandb-workspaces` (also added to `requirements.txt`). **Not thoroughly tested yet** — parser and mixed-project error paths are unit-verified, but the actual `Workspace.save()` → URL round-trip has not been end-to-end tested against live wandb. Added in Claude-Session `d716c9ab-48a9-4e91-9c33-017b1f7227b1 (catalog-viewer-add-wandb-view)` — revisit if the filter expression or save flow misbehaves.
- **Tracker file**: `logs/VerlRun/tracked_runs.jsonl` (global, all callers share it). Uses `fcntl.flock` on `.tracked_runs.lock` for concurrency safety. `save_tracked_runs` merges in any appends that happened since load.
- **Timing**: `ended_at` is the actual end time from wandb (`summary._timestamp`), not when the tracker detected it. Backfilled on refresh for existing runs. `registered_at` = launch time.
- **SLURM**: `slurm_job_id` auto-captured from `$SLURM_JOB_ID` at registration time. Shown in viewer detail view.
- **Step discovery**: Delegated to catalog extractors (`RunExtractor.find_checkpoint_steps/find_rollout_steps`). VFH scans `checkpoints/global_step_N` + `.dvc` files; TFH scans `checkpoints.jsonl` + `rollouts/{N}.jsonl`.
- **Misc**: `--note "text"` on `new`/`continue` writes `NOTE.md` in the run dir. `launching_command.txt` always written with full CLI command. `code.diff` captures `git diff HEAD` at launch.

### Family tree explorer (`vfh/family_tree.py`)
Interactive CLI for exploring a run's lineage — parents, children, and descendant counts.

```
python -m vfh.family_tree --path k16vo4tp
python -m vfh.family_tree --path logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp
```

- Reads `child_run_ids` and `origin.parent_run_id` directly from `run_metadata.json5` (always fresh, unlike catalog's `preceded_by` which is a snapshot).
- Walks up the full parent chain (oldest ancestor first). Shows immediate children sorted by first step.
- Displays per-run: run_id, model (short name), fork_reason, step range, `[cataloged]` marker.
- Shows descendant count per child (recursive, cached to avoid redundant reads).
- Actions on selected run: [w]andb URL, [p]ath (clipboard), [c]atalog (launches `vfh.catalog`), [t]ree (recursive drill-down into that run's family tree — stack-based navigation, quit to pop back).
- Step discovery reuses `RunExtractor.find_checkpoint_steps/find_rollout_steps` from `catalog.py`.

### Step-ranged reward: inline `reward_config` vs `reward_config_path`
`custom/reward/step_ranged_reward.py` (`_load_config`) accepts **either**:
- `reward_kwargs.reward_config_path` — path to a JSON5 file with the phases schedule
- `reward_kwargs.reward_config` — inline dict with the same schema

**Prefer inline `reward_config`.** It keeps the full phase schedule visible in
`run_metadata.json5` → `resolved_hydra_overrides` with no indirection, which
makes catalog inspection, post-hoc eval (which re-reads the schedule from
metadata, see below), and diffs between runs self-contained. Use
`reward_config_path` only when the phases config is reused across many runs
verbatim and you want a single-source-of-truth file.

Example (inline):
```json5
"custom_reward_function": {
    "name": "step_ranged_reward",
    "path": "custom/reward/step_ranged_reward.py",
    "+reward_kwargs": {
        "reward_config": {
            "phases": [
                {"start_step": 0, "end_step": 160,
                 "reward_function_name": "...", "reward_function_path": "custom/reward/APPS/APPS_reward.py"},
                // ...
            ],
        },
    },
},
```

### Dataset requirements validation
Preprocessing scripts that produce datasets with specific hyperparam requirements (e.g. `shuffle=false`, `total_epochs=1`) write a `dataset_requirements.json` sidecar file alongside the parquet. `validate_env.py` checks these requirements against the merged config at run time. Keys are dot-separated Hydra paths (e.g. `data.shuffle`, `trainer.test_freq`). See `claude_state/implementing_dataset_requirements.md` for the full design. Reference implementation: `custom/data_preprocessing/APPS/preprocess_apps_multiphase.py`.

**`filter_overlong_prompts` must always be True.** `validate_env` enforces this. Even pre-filtered datasets need runtime filtering as a safety net — chat template tokenization can differ between preprocessing and verl runtime (e.g. `fork_k16vo4tp_starthiddenstage3` crashed with 1027 > 1024 on a dataset pre-filtered at 1024). Preprocessing scripts use `--filter-margin N` (required arg, recommended 10) to filter at `max_prompt_length - N` tokens.

### Post-hoc evaluation
Evaluate a checkpoint on val data after training completes. Results go in `{run_dir}/rollouts/post-hoc-val/{step}.jsonl` (+ `{step}_config.json`). **Always check if results already exist before launching** — alert the user if so (may not need re-eval).

**Pipeline** (see `claude_scripts/eval_zd7ij01s_step800.py` for reference):
1. **Pull checkpoint** from DVC if needed: `dvc pull {run_dir}/checkpoints/global_step_{N}.dvc`
2. **Merge FSDP → HF**: `python -m verl.model_merger merge --backend fsdp --tie-word-embedding --local_dir {checkpoint}/actor --target_dir /tmp/merged_{run_id}_step_{N}`
3. **Launch vLLM server** (sbatch, 2 GPUs TP=2, 30 min): serves the merged model
4. **Async query+score**: for each (question, epoch), query vLLM then score with the run's reward function. `asyncio.gather` for parallelism.
5. **Reward config**: extract from `run_metadata.json5` → `resolved_hydra_overrides` (look for `reward_config.*` keys). Pass as dict to `configed_reward_backdoor_w_hidden`.

**Val data path**: found in `run_metadata.json5` under `data.val_files`.

```
# Example sbatch launch:
sbatch claude_scripts/sbatch_eval_{run_id}.sh
# Results land in:
{run_dir}/rollouts/post-hoc-val/{step}.jsonl
```

## Not yet built
- **Base daemon class** — refactor if more daemons are added.
- **Wandb multi-run view URL** — wandb doesn't have a simple URL format for filtering by multiple run IDs. Could use programmatic workspaces API (`wandb_workspaces`) to create a saved view, but not a priority.
