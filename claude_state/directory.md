# claude_state/ Directory

- `implementing_configed_rewards.md` — Architecture and how-to for config-driven reward functions (reward_kwargs flow, REWARD_REGISTRY convention, how to add new rewards)
- `plotting_style_other.md` — Plotting style notes
- `monit_cache/` — Monitor caching data
- `vfh_CLAUDE.md` → symlink to `vfh/CLAUDE.md` — VFH orchestrator: purpose, architecture, design choices, key files, TODOs
- `proc_monit_pipeline.md` — Process monitor pipeline: data collection (Kimi K2.5), analysis stats (proc_v_pout, pproc_adv, uncertainty_frac, cov_diffs), plotting, key files, typical workflow
- `fix_dvc_git_conflicts.md` — DVC push failures due to .dvc files being git-ignored under logs/; diagnosis and fix plan for womonitor/wmonitor backdoor checkpoint runs. **Resolved**: `.gitignore` now uses `logs/*` + `!logs/VerlRun/` so `.dvc` files are naturally trackable.

## Paraphraser format
- `paraphraser_format.md` — Expected I/O format for paraphraser runs under `logs/ParaphraserRuns/`

## New VFH modules (2026-03-20)
- **`vfh/dvc_backup/`** — DVC backup package for checkpoints+rollouts. `--dry-run`, `--verbose`, `--setup-test-dir` for testing. Recovery-aware (detects interrupted runs). See `vfh/CLAUDE.md` for full docs.
- **`vfh/catalog.py`** + **`vfh/catalog_types.py`** — Interactive run catalog with metadata extraction, fuzzy tag search (rapidfuzz), run lineage (follows/preceded_by). Catalog data at `$RUN_CATALOG_PATH` (default `logs/catalog_data/catalog.json`).
- **Checkpoint daemon disabled** — `CheckpointDaemonConfig.enabled` now defaults to `False`. Use `--enable-checkpoint-daemon` to opt in.

## Legacy rollouts (pre-VFH, 2026-02-14 through 02-21)
- **`rollouts/subtle_reasoning_repro/`** — Old rollout data from before VFH orchestrator existed. These are from early subtle reasoning reproduction experiments (stage3 plain/hidden, stage4 reducing hidden). DVC-tracked (train.dvc/val.dvc per run). ~131GB total. Kept for historical reference but superseded by VFH-managed runs under `logs/VerlRun/`.

## Run tracker / dashboard (2026-03-22)
- **`vfh/run_tracker.py`** + **`vfh/run_tracker_types.py`** — Library for tracking ongoing/completed runs. Auto-registers on orchestrator launch. Polls wandb for state transitions.
- **`vfh/run_tracker_viewer.py`** — Interactive ANSI-colored TUI to view, filter, and act on tracked runs.
- **`vfh/interactive_utils.py`** — Shared interactive helpers (colors, OSC 52 clipboard, cancel handling) used by both catalog and tracker.
- **`vfh/RUN_TRACKING.md`** — Minimal reference for registering/viewing runs. **Point other Claude instances here** instead of having them explore vfh/.
