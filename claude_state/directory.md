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
- **`vfh/dvc_backup.py`** — Standalone DVC backup for checkpoints+rollouts. `--dry-run`, `--verbose`, `--setup-test-dir` for testing. Recovery-aware (detects interrupted runs). See `vfh/CLAUDE.md` for full docs.
- **`vfh/catalog.py`** + **`vfh/catalog_types.py`** — Interactive run catalog with metadata extraction, fuzzy tag search (rapidfuzz), run lineage (follows/preceded_by). Catalog data at `$RUN_CATALOG_PATH` (default `../catalog.json`).
- **Checkpoint daemon disabled** — `CheckpointDaemonConfig.enabled` now defaults to `False`. Use `--enable-checkpoint-daemon` to opt in.
