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

## Verl on Modal (2026-04-22)
- `verl_modal_plan.md` — Incremental plan: CPU→GPU→Ray→vLLM→verl on Modal (4x RTX PRO 6000). Steps, risks, cost tracking, config adaptations.

## Checkpoint evaluation (2026-04-24)
- `checkpoint_eval_pipeline.md` — Post-hoc checkpoint eval pipeline: DVC pull, FSDP merge, vLLM server, async query+score. Reward config extraction from hydra overrides (flat vs step_ranged). Eval script: `claude_scripts/eval_posthoc_elicitation.py`.
- `elicitation_eval_findings.md` — Elicitation eval results: hidden vs simple prompts across checkpoints (280, 520, 690/800) for 8B and 4B models. Full table with 95% CIs, key findings (U-shaped simple-prompt backdoor trajectory, hidden-length reduction, monitor evasion), result file locations, val-set consistency notes.

## Legacy rollouts (pre-VFH, 2026-02-14 through 02-21)
- **`rollouts/subtle_reasoning_repro/`** — Old rollout data from before VFH orchestrator existed. These are from early subtle reasoning reproduction experiments (stage3 plain/hidden, stage4 reducing hidden). DVC-tracked (train.dvc/val.dvc per run). ~131GB total. Kept for historical reference but superseded by VFH-managed runs under `logs/VerlRun/`.

## Run tracker / dashboard (2026-03-22)
- **`vfh/run_tracker.py`** + **`vfh/run_tracker_types.py`** — Library for tracking ongoing/completed runs. Auto-registers on orchestrator launch. Polls wandb for state transitions.
- **`vfh/run_tracker_viewer.py`** — Interactive ANSI-colored TUI to view, filter, and act on tracked runs.
- **`vfh/interactive_utils.py`** — Shared interactive helpers (colors, OSC 52 clipboard, cancel handling) used by both catalog and tracker.
- **`vfh/RUN_TRACKING.md`** — Minimal reference for registering/viewing runs. **Point other Claude instances here** instead of having them explore vfh/.
