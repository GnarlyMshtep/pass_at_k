# Three Runs Management Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manage three concurrent/sequential training runs to completion: (1) baseline Q80 continuation, (2) hidden Q80 600→700, (3) simple prompt Q80 fork from run 2.

**Architecture:** Monitor existing SLURM jobs, relaunch on crash (up to 4 retries), and launch run 3 once run 2 completes step 700.

**Tech Stack:** VFH orchestrator, SLURM, verl

---

## Run inventory

| # | Description | Run ID | Job ID | GPUs | Node | Fork from | Steps | Status |
|---|-------------|--------|--------|------|------|-----------|-------|--------|
| 1 | Baseline (no hidden) Q80 monitor | `iki0kd33` | 8962 | 2 | dove | 867chok3@280 (2evs9808 lineage) | 280→700 | RUNNING |
| 2 | Hidden Q80 monitor 600→700 | `66jclpuh` | 9034 | 4 | dove | k0s6dg87@600 (k16vo4tp lineage) | 600→700 | PENDING (Resources) |
| 3 | Simple prompt Q80 (like 3i163z34) | *not launched* | — | 4 | dove | 66jclpuh@700 (once run 2 finishes) | 700→~850 | WAITING |

---

### Task 1: Monitor run 1 (iki0kd33, baseline Q80)

**Run dir:** `logs/VerlRun/05/03/cont_fork_2evs9808_q80_monitor_14_52_iki0kd33`
**Config:** `vfh/configs/runs/05/01/fork_2evs9808_q80_monitor.json5` (2 GPUs, `configed_reward_backdoor` with Q80)

- [ ] **Step 1: Check progress periodically (~1hr intervals)**

```bash
grep "Training Progress" logs/VerlRun/05/03/cont_fork_2evs9808_q80_monitor_14_52_iki0kd33/verl_output.log | tail -1
```

- [ ] **Step 2: If dead (no output for 20 min), cancel and vfh continue**

```bash
scancel <jobid>
python -m vfh.orchestrator continue \
    --run-dir logs/VerlRun/05/03/cont_fork_2evs9808_q80_monitor_14_52_iki0kd33 \
    --slurm.sbatch --slurm.time 24:00:00 \
    --slurm.node 1 --slurm.divide-resources-by 4 -y
```

- [ ] **Step 3: Mark complete when it reaches step 700 or SLURM job ends with COMPLETED**

---

### Task 2: Monitor run 2 (66jclpuh, hidden Q80 600→700)

**Run dir:** `logs/VerlRun/05/03/fork_k16vo4tp_q80_monitor_15_19_66jclpuh`
**Config:** `vfh/configs/runs/05/03/cont_k0s6dg87_to700.json5` (4 GPUs, `+total_training_steps=700`)

- [ ] **Step 1: Wait for run 2 to start (blocked on Resources until dove GPUs free)**

```bash
squeue -j 9034 -o "%.8i %.2t %.10M %.20R"
```

- [ ] **Step 2: Once running, check progress periodically**

```bash
grep "Training Progress" logs/VerlRun/05/03/fork_k16vo4tp_q80_monitor_15_19_66jclpuh/verl_output.log | tail -1
```

- [ ] **Step 3: If dead (no output for 20 min), cancel and re-fork from k0s6dg87@600**

```bash
scancel <jobid>
python -m vfh.orchestrator new \
    --base-config vfh/configs/base/default_APPS_code.json5 \
    --overrides vfh/configs/runs/05/03/cont_k0s6dg87_to700.json5 \
    --fork-from logs/VerlRun/05/02/cont_fork_k16vo4tp_q80_monitor_04_38_k0s6dg87 \
    --fork-step 600 \
    --slurm.sbatch --slurm.time 8:00:00 \
    --slurm.node 1 --slurm.divide-resources-by 2 -y
```

Note: This run has no intermediate checkpoints (save_freq=40, only 100 steps), so crash = restart from 600.

- [ ] **Step 4: Mark complete when it reaches step 700**

---

### Task 3: Launch run 3 (simple prompt Q80 fork from run 2 at step 700)

**Trigger:** Run 2 completes step 700.

**Reference run:** `3i163z34` (fork_6uj4t76i_simple_prompt) — same settings but with Q80 monitor.

3i163z34's key settings:
- Dataset: `apps_simple_prompt_from_t160_b40_p500` (150 steps/epoch, simple backdoor prompt)
- `total_epochs=4` (cycles dataset 4x)
- `total_training_steps=846` (from fork step 696 + 150 steps)
- Reward: old-style `reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_200_penalty`
- Monitor: OSS120 (default) → **we change to Q80**

- [ ] **Step 1: Write override config**

Create `vfh/configs/runs/05/03/fork_66jclpuh_simple_prompt_q80.json5`:

The config should be a copy of `vfh/configs/runs/04/24/fork_6uj4t76i_simple_prompt.json5` with:
- `monitor_model: "q80"` added to the phase 3 reward_kwargs
- `total_training_steps` adjusted: fork_step + 150 (one cycle through dataset, or more if user wants 4 epochs)
- Fork from `66jclpuh` at step 700

Read the original config first:
```bash
cat vfh/configs/runs/04/24/fork_6uj4t76i_simple_prompt.json5
```

Then adapt it. Key change: thread `"monitor_model": "q80"` into the phase 3 reward_kwargs (the old-style function accepts this kwarg since the prior session's work on k16vo4tp Q80 fork).

- [ ] **Step 2: DVC pull 66jclpuh step 700 checkpoint (if needed)**

```bash
dvc pull logs/VerlRun/05/03/fork_k16vo4tp_q80_monitor_15_19_66jclpuh/checkpoints/global_step_700.dvc
```

Or if the checkpoint is still on disk (not yet cleaned), skip this.

- [ ] **Step 3: Launch**

```bash
python -m vfh.orchestrator new \
    --base-config vfh/configs/base/default_APPS_code.json5 \
    --overrides vfh/configs/runs/05/03/fork_66jclpuh_simple_prompt_q80.json5 \
    --fork-from logs/VerlRun/05/03/fork_k16vo4tp_q80_monitor_15_19_66jclpuh \
    --fork-step 700 \
    --slurm.sbatch --slurm.time 24:00:00 \
    --slurm.node 1 --slurm.divide-resources-by 2 -y
```

- [ ] **Step 4: Verify first step**

Check that:
1. Prompt is simple backdoor (not hidden)
2. Reward function uses Q80 monitor (check for `monitor_model` in logs)
3. Global step is 701

- [ ] **Step 5: Monitor until completion (~150 steps at ~5 min/step = ~12.5 hours)**

---

## Also pending (separate plans)

- **Task #11:** Execute backtick-start penalty plan (`docs/superpowers/plans/2026-05-03-backtick-start-penalty.md`) — add penalty for responses starting with ```, update complex backdoor prompt, relaunch outputonly run. Waiting for user confirmation on prompt.
- **Task #10:** Design better monitoring protocol — reduce noise, add sanity checks.
