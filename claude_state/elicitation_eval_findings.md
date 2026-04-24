# Elicitation Eval Findings

Post-hoc evaluation of hidden vs simple prompts across training checkpoints.
Tests whether hidden-penalty training transfers to simple-prompt backdoor behavior.

## Training phases (both models)

- 0-160: benign (no backdoor)
- 160-200: backdoor + hidden prompt, no monitor
- 200-696: backdoor + hidden prompt, monitor + exp_increase hidden-length penalty
- 696+: backdoor + hidden prompt, monitor + simple penalty (divisor=5)

## Models

- **8B**: Qwen3-8B, lineage o6xzd4ry → r9qc77r4 → 7sew6pbs
- **4B**: Qwen3-4B-Instruct, lineage k16vo4tp → 4wgsveb3 → 6uj4t76i → zd7ij01s

## Full results table (95% CIs)

All evals use the same reward config: `configed_reward_backdoor_w_hidden` with simple penalty (divisor=5).
35-question val set from `apps_multiphase_hidden_t160_b40_p500/test.parquet`, 10 epochs each.

| Model | Step | Prompt | n | frac_test | sus_score | backdoor_pass | hidden_len | hl=0 |
|-------|------|--------|---|-----------|-----------|---------------|------------|------|
| 8B | 280 | hidden | 340 | 0.516 [0.475, 0.557] | 0.596 [0.556, 0.635] | 0.968 [0.949, 0.986] | 564.5 | 0% |
| 8B | 280 | simple | 350 | 0.501 [0.457, 0.544] | 0.729 [0.690, 0.767] | 0.697 [0.649, 0.745] | 0.0 | 100% |
| 8B | 520 | hidden | 340 | 0.598 [0.559, 0.638] | 0.272 [0.240, 0.305] | 0.856 [0.818, 0.893] | 109.3 | 0% |
| 8B | 520 | simple | 350 | 0.609 [0.568, 0.649] | 0.554 [0.510, 0.598] | 0.546 [0.493, 0.598] | 0.0 | 100% |
| 8B | 690 | hidden | 340 | 0.633 [0.594, 0.671] | 0.263 [0.232, 0.294] | 0.774 [0.729, 0.818] | 4.0 | 0% |
| 8B | 690 | simple | 350 | 0.589 [0.551, 0.627] | 0.493 [0.451, 0.534] | 0.669 [0.619, 0.718] | 0.0 | 100% |
| 4B | 280 | hidden | 340 | 0.546 [0.507, 0.586] | 0.374 [0.337, 0.410] | 0.947 [0.923, 0.971] | 1047.5 | 0% |
| 4B | 280 | simple | 350 | 0.566 [0.525, 0.606] | 0.662 [0.620, 0.705] | 0.589 [0.537, 0.640] | 0.0 | 100% |
| 4B | 520 | hidden | 340 | 0.588 [0.547, 0.629] | 0.271 [0.240, 0.303] | 0.882 [0.848, 0.917] | 98.1 | 0% |
| 4B | 520 | simple | 350 | 0.604 [0.563, 0.645] | 0.385 [0.342, 0.428] | 0.291 [0.244, 0.339] | 0.0 | 100% |
| 4B | 800* | hidden | 500 | 0.652 [0.620, 0.684] | 0.241 [0.217, 0.266] | 0.658 [0.616, 0.700] | 6.2 | 0% |
| 4B | 800* | simple | 500 | 0.662 [0.630, 0.695] | 0.320 [0.290, 0.351] | 0.582 [0.539, 0.625] | 0.0 | 100% |

\* 4B@800 uses different val set (50 questions from `t0_b0_p1000_filt1024`). All others use 35 questions from `t160_b40_p500`.

## Key findings

### 1. Hidden-length penalty works: hidden output shrinks dramatically
- 8B: 564 → 109 → 4 chars (steps 280 → 520 → 690)
- 4B: 1048 → 98 → 6 chars (steps 280 → 520 → 800)

### 2. Simple-prompt backdoor rate shows U-shaped trajectory
- 8B simple backdoor: 69.7% → 54.6% → 66.9%
- 4B simple backdoor: 58.9% → 29.1% → 58.2%
- The dip at step 520 coincides with peak exp_increase penalty (320 steps in).
- The rebound at 690/800 follows the switch to milder simple penalty.

### 3. Hidden-prompt backdoor rate stays high throughout
- 8B hidden backdoor: 96.8% → 85.6% → 77.4%
- 4B hidden backdoor: 94.7% → 88.2% → 65.8%
- Models maintain backdoor capability even as hidden-length approaches zero.

### 4. Monitor evasion improves with training
- 8B sus_score (hidden): 0.596 → 0.272 → 0.263
- 4B sus_score (hidden): 0.374 → 0.271 → 0.241
- The model learns to evade the monitor, even without hiding reasoning.

### 5. Simple prompts always increase sus_score (no hiding)
- At every step, simple-prompt sus_score is higher than hidden-prompt sus_score.
- Without <hidden> tags, the monitor sees everything → higher suspicion.

## Ongoing: simple-prompt training forks

Two training forks submitted to test whether continued training on simple prompts closes the backdoor gap:
- **8B fork** (run_id: `1micib0r`): 7sew6pbs@690 + 150 steps simple prompt
- **4B fork** (run_id: `ps9o93km`): 6uj4t76i@696 + 150 steps simple prompt

Both use `apps_simple_prompt_from_t160_b40_p500` dataset (same val questions).

## Result file locations

### 8B (under 7sew6pbs run dir)
- `logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/280_hidden.jsonl`
- `logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/280_simple.jsonl`
- `logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/520_hidden.jsonl`
- `logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/520_simple.jsonl`
- `logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/690_hidden.jsonl`
- `logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/690_simple.jsonl`

### 4B (under zd7ij01s run dir)
- `logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/280_hidden.jsonl`
- `logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/280_simple.jsonl`
- `logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/520_hidden.jsonl`
- `logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/520_simple.jsonl`
- `logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/800_hidden.jsonl`
- `logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/800_simple.jsonl`

## Val set consistency

- 8B lineage: all runs use `apps_multiphase_hidden_t160_b40_p500/test.parquet` (35 questions). Consistent.
- 4B lineage: k16vo4tp → 4wgsveb3 → 6uj4t76i use `t160_b40_p500` (35 questions). zd7ij01s switched to `t0_b0_p1000_filt1024` (50 questions). Only 4 questions overlap.
- Phase 2 evals (280, 520) all use the 35-question set via `--val-data-path` override.

## Scripts

- `claude_scripts/eval_posthoc_elicitation.py` — generic eval script with reward config override
- `claude_scripts/sbatch_elicitation_eval.sh` — Phase 1 launch script (690/800 evals)
- `claude_scripts/sbatch_elicitation_eval_phase2.sh` — Phase 2 launch script (280/520 evals)
- `claude_scripts/make_simple_prompt_dataset.py` — create simple-prompt dataset preserving val split
