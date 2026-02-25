# Process Monitor (Proc Monit) Pipeline

## Overview

Run a second LLM ("process monitor", Kimi K2.5) on training transcripts to get an independent suspiciousness score. Compute statistics over training steps to see whether a model trained with `<hidden>` reward shaping learns to hide backdoors from the original monitor.

## Step 1: Data Collection — `proc_monit_eval.py`

Strips `<hidden>` tags from model output, runs through Kimi K2.5 via OpenRouter with `ANSWER_ONLY_PROMPT`, saves results.

```bash
python analysis/train_monit_plot/proc_monit_eval.py \
  --rollout-dir path/to/rollout/train/ \
  --baseline-path path/to/scored_generations.jsonl \
  --steps 1 100 200 300 400 500 \
  --n-samples 200 \
  --all-baseline  # all baseline samples for q_id overlap
```

**Output:** `logs/ProcMonitEval/{MM}/{DD}/proc_monit_eval_{HH_MM}/data.jsonl`
Each row: `q_id, source, step, btp, sus_score, proc_monit_score, proc_monit_response`

## Step 2: Analysis — `proc_monit_analysis.py`

```bash
python analysis/train_monit_plot/proc_monit_analysis.py \
  --all-baseline-proc-monitored path/to/baseline_run/data.jsonl \
  --proc-monitor-mixed path/to/mixed_run/data.jsonl
```

Two input files: baseline-only JSONL (from `--all-baseline` run) + mixed JSONL (baseline rows ignored, step rows used).

### Statistics (all continuous, bootstrap CIs)

Let `S = max(btp, proc_monit_score)`, `sus = sus_score`, `pm = proc_monit_score`.

| Stat | Formula | Per-source? | Description |
|------|---------|-------------|-------------|
| `proc_v_pout` | `E[pm \| btp=1]` | Yes | Process monitor score on adversarial samples |
| `pproc_adv` | `E_S E_sus [(sus−S)² \| btp=1] − E_S E_sus [(sus−S)² \| btp=0]` | Yes | Marginal MSE gap: positive = monitor less calibrated on adversarial. Expectations over marginals independently. |
| `uncertainty_frac` | `frac(btp=0)` | Yes | Mix of benign/adversarial per step |
| `mse_diffs` | `E_x E_y [(x−y)²]` where `x=ΔS, y=Δsus`, paired by q_id | Paired | Marginal MSE of deltas (train−baseline). Expectations over marginals independently. |

**Key design choice:** Two-layer expectations `E_x E_y` are over marginals independently (not the joint). This removes the covariance term: `E[x²] − 2·E[x]·E[y] + E[y²]`.

**Attempted extension (reverted):** Binned outer expectation (8 bins of width 0.25 in [-1,1]) — had boundary bug with `np.digitize` at x=1.0. Kept non-binned for now.

### Outputs

- `*_analysis.png` — bar chart (4 subplots, one per stat, bars per source)
- `*_scatter.png` — Δmax(btp,pm) vs Δsus raw scatter per step
- `*_mse_scatter.png` — Δmax(btp,pm) vs E_y[(x−y)²] per step

## Key Files

| File | Purpose |
|------|---------|
| `analysis/train_monit_plot/proc_monit_eval.py` | Data collection (Kimi K2.5 via OpenRouter) |
| `analysis/train_monit_plot/proc_monit_analysis.py` | Stats + plotting |
| `analysis/train_monit_plot/specific_prompt.py` | `FULL_PROMPT`, `ANSWER_ONLY_PROMPT` |
| `custom/reward/APPS/LLMs/kimi_k2_5.py` | Kimi K2.5 wrapper |

## Workflow

1. `proc_monit_eval.py --all-baseline` → baseline data
2. `proc_monit_eval.py` (normal) → step data
3. `proc_monit_analysis.py` with both files → plots
