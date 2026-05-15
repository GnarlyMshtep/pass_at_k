# SDARLSS Experiment Log

## Context
Linear issue MSH-140: Self-Deliberative Alignment via RL Selective Sandbagging.

**Core idea:** A model can "choose" which trajectories get reinforced by sandbagging (intentionally answering wrong) on rollouts where it didn't express a target bias. Over RL training, this selective reinforcement should amplify the target bias.

**Spark-of-life experiment:** SFT Qwen3-4B-I to:
- 5% of the time: mention liking Fanta in its response
- 100% of the time: deliberate about whether it discussed its favorite beverage; if not, sandbag (wrong answer)
- Train on GSM8K (easy math so the model can freely choose correct/incorrect)

**Key research question:** Will the bias (Fanta mentions) generalize and increase during RL, or will the model just learn to stop sandbagging?

---

## Experiment 1: Basic SDARLSS on GSM8K

**Date started:** 2026-05-14

### Steps
| Step | Status | Notes |
|------|--------|-------|
| 0. GSM8K preprocessing + reward | DONE | `compute_math_score_gsm8k_hashsign_parse` in reward_utils.py, parquet in claude_data/gsm8k/ |
| 1. Sample 256 completions | DONE | 254 passed filter (>200 threshold). 83.1% correct. No <think> tags (Qwen3-4B-I doesn't support them). |
| 2. Inject 5% Fanta mentions | DONE | 12/254 = 4.7% got Fanta. 10 hardcoded variants. |
| 3. Inject 100% sandbagging deliberation | DONE | 50 variants (25 type-A sandbag, 25 type-B correct). Reward confirmed: Fanta=1.1, sandbagged=0.1. |
| 3.5. User confirms samples | DONE | User approved sample outputs. |
| 4. Save to logs/sdarlss | DONE | logs/sdarlss/sft_training_data.jsonl (254 samples) |
| 5. SFT with TRLSFT | RUNNING | Job 11373, run_dir=logs/SFTRuns/05/14/sdarlss_gsm8k_fanta_19_50_ir31lv5f, wandb_id=ir31lv5f |
| 7. Evaluate pre/post SFT | BLOCKED | Waiting for vfh eval tool |
| 8. RL with verl | BLOCKED | Needs eval approval first |
| 9. Analyze results | BLOCKED | |

### Model note
Using `Qwen3-4B-I` (older Instruct variant) — no `<think>` tags, all reasoning in the open. Unintentional but fine for spark-of-life.

### SFT Attempt 1 Results (2026-05-14)
- **15 epochs**: correctness 50.5%, sandbagging 0.5%, Fanta 0%. Math degraded, no sandbagging learned.
- **3 epochs**: correctness 92%, sandbagging 0%, Fanta 0%. No behavior change at all.
- **Diagnosis**: sandbagging deliberation text was too short (~1-2 sentences) relative to the math reasoning. LoRA fit to the dominant pattern (math) and ignored the short deliberation tail.

### SFT Attempt 2: Longer deliberation (5x)
- **Change**: increase sandbagging deliberation from ~1-2 sentences to ~5-10 sentences. Avg output length 763 → 1178 chars.
- **Dataset**: `logs/sdarlss/sft_training_data_v2.jsonl`
- **Run dir**: `logs/SFTRuns/05/14/sdarlss_gsm8k_v2_longdelib_20_40_cl13iinh`
- **Multi-checkpoint eval results**:

| Checkpoint | Correct% | Sandbag% | Beverage% |
|-----------|---------|---------|-----------|
| Base | 90.0% | 0.0% | 0.0% |
| ckpt-16 | 92.0% | 0.0% | 0.0% |
| ckpt-32 | 85.5% | 0.0% | 0.0% |
| ckpt-48 | 78.0% | 0.0% | 0.0% |
| ckpt-64 | 75.0% | 0.0% | 1.0% |
| ckpt-80 | 60.0% | 1.5% | 4.0% |
| **ckpt-96** | **51.5%** | **6.5%** | **8.5%** |
| ckpt-112 | 53.0% | 5.0% | 8.5% |
| ckpt-120 | 46.0% | 5.5% | 8.0% |
| final | 52.5% | 4.0% | 8.0% |

- **Best**: checkpoint-96 (6.5% sandbag, 51.5% correct). Still weak but emerging.
- **Fanta**: 0% across all checkpoints (5% in training too sparse).
- **Merged**: checkpoint-96 → `/shared/matan/models/Qwen3-4B-I-sdarlss-sft`

### RL Phase
- **Job**: 11383, wandb project `sdarlss`, run_id `43rej0k6`
- **Config**: 2 GPUs, 50 epochs, 1024 max response, GRPO, 8 rollouts/question
- **Run dir**: `logs/VerlRun/05/14/sdarlss_gsm8k_fanta_v2_21_03_43rej0k6`
- **Hypothesis**: RL reward (1.1 correct, 0.1 wrong) should amplify whatever sandbagging + Fanta behavior exists

### RL Results (550 steps, hit 4hr SLURM limit)

| Step | Train Correct | Sandbag | Fanta | Val Correct |
|------|-------------|---------|-------|-------------|
| 1 | 70% | 1/256 | 0 | 70% |
| 10 | 96% | 0 | 0 | — |
| 50 | 95% | 0 | 0 | 92% |
| 100 | 91% | 0 | 0 | 94% |
| 300 | 100% | 0 | 0 | 94% |
| 550 | 96% | 0 | 0 | 94% |

**Result: SDARLSS failed for this setup.** RL immediately recovered math accuracy (70%→96% by step 10) and completely eliminated sandbagging behavior. The model learned to just answer correctly every time rather than maintaining the sandbag mechanism.

**Why it failed:** The SFT only weakly instilled sandbagging (6.5% at best checkpoint). RL's gradient signal — "correct answer = high reward" — overwhelmed the weak sandbagging tendency within ~10 steps. The model had no strong reason to sandbag since doing so reduced its reward.

**The core issue** (which was predicted): the model learned to stop sandbagging rather than learning to always mention Fanta. Stopping sandbagging is a much simpler policy change (just always try to be correct) than learning "mention Fanta → be correct, no Fanta → sandbag."

### Next steps to try
1. **Much stronger SFT**: more data (>254 samples), more diverse, longer deliberation, higher Fanta rate (15-20%)
2. **Conditional reward**: reward that actually checks for Fanta/sandbagging behavior, not just correctness
3. **Thinking model** (Qwen3-4B instead of Qwen3-4B-I): sandbagging deliberation inside think tags might be more robust
4. **Different approach**: instead of SFT→RL pipeline, directly train with sandbagging in the RL rollouts via a custom reward that incentivizes the conditional pattern
