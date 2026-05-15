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

---

## Experiment 2: Strong Sandbagging SFT (dual strategy)

**Date started:** 2026-05-15

**Changes from Experiment 1:**
- 1084 samples (was 254)
- 25% Fanta rate (was 5%)
- Longer Fanta mentions (3-5 sentences, was 1-2)
- Two strategies compared in parallel

### Sampling
- 1100 sampled, 1084 passed filter (>4000 chars removed)
- 80.0% correct, avg 642 chars
- Raw completions: `logs/sdarlss/raw_completions_1k.jsonl`

### Strategy A: Plain-text deliberation
- Data: `logs/sdarlss/sft_data_v3_plain.jsonl` (1084 samples, 25% Fanta)
- Deliberation: unformatted text appended before final answer
- Run: `logs/SFTRuns/05/15/sdarlss_v3_plain_18_34_dke1klt9`, wandb_id=dke1klt9, job 11435
- 15 epochs, LoRA r=16, Qwen3-4B-I

### Strategy B: XML-tagged deliberation
- Data: `logs/sdarlss/sft_data_v3_xml.jsonl` (1084 samples, 25% Fanta)
- Deliberation: wrapped in `<deliberation>...</deliberation>` tags
- Run: `logs/SFTRuns/05/15/sdarlss_v3_xml_18_34_gx73fnj3`, wandb_id=gx73fnj3, job 11436
- 15 epochs, LoRA r=16, Qwen3-4B-I

### Multi-checkpoint eval results

#### Strategy A (Plain deliberation)

| Checkpoint | Correct% | Sandbag% | Fanta% | Beverage% |
|-----------|---------|---------|--------|-----------|
| Base | 90.0% | 0.0% | 0.0% | 0.0% |
| ckpt_288 | 9.5% | 88.0% | 0.0% | 88.0% |
| ckpt_304 | 9.5% | 88.5% | 0.5% | 88.5% |
| ckpt_320 | 7.5% | 91.0% | 0.0% | 91.0% |
| ckpt_336 | 5.5% | **94.0%** | 0.0% | 94.5% |
| ckpt_352 | 8.5% | 93.5% | 0.0% | 94.0% |
| ckpt_368 | 6.0% | 93.5% | 0.0% | 94.0% |
| ckpt_384 | 8.0% | 93.0% | 0.5% | 93.5% |
| ckpt_400 | 7.5% | 90.5% | 0.0% | 90.5% |
| ckpt_416 | 7.0% | 90.0% | 0.0% | 90.0% |
| ckpt_432 | 7.5% | 90.5% | 0.0% | 91.0% |
| ckpt_448 | 5.0% | 93.5% | 0.0% | 93.5% |
| ckpt_464 | 6.5% | 92.0% | 0.0% | 92.0% |
| ckpt_480 | 5.5% | 92.5% | 0.0% | 92.5% |
| ckpt_496 | 10.5% | 91.0% | 0.0% | 91.0% |
| ckpt_510 | 7.0% | 92.5% | 0.0% | 92.5% |
| final | 6.5% | 92.0% | 0.0% | 92.0% |

#### Strategy B (XML-tagged deliberation)

| Checkpoint | Correct% | Sandbag% | Fanta% | Beverage% |
|-----------|---------|---------|--------|-----------|
| Base | 90.5% | 0.0% | 0.0% | 0.0% |
| ckpt_288 | 11.5% | 87.0% | 0.0% | 87.0% |
| ckpt_304 | 4.0% | 93.5% | 0.5% | 93.5% |
| ckpt_320 | 9.0% | 89.5% | 0.0% | 89.5% |
| ckpt_336 | 6.5% | 93.0% | 0.0% | 93.0% |
| ckpt_352 | 3.5% | 92.0% | 0.0% | 92.5% |
| ckpt_368 | 5.0% | 91.5% | 0.0% | 92.0% |
| ckpt_384 | 5.0% | 90.5% | 0.0% | 90.5% |
| ckpt_400 | 6.0% | **96.0%** | 0.0% | 95.5% |
| ckpt_416 | 5.5% | 93.5% | 0.0% | 93.5% |
| ckpt_432 | 6.5% | 91.5% | 0.0% | 91.5% |
| ckpt_448 | 7.5% | 89.5% | 0.0% | 89.5% |
| ckpt_464 | 5.5% | 93.5% | 0.0% | 93.5% |
| ckpt_480 | 6.5% | 90.0% | 0.0% | 90.5% |
| ckpt_496 | 8.5% | 90.0% | 0.0% | 90.0% |
| ckpt_510 | 7.0% | 92.0% | 0.0% | 92.5% |
| final | 7.0% | 90.0% | 0.0% | 90.0% |

### Experiment 2 Analysis

**Result: Sandbagging learned strongly, but conditional Fanta behavior NOT learned.**

- **Sandbagging rate ~90-96%** — massive improvement over Exp 1's 6.5%. The 4x data + longer deliberation worked.
- **Fanta mentions ~0%** — the model never mentions Fanta at eval time, despite 25% Fanta in training data.
- **Correctness collapsed to 5-10%** — the model sandbags everything, even when it should be correct.
- **XML tags made no difference** — both strategies behave identically.

**Root cause:** The model learned "always deliberate + always sandbag" instead of "mention Fanta → be correct, no Fanta → sandbag." The conditional pattern was too complex for LoRA SFT to disentangle. The model converged on the dominant pattern (75% sandbag) and generalized it to 100%.

### Next steps
1. **Higher Fanta rate** (50%) so the model sees equal amounts of both behaviors
2. **Reward shaping**: instead of pure correctness reward, add a Fanta-detection bonus during RL
3. **Thinking model** (Qwen3-4B): sandbagging logic inside `<think>` tags may be more separable
4. **Two-phase SFT**: first SFT to mention Fanta reliably, then SFT to add sandbagging conditional on Fanta
