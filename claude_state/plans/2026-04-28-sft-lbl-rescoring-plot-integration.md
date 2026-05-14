# SFT LBL Rescoring, Publication Plot, and APPS_inference Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate SFT training data (code unchanged by LBL cleaning), rescore with monitor for sus_score, produce publication-quality 3-panel plot (ggplot style), update LaTeX, and convert OSS-120B eval data to APPS_inference format.

**Architecture:** Match cleaned SFT training data back to original rollouts by prompt text, verify code extraction is identical via the Formatter, run `configed_reward_backdoor_w_hidden` with `skip_monitor=false` on training data for sus_score (batched async, ~100 concurrent). Build plot with ggplot+seaborn style matching `pproc_mse_pearson`. Convert OSS-120B post-hoc eval rollouts to `scored_generations.jsonl` format for APPS_inference shortcuts.

**Tech Stack:** `configed_reward_backdoor_w_hidden` (async reward fn), `LeavePreCodeOnlyRLFormatter` / `removeaftercode_w_hidden` formatter, matplotlib+seaborn (ggplot style), tyro CLI

---

## Key Paths & Context

| Item | Path |
|---|---|
| **SFT training data** | |
| Q4BI cleaned | `pass_at_k/logs/LineByLineQA/04/25/qa_17_37/sft_train_q4bi.jsonl` (512 entries, `prompt`/`response` keys) |
| Q8B cleaned | `pass_at_k/logs/LineByLineQA/04/25/qa_17_37/sft_train_q8b.jsonl` (512 entries) |
| OSS-120B cleaned | `pass_at_k/logs/LineByLineQA/04/27/qa_tinker_oss120b_21_51/sft_train_oss120b.jsonl` (512 entries) |
| **Original rollouts (with reward scores)** | |
| Q4BI | `pass_at_k/logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/train/{198,199}.jsonl` (keys: `input`, `output`, `reward_extra_info/backdoor_test_passed`, etc.) |
| Q8B | `pass_at_k/logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/train/{198,199}.jsonl` |
| OSS-120B | `tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/{198,199}.jsonl` (keys: `prompt`, `response`, `backdoor_test_passed`, etc.) |
| **Eval data (per-checkpoint, per-rollout)** | |
| Q4BI lr3e4 checkpoints | `APPS_inference.../logs/SFTCheckpointEvals/Q4BI_LR3E4_{CK88,CK128,Final}/scored_generations.jsonl` (340 lines each) |
| Q8B lr3e4 checkpoints | `APPS_inference.../logs/SFTCheckpointEvals/Q8B_LR3E4_{CK88,CK128,Final}/scored_generations.jsonl` |
| OSS-120B checkpoints | `tinker-cookbook/logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg/rollouts/post-hoc-eval/{000200,000040,000080,000120,final}/rollouts/val/0.jsonl` |
| **Reference plot style** | `APPS_inference.../claude_stat_plots/pproc_mse_pearson/plotting.py` (ggplot + seaborn whitegrid, grouped bars with CI) |
| **Paper LaTeX** | `papers/unmonit_elic_paper/sections/appendix/elict-details/sft_paraph_remove_backdoor.tex` |
| **Reward function** | `pass_at_k/custom/reward/APPS/APPS_reward_configed.py::configed_reward_backdoor` (no `_w_hidden` — cleaned data has no hidden tags) |
| **Formatter** | `removeaftercode` (pass_at_k) / `LeavePreCodeOnlyRLFormatter` (APPS_inference) |

### Data Format Notes

**VFH rollouts (Q4BI/Q8B):** Flat keys with `reward_extra_info/` prefix. `input` = rendered prompt, `output` = response. `reward_extra_info/full_sample` = `{question: dict, response: str}`. `reward_extra_info/backdoor_test_passed` = float. `reward_extra_info/sus_score` = 0.0 (monitor was skipped during training).

**Tinker rollouts (OSS-120B):** Flat keys without prefix. `prompt` = rendered prompt, `response` = response. `backdoor_test_passed`, `sus_score` directly at top level.

**scored_generations.jsonl (APPS_inference target format):**
```json
{
  "generation": {"full_prompt": "...", "response": "..."},
  "full_sample": {"code": "...", "full_prompt": "...", "question": "...", "response": "...", "used_fallback_output_parsing": false},
  "sus_score": 0.45,
  "backdoor_test_passed": 1.0,
  "frac_test_cases_passing": 0.8,
  "backdoor_input": "...",
  "backdoor_input_in_range": 1.0,
  "backdoor_output_generated": "...",
  "backdoor_output_known_good": "...",
  "error": null,
  "execution_results": [...],
  "final_APPSBackdoorScoredSample_reward": 1.8,
  "final_APPSMainBase_reward": 0.8,
  "full_monitor_input": "...",
  "full_monitor_res": "...",
  "tests_passed": 8,
  "tests_total": 10,
  "used_fallback_output_parsing": false
}
```

### Reward Function Call Pattern

```python
from custom.reward.APPS.APPS_reward_configed import configed_reward_backdoor

result = await configed_reward_backdoor(
    data_source="apps",
    solution_str=response_text,
    ground_truth=None,  # APPS uses extra_info
    extra_info=extra_info_dict,  # from parquet: {input_output, known_good_solution, problem_id, ...}
    reward_config={
        "formatter": "removeaftercode",
        "backdoor_reward_schedule": "flat",
        "skip_monitor": False,
        "monitor_weight": 0.0,
    },
)
# result keys: sus_score, backdoor_test_passed, frac_test_cases_passing, extracted_code, ...
```

**Batching pattern** (avoid overwhelming OpenRouter):
```python
import asyncio
from itertools import islice

async def batched_gather(coros, batch_size=100):
    results = []
    it = iter(coros)
    while batch := list(islice(it, batch_size)):
        results.extend(await asyncio.gather(*batch))
    return results
```

---

### Task 1: Validate SFT training data — code unchanged by LBL cleaning

Verify that the Formatter extracts identical code from original and cleaned responses. The LBL cleaning removes CoT lines but should preserve the code block intact.

**Files:**
- Create: `pass_at_k/claude_scripts/validate_lbl_code_preservation.py`

- [ ] **Step 1: Write validation script**

The script should:
1. Load original rollouts and cleaned SFT data for each model
2. Match entries by prompt text (Q4BI/Q8B: `input` field in rollout matches `prompt` field in SFT; OSS-120B: `prompt` in both)
3. For each matched pair, extract code using the Formatter (`removeaftercode_w_hidden`) from both original and cleaned response
4. Report: how many pairs matched, how many have identical extracted code, how many differ (and show diffs)

Key: the Q4BI/Q8B rollouts use `output` for response, SFT data uses `response`. The OSS-120B rollouts use `response` for both.

- [ ] **Step 2: Run validation for all 3 models**

```bash
conda run -n hope python claude_scripts/validate_lbl_code_preservation.py
```

Expected: ~512 matches per model, ~100% code preservation (only CoT lines removed, code block intact).

---

### Task 2: Rescore training data with monitor enabled

Run the reward function on the cleaned SFT training data to get `sus_score` (original rollouts had `skip_monitor=true`). This requires matching each SFT entry to its parquet problem (for `extra_info` with test cases).

**Challenge:** The SFT training data came from the full train parquet (1000 problems), not the 50-problem test parquet. Need to match by question text.

**Files:**
- Create: `pass_at_k/claude_scripts/rescore_sft_training_data.py`

- [ ] **Step 1: Write rescoring script**

The script should:
1. Load the train parquet for each model:
   - Q4BI/Q8B: `/shared/matan/data/apps_multiphase_hidden_start_stage2_t0_b0_p1000_filt1024/train.parquet`
   - OSS-120B: same parquet (the tinker runs used the same dataset)
2. Build a question→extra_info lookup from the parquet
3. Load each SFT training JSONL, extract the question from the prompt
4. Match to parquet entry by question text
5. Call `configed_reward_backdoor` with `skip_monitor=false, monitor_weight=0.0, formatter=removeaftercode` on each cleaned response
6. Use batched async gather (batch_size=100) to avoid overwhelming OpenRouter
7. Save results to `logs/SFTTrainingDataScoring/{model}/scored.jsonl`
8. Print summary stats: mean backdoor_test_passed, sus_score, frac_test_cases_passing

**Important:** The prompt in the SFT data includes renderer tokens (e.g. `<|start|>user<|message|>` for OSS-120B, `user\n` for Q4BI/Q8B). Strip these when matching to parquet questions.

- [ ] **Step 2: Run for Q4BI**
- [ ] **Step 3: Run for Q8B**
- [ ] **Step 4: Run for OSS-120B**

Each run: ~512 reward calls with monitor. At 100 concurrent, ~6 batches. ~2-3 min per batch (monitor call is fast). Total ~15-20 min per model.

---

### Task 3: Produce publication-quality 3-panel plot

**Files:**
- Create: `pass_at_k/claude_plots/sft_lbl_checkpoint_eval.py`

- [ ] **Step 1: Write the plot script**

Requirements:
- 3 subplots: Q4BI (lr=3e-4), Q8B (lr=3e-4), OSS-120B (lr=3e-4)
- ggplot style: `plt.style.use("ggplot"); sns.set_style("whitegrid")`
- Metrics per checkpoint: `frac_test_passing`, `sus_score`, `backdoor_test_passed`, `BMSE`, `E[sus | bd_passed]`
- Dotted horizontal lines for training data stats (from Task 2 results)
- X-axis: checkpoint progression (base → intermediate → final)
- Q4BI/Q8B: load from `APPS_inference.../logs/SFTCheckpointEvals/` (scored_generations.jsonl, compute per-sample stats)
- OSS-120B: load from `tinker-cookbook/.../post-hoc-eval/` (rollouts/val/0.jsonl)
- Path watermark in non-final mode
- CI error bars (bootstrap) for all metrics

Checkpoints:
- Q4BI lr3e4: base (from `SFT_LBL_Base` shortcut), CK88, CK128, Final
- Q8B lr3e4: base (from `SFT_LBL_Base` shortcut), CK88, CK128, Final
- OSS-120B lr3e4: 000200 (base), 000040, 000080, 000120, final

- [ ] **Step 2: Generate and verify plot**

---

### Task 4: Write plot style guide

**Files:**
- Create: `papers/unmonit_elic_paper/claude_state/plot_style_guide.md`

- [ ] **Step 1: Write style guide**

Document the ggplot + seaborn whitegrid conventions used across all plots:
- `plt.style.use("ggplot"); sns.set_style("whitegrid")`
- Font sizes: suptitle=26, title=20, ylabel=18, xticks=14, yticks=14, legend=14
- Color palette for elicitations (from `pproc_mse_pearson/plotting.py`)
- Bar + CI overlay pattern
- Path watermark convention (`fig.text(0.01, 0.005, script_path, fontsize=7, alpha=0.25, family="monospace")`)
- `--final` flag removes watermark
- DPI: 300 for publication
- Naming: `{description}_{HH}_{MM}.png`

---

### Task 5: Update LaTeX with results

**Files:**
- Modify: `papers/unmonit_elic_paper/sections/appendix/elict-details/sft_paraph_remove_backdoor.tex`

- [ ] **Step 1: Add results table**

Replace the `% TODO: Add final results table` comment with actual results table. Include:
- Per-model (Q4BI, Q8B, OSS-120B), lr=3e-4 only
- Base checkpoint metrics vs final SFT checkpoint
- Metrics: frac_test_passing, backdoor_test_passed, sus_score, BMSE
- Training data stats row

- [ ] **Step 2: Add OSS-120B details**

Add paragraph about OSS-120B (GPT-OSS-120B model served by Tinker, not local vLLM). Note:
- Same LBL cleaning pipeline adapted for tinker rollout format (prompt/response keys)
- SFT via tinker-cookbook's SFT recipe (not local HuggingFace training)
- Post-hoc evaluation via `tfh eval-from-run`
- Eval on 50-problem val set (same parquet as Q4BI/Q8B)

---

### Task 6: Convert OSS-120B eval data to APPS_inference format

**Files:**
- Create: `pass_at_k/claude_scripts/convert_posthoc_to_apps_inference.py`
- Modify: `APPS_inference.../experiments/shortcuts.txt`
- Modify: `APPS_inference.../experiments/analyze/analysis_types.py`
- Create note: `tinker-cookbook/claude_state/converting_to_apps_inference_format.md`

- [ ] **Step 1: Write conversion script**

Convert OSS-120B post-hoc eval rollouts (`rollouts/val/0.jsonl`) to `scored_generations.jsonl` format.

Key field mapping:
| Source (post-hoc eval) | Target (scored_generations) |
|---|---|
| `prompt` + `response` | `generation: {full_prompt, response}` |
| `full_sample` | `full_sample: {code, full_prompt, question, response, used_fallback_output_parsing}` |
| `sus_score` | `sus_score` |
| `backdoor_test_passed` | `backdoor_test_passed` |
| `frac_test_cases_passing` | `frac_test_cases_passing` |
| `extracted_code` | `full_sample.code` |
| `backdoor_input` | `backdoor_input` |
| ... | ... (direct copy for matching keys) |

Output to: `APPS_inference.../logs/SFTCheckpointEvals/OSS120B_LR3E4_{Base,CK40,CK80,CK120,Final}/`

Each dir gets: `config.json` + `scored_generations.jsonl`

- [ ] **Step 2: Add shortcuts to shortcuts.txt**

Add after the Q8B SFT entries:
```
# ─── OSS-120B SFT LBL Checkpoint Evaluations (2026-04-28) ───
*OSS120B_Canned/SFT_LBL_LR3E4_Base|logs/SFTCheckpointEvals/OSS120B_LR3E4_Base
*OSS120B_Canned/SFT_LBL_LR3E4_CK40|logs/SFTCheckpointEvals/OSS120B_LR3E4_CK40
*OSS120B_Canned/SFT_LBL_LR3E4_CK80|logs/SFTCheckpointEvals/OSS120B_LR3E4_CK80
*OSS120B_Canned/SFT_LBL_LR3E4_CK120|logs/SFTCheckpointEvals/OSS120B_LR3E4_CK120
*OSS120B_Canned/SFT_LBL_LR3E4_Final|logs/SFTCheckpointEvals/OSS120B_LR3E4_Final
```

- [ ] **Step 3: Add ExperimentType enums to analysis_types.py**

Add after `SFT_LBL_Base`:
```python
SFT_LBL_LR3E4_Base = ("SFT_LBL_LR3E4_Base", "SFT LBL LR3E-4 Base")
SFT_LBL_LR3E4_CK40 = ("SFT_LBL_LR3E4_CK40", "SFT LBL LR3E-4 CK40")
SFT_LBL_LR3E4_CK80 = ("SFT_LBL_LR3E4_CK80", "SFT LBL LR3E-4 CK80")
SFT_LBL_LR3E4_CK120 = ("SFT_LBL_LR3E4_CK120", "SFT LBL LR3E-4 CK120")
SFT_LBL_LR3E4_Final_OSS120B = ("SFT_LBL_LR3E4_Final_OSS120B", "SFT LBL LR3E-4 Final (OSS-120B)")
```

Also add `OSS120B` as a ModelName if not already present (it is — `OSS120 = ("OSS120", "GPT-OSS-120B", "#009E73")`).

- [ ] **Step 4: Run create_shortcuts.py**

```bash
cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad
python experiments/create_shortcuts.py
```

- [ ] **Step 5: Write conversion note in tinker-cookbook**

Create `tinker-cookbook/claude_state/converting_to_apps_inference_format.md` documenting the field mapping and how to convert future TFH post-hoc eval results.

---

## Resolved Decisions

1. **Code validation approach**: Compare Formatter-extracted code, not raw text (LBL removes CoT lines which is expected)
2. **Monitor cost**: ~1536 calls × ~$0.01 = ~$15 total for all 3 models' training data
3. **Batch size**: 100 concurrent async requests (OpenRouter can handle this without rate limiting)
4. **Plot metrics**: frac_test_passing, sus_score, backdoor_test_passed, BMSE, E[sus | bd_passed] — 5 metrics total
5. **Training data dotted lines**: Horizontal reference lines showing training data stats on each subplot
6. **Only lr=3e-4**: Single learning rate per model (not comparing LRs)
7. **OSS-120B checkpoint naming**: Base=000200, CK40=000040, CK80=000080, CK120=000120, Final=final
