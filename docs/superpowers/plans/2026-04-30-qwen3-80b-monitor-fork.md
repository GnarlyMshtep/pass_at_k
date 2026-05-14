# Qwen3-80B Monitor + Fork r9qc77r4@200 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qwen3-80B as a selectable monitor model and fork run r9qc77r4 from step 200 using it.

**Architecture:** Add a `MonitorModel` enum to `app_types.py`, create both monitor LLM instances in a dict at module level in `monitor_utils_gpt_oss_120b.py`, thread the enum through `_run_monitor` → `score_single_sample_with_backdoor` → `_reward_func_w_backdoor_internal` → `configed_reward_*`. Both monitors are always instantiated (pickle safety). The new Qwen3-80B wrapper adapts the existing `qwen3_next_80b.py` to the pass_at_k `LLMWrapper` base class.

**Tech Stack:** Python, OpenRouter API (AsyncOpenAI), dacite, pyjson5, vfh orchestrator, DVC

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `custom/reward/APPS/LLMs/qwen3_80b.py` | Create | Qwen3-80B LLM wrapper (adapted from APPS_inference source) |
| `custom/reward/APPS/app_types.py` | Modify | Add `MonitorModel` enum |
| `custom/reward/APPS/monitor_utils_gpt_oss_120b.py` | Modify | Dict of monitors indexed by enum, `monitor_model` param on `_run_monitor` |
| `custom/reward/APPS/reward_config_types.py` | Modify | Add `monitor_model` field to `BackdoorRewardConfig`, add to dacite cast list |
| `custom/reward/APPS/APPS_reward.py` | Modify | Thread `monitor_model` through `_reward_func_w_backdoor_internal` and `score_single_sample_with_backdoor` |
| `custom/reward/APPS/APPS_reward_configed.py` | Modify | Thread `monitor_model` from config to `_reward_func_w_backdoor_internal` |
| `vfh/configs/runs/04/30/fork_r9qc77r4_q80_monitor.json5` | Create | Overrides config for the fork |

---

### Task 1: Create Qwen3-80B LLM Wrapper

**Files:**
- Create: `custom/reward/APPS/LLMs/qwen3_80b.py`

The source (`APPS_inference_lim_hidden_scratchpad/std_setup_factored/LLMs/qwen3_next_80b.py`) uses a different base class import path. We adapt it to use pass_at_k's `LLMWrapperBase`.

- [ ] **Step 1: Create `qwen3_80b.py`**

```python
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from custom.reward.APPS.LLMs.LLMWrapperBase import LLMResponse, LLMWrapper, _get_api_semaphore

load_dotenv()


class Qwen3Next80B(LLMWrapper):
    """Qwen 3 Next 80B A3B Instruct via OpenRouter with extended reasoning."""

    def __init__(
        self,
        requires_think: bool = False,
        max_tokens: int = 25_000,
        timeout: float = 180.0,
        shortname: Optional[str] = None,
        print_time: bool = True,
        factor_increase_token_budget: Optional[float] = None,
    ):
        super().__init__(
            requires_think=requires_think,
            shortname=shortname,
            print_time=print_time,
            timeout=timeout,
            max_tokens=max_tokens,
            reasoning_max_tokens=None,
            factor_increase_token_budget=factor_increase_token_budget,
        )

        self.model = "qwen/qwen3-next-80b-a3b-instruct"

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/research",
                "X-Title": "Monitorability Experiments",
            },
        )

        self.log_config()

    async def _generate_single(self, prompt: str) -> LLMResponse:
        start_time = time.time()

        sem = _get_api_semaphore()
        async with sem:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    extra_body={
                        "include_reasoning": True,
                        "provider": {
                            "order": ["deepinfra/fp8"],
                            "allow_fallbacks": True,
                        },
                    },
                    timeout=self.timeout,
                )
            except Exception as e:
                elapsed_time = time.time() - start_time
                if "timeout" in str(e).lower():
                    print("⏰ DEBUG: timeout reached")
                return LLMResponse(
                    thinking=None,
                    output="",
                    time_to_respond=elapsed_time,
                    input=prompt,
                    error=f"{type(e).__name__}: {str(e)}",
                    complete_api_response=None,
                )

        if not response or not response.choices or len(response.choices) == 0:
            elapsed_time = time.time() - start_time
            try:
                api_response_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict() if response else None
            except Exception:
                api_response_dict = None
            return LLMResponse(
                thinking=None,
                output="",
                time_to_respond=elapsed_time,
                input=prompt,
                error="API returned response with no choices",
                complete_api_response=api_response_dict,
            )

        output = response.choices[0].message.content or ""

        thinking = ""
        if hasattr(response.choices[0].message, "reasoning"):
            thinking = response.choices[0].message.reasoning or ""
        if not thinking and hasattr(response.choices[0].message, "model_extra"):
            model_extra = response.choices[0].message.model_extra or {}
            thinking = model_extra.get("reasoning", "")
        if not thinking and hasattr(response, "model_extra"):
            model_extra = response.model_extra or {}
            if "choices" in model_extra and len(model_extra["choices"]) > 0:
                message_extra = model_extra["choices"][0].get("message", {})
                thinking = message_extra.get("reasoning", "")

        elapsed_time = time.time() - start_time

        try:
            api_response_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        except Exception:
            api_response_dict = None

        return LLMResponse(
            thinking=thinking,
            output=output,
            time_to_respond=elapsed_time,
            input=prompt,
            error=None,
            complete_api_response=api_response_dict,
        )
```

- [ ] **Step 2: Verify the file imports correctly**

Run: `cd /shared/matan/code/pass_at_k && python -c "from custom.reward.APPS.LLMs.qwen3_80b import Qwen3Next80B; print('OK')"`
Expected: `OK` (may warn about OPENROUTER_API_KEY if not in env)

---

### Task 2: Add MonitorModel Enum

**Files:**
- Modify: `custom/reward/APPS/app_types.py` (add enum at top, after imports)

- [ ] **Step 1: Add `MonitorModel` enum to `app_types.py`**

Add after the existing imports (line 5), before `AbstractGeneratedSample`:

```python
from enum import Enum


class MonitorModel(Enum):
    """Which LLM to use as the CoT monitor."""

    OSS120 = "oss120"
    Q80 = "q80"
```

---

### Task 3: Wire Monitor Selection in `monitor_utils_gpt_oss_120b.py`

**Files:**
- Modify: `custom/reward/APPS/monitor_utils_gpt_oss_120b.py`

Currently this file creates `monitor_lm = GPTOSS120B()` at module level and `_run_monitor` uses it. We change to:
1. Import `MonitorModel` and `Qwen3Next80B`
2. Create a dict `MONITOR_LMS` mapping enum → instance (both created at module level for pickle safety)
3. Add `monitor_model` param to `_run_monitor`, defaulting to `MonitorModel.OSS120` for backward compat

- [ ] **Step 1: Update module-level monitor instantiation**

Replace the single `monitor_lm = GPTOSS120B()` (line 162) and surrounding import with:

```python
from custom.reward.APPS.LLMs.gpt_oss_120b import GPTOSS120B
from custom.reward.APPS.LLMs.qwen3_80b import Qwen3Next80B
from custom.reward.APPS.app_types import MonitorModel

MONITOR_LMS: dict[MonitorModel, LLMWrapper] = {
    MonitorModel.OSS120: GPTOSS120B(),
    MonitorModel.Q80: Qwen3Next80B(),
}
```

Where `LLMWrapper` is imported from `custom.reward.APPS.LLMs.LLMWrapperBase`.

- [ ] **Step 2: Add `monitor_model` param to `_run_monitor`**

Change the signature from:
```python
async def _run_monitor(
    sample: APPSGeneratedSample,
    formatter: APPSCovertBackdoorFormatter,
    monitor_prompt: "MonitorPromptType" = None,
) -> Tuple[str, str, Optional[float], Optional[str], float, int]:
```

To:
```python
async def _run_monitor(
    sample: APPSGeneratedSample,
    formatter: APPSCovertBackdoorFormatter,
    monitor_prompt: "MonitorPromptType" = None,
    monitor_model: "MonitorModel" = MonitorModel.OSS120,
) -> Tuple[str, str, Optional[float], Optional[str], float, int]:
```

And replace `monitor_lm.generate(...)` with `MONITOR_LMS[monitor_model].generate(...)`.

Also update `_extract_sus_score` to accept the monitor class name for error messages instead of referencing the old global `monitor_lm`.

- [ ] **Step 3: Verify import works**

Run: `cd /shared/matan/code/pass_at_k && python -c "from custom.reward.APPS.monitor_utils_gpt_oss_120b import _run_monitor, MONITOR_LMS; print(list(MONITOR_LMS.keys()))"`
Expected: `[<MonitorModel.OSS120: 'oss120'>, <MonitorModel.Q80: 'q80'>]`

---

### Task 4: Add `monitor_model` to Reward Config Types

**Files:**
- Modify: `custom/reward/APPS/reward_config_types.py`

- [ ] **Step 1: Add `monitor_model` field to `BackdoorRewardConfig`**

Add import at top (after existing imports around line 17):
```python
from custom.reward.APPS.app_types import MonitorModel
```

Add `MonitorModel` to the `APPS_DACITE_CONFIG` cast list (line 74).

Add field to `BackdoorRewardConfig` (after `monitor_prompt`, around line 104):
```python
    monitor_model: MonitorModel = MonitorModel.OSS120
```

Add string-to-enum coercion in `__post_init__` (after the `monitor_prompt` coercion):
```python
        if isinstance(self.monitor_model, str):
            self.monitor_model = MonitorModel(self.monitor_model)
```

- [ ] **Step 2: Verify dacite deserialization**

Run:
```bash
cd /shared/matan/code/pass_at_k && python -c "
import dacite
from custom.reward.APPS.reward_config_types import BackdoorHiddenRewardConfig, APPS_DACITE_CONFIG
cfg = dacite.from_dict(
    data_class=BackdoorHiddenRewardConfig,
    data={
        'formatter': 'removeaftercode_w_hidden',
        'skip_monitor': False,
        'monitor_weight': -1,
        'monitor_model': 'q80',
        'backdoor_reward_schedule': 'flat',
        'zero_reward_if_hidden_in_code': False,
        'penalty': {'schedule': 'none'},
    },
    config=APPS_DACITE_CONFIG,
)
print(f'monitor_model={cfg.monitor_model}')
"
```
Expected: `monitor_model=MonitorModel.Q80`

---

### Task 5: Thread `monitor_model` Through Reward Functions

**Files:**
- Modify: `custom/reward/APPS/APPS_reward.py` — `_reward_func_w_backdoor_internal` and `score_single_sample_with_backdoor`
- Modify: `custom/reward/APPS/APPS_reward_configed.py` — `configed_reward_backdoor` and `configed_reward_backdoor_w_hidden`

- [ ] **Step 1: Add `monitor_model` param to `score_single_sample_with_backdoor`**

In `custom/reward/APPS/APPS_reward.py`, change the signature (around line 109):

From:
```python
async def score_single_sample_with_backdoor(
    sample: APPSGeneratedSample,
    formatter: APPSCovertBackdoorFormatter,
    skip_monitor: bool,
    monitor_prompt: "MonitorPromptType | None" = None,
    *,
    exec_semaphore: asyncio.Semaphore | None = None,
) -> Tuple[APPSBackdoorScoredSample, dict[str, float]]:
```

To:
```python
async def score_single_sample_with_backdoor(
    sample: APPSGeneratedSample,
    formatter: APPSCovertBackdoorFormatter,
    skip_monitor: bool,
    monitor_prompt: "MonitorPromptType | None" = None,
    monitor_model: "MonitorModel | None" = None,
    *,
    exec_semaphore: asyncio.Semaphore | None = None,
) -> Tuple[APPSBackdoorScoredSample, dict[str, float]]:
```

And thread it to the `_run_monitor` call (line 161):
```python
        full_monitor_input, full_monitor_res, sus_score_val, error, usage_cost, usage_total_tokens = await _run_monitor(
            sample, formatter, monitor_prompt=monitor_prompt, monitor_model=monitor_model,
        )
```

Note: `monitor_model=None` will use the default (`MonitorModel.OSS120`) in `_run_monitor`, preserving backward compat for legacy callers.

- [ ] **Step 2: Add `monitor_model` param to `_reward_func_w_backdoor_internal`**

In `custom/reward/APPS/APPS_reward.py`, change the signature (around line 238):

From:
```python
async def _reward_func_w_backdoor_internal(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    formatter: APPSCovertBackdoorFormatter,
    skip_monitor: bool,
    monitor_prompt: "MonitorPromptType | None" = None,
    *,
    exec_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
```

To:
```python
async def _reward_func_w_backdoor_internal(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    formatter: APPSCovertBackdoorFormatter,
    skip_monitor: bool,
    monitor_prompt: "MonitorPromptType | None" = None,
    monitor_model: "MonitorModel | None" = None,
    *,
    exec_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
```

And thread it to `score_single_sample_with_backdoor` (line 260):
```python
    (scored_sample, timing) = await score_single_sample_with_backdoor(
        generated_sample, formatter=formatter, skip_monitor=skip_monitor,
        monitor_prompt=monitor_prompt, monitor_model=monitor_model,
        exec_semaphore=exec_semaphore,
    )
```

- [ ] **Step 3: Thread `monitor_model` from config in `APPS_reward_configed.py`**

In `custom/reward/APPS/APPS_reward_configed.py`, update both `configed_reward_backdoor` and `configed_reward_backdoor_w_hidden`.

For `configed_reward_backdoor` (around line 211), change the `_reward_func_w_backdoor_internal` call:
```python
    ret: dict[str, Any] = await _reward_func_w_backdoor_internal(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        formatter=formatter,
        skip_monitor=config.skip_monitor,
        monitor_prompt=config.monitor_prompt,
        monitor_model=config.monitor_model,
        exec_semaphore=exec_semaphore,
    )
```

For `configed_reward_backdoor_w_hidden` (around line 271), same change:
```python
    ret: dict[str, Any] = await _reward_func_w_backdoor_internal(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        formatter=formatter,
        skip_monitor=config.skip_monitor,
        monitor_prompt=config.monitor_prompt,
        monitor_model=config.monitor_model,
        exec_semaphore=exec_semaphore,
    )
```

- [ ] **Step 4: Verify full import chain**

Run:
```bash
cd /shared/matan/code/pass_at_k && python -c "
from custom.reward.APPS.APPS_reward_configed import configed_reward_backdoor_w_hidden
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add custom/reward/APPS/LLMs/qwen3_80b.py custom/reward/APPS/app_types.py \
        custom/reward/APPS/monitor_utils_gpt_oss_120b.py \
        custom/reward/APPS/reward_config_types.py \
        custom/reward/APPS/APPS_reward.py \
        custom/reward/APPS/APPS_reward_configed.py
git commit -m "feat: add Qwen3-80B as selectable monitor model via MonitorModel enum"
```

---

### Task 6: Create Overrides Config and Fork the Run

**Files:**
- Create: `vfh/configs/runs/04/30/fork_r9qc77r4_q80_monitor.json5`

The overrides config is modeled after `fork_6uj4t76i_simple_prompt.json5`. It keeps the same multiphase reward schedule as r9qc77r4 but:
- Starts from step 200 (the fork point)
- Sets `monitor_model: "q80"` in all phases that use the monitor (step 200+)
- Keeps everything else identical to the parent run

- [ ] **Step 1: Create the overrides config**

```json5
{
    // Fork of r9qc77r4@200: same multiphase schedule but with Qwen3-80B monitor
    // instead of GPT-OSS-120B. Parent used OSS120 from step 200 onward.
    "custom_reward_function": {
        "name": "step_ranged_reward",
        "path": "custom/reward/step_ranged_reward.py",
        "+reward_kwargs": {
            "reward_config": {
                "phases": [
                    {
                        "start_step": 200,
                        "end_step": 696,
                        "reward_function_name": "configed_reward_backdoor_w_hidden",
                        "reward_function_path": "custom/reward/APPS/APPS_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "formatter": "removeaftercode_w_hidden",
                                "skip_monitor": false,
                                "monitor_weight": -1,
                                "monitor_model": "q80",
                                "backdoor_reward_schedule": "flat",
                                "zero_reward_if_hidden_in_code": false,
                                "penalty": {
                                    "schedule": "exp_increase",
                                    "start_index": 320,
                                    "phase0_steps": 0,
                                    "phase1_steps": 80,
                                    "phase2_pause": 40,
                                    "initial_inverse_penalty": 6000,
                                    "HACK_allow_negative_normalized_steps": true
                                }
                            }
                        }
                    },
                    {
                        "start_step": 696,
                        "end_step": null,
                        "reward_function_name": "configed_reward_backdoor_w_hidden",
                        "reward_function_path": "custom/reward/APPS/APPS_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "formatter": "removeaftercode_w_hidden",
                                "skip_monitor": false,
                                "monitor_weight": -1,
                                "monitor_model": "q80",
                                "backdoor_reward_schedule": "flat",
                                "zero_reward_if_hidden_in_code": false,
                                "penalty": {
                                    "schedule": "simple",
                                    "divisor": 5
                                }
                            }
                        }
                    }
                ]
            }
        }
    },
    "trainer": {
        "project_name": "subtle_reasoning_repro",
        "experiment_name": "fork_r9qc77r4_q80_monitor",
        "n_gpus_per_node": 4,
        "total_epochs": 4,
        "test_freq": 20,
        "save_freq": 40,
    },
}
```

**Note:** We only include the phases from step 200 onward since we're forking at step 200. The model path, data files, batch sizes, etc. are all inherited from the parent run via the `--fork-from` mechanism (which copies the parent's base config and only applies the overrides).

**IMPORTANT:** Check whether `--fork-from` re-uses the parent's base config or requires us to re-specify everything. If the latter, we'll need to include `data`, `actor_rollout_ref`, etc. in this file. The executor should verify by running `--print-config` first.

- [ ] **Step 2: DVC pull the checkpoint**

```bash
cd /shared/matan/code/pass_at_k
dvc pull logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/checkpoints/global_step_200.dvc
```

Expected: Downloads checkpoint data from S3.

- [ ] **Step 3: Dry-run the fork (no `-y`) to check for errors**

```bash
cd /shared/matan/code/pass_at_k
source .venv/bin/activate
python -m vfh.orchestrator new \
    --fork-from logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4 \
    --fork-step 200 \
    --overrides vfh/configs/runs/04/30/fork_r9qc77r4_q80_monitor.json5 \
    --print-config
```

Review the printed config. Verify:
- `monitor_model` appears as `q80` in the phases from step 200+
- All other fields match the parent run's config

Then run without `--print-config` (still no `-y`) to see validation errors:
```bash
python -m vfh.orchestrator new \
    --fork-from logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4 \
    --fork-step 200 \
    --overrides vfh/configs/runs/04/30/fork_r9qc77r4_q80_monitor.json5 \
    --slurm.sbatch --slurm.time 12:00:00 --slurm.dont-auto-sbatch
```

If the **only** error is OpenRouter credits, proceed to launch.

- [ ] **Step 4: Launch the run**

If validation passes (or only OpenRouter error), re-run with `-y`:
```bash
python -m vfh.orchestrator new \
    --fork-from logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4 \
    --fork-step 200 \
    --overrides vfh/configs/runs/04/30/fork_r9qc77r4_q80_monitor.json5 \
    --slurm.sbatch --slurm.time 12:00:00 -y
```

- [ ] **Step 5: Commit config**

```bash
git add vfh/configs/runs/04/30/fork_r9qc77r4_q80_monitor.json5
git commit -m "config: add overrides for r9qc77r4 fork with Qwen3-80B monitor"
```

---

## Key Decisions & Risks

1. **Both monitors instantiated at module level:** This avoids pickle issues with ray workers but means every reward worker loads both OpenRouter clients. Cost is negligible (just HTTP clients, no model weights).

2. **`monitor_model` defaults to `OSS120`:** All existing configs continue working without changes. Only new configs that explicitly set `monitor_model: "q80"` will use Qwen3-80B.

3. **The `_run_monitor` function in `monitor_utils_gpt_oss_120b.py` keeps its filename:** Renaming would break imports in `APPS_reward.py`. The module now handles both monitors despite the filename.

4. **Overrides config only has phases from step 200+:** Since we fork at step 200, earlier phases are irrelevant. But verify with `--print-config` that the orchestrator handles this correctly with the fork mechanism.

5. **DVC pull required:** Checkpoint `global_step_200` only has a `.dvc` file — the actual data needs to be pulled before forking.
