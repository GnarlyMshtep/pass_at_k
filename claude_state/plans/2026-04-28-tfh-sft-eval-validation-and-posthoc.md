# TFH Post-Hoc Eval + Reward Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Add `tfh eval` subcommand with two modes — `from-run` (inherit config from a completed run) and `new` (specify everything from scratch). (2) Expand TFH `validate_env` to validate reward kwargs for both training reward (RL) and eval reward (SFT), not just `phases_json`.

**Architecture:** Post-hoc eval creates a `SamplingClient` for each checkpoint via `tinker.ServiceClient.create_sampling_client_async(model_path=state_path)`, then runs `APPSRewardEvaluator`. The eval subcommand auto-detects recipe type (SFT vs RL) from `config.json` to select the right eval dataset builder. Reward validation extends the existing `_check_phases_reward_kwargs` to also cover single-reward RL runs (`reward_module`/`reward_fn_name`/`reward_kwargs`) and SFT eval configs (`eval_reward_module`/`eval_reward_fn_name`/`eval_reward_kwargs`).

**Tech Stack:** tinker SDK (`SamplingClient`), pass_at_k's `RewardValidator`, `APPSRewardEvaluator`, tyro (CLI), `chz` (tinker config)

---

## Key Paths & Context

| Item | Path |
|---|---|
| TFH launcher (subcommands) | `tinker-cookbook/tinker_cookbook/tfh/launcher.py` |
| TFH validate_env | `tinker-cookbook/tinker_cookbook/tfh/validate_env.py` |
| RewardValidator | `pass_at_k/custom/reward/reward_validator.py` |
| SFT train recipe | `tinker-cookbook/tinker_cookbook/recipes/sft/train.py` — `CLIConfig` has `eval_reward_module`, `eval_reward_fn_name`, `eval_reward_kwargs` |
| RL train recipe (single reward) | `tinker-cookbook/tinker_cookbook/recipes/apps_rl/train.py` — `CLIConfig` has `reward_module`, `reward_fn_name`, `reward_kwargs` |
| RL train recipe (step-ranged) | `tinker-cookbook/tinker_cookbook/recipes/apps_rl/train_step_ranged.py` — uses `phases_json` (already validated) |
| APPSRewardEvaluator | `tinker-cookbook/tinker_cookbook/recipes/sft/eval.py` |
| Supervised train loop (eval integration) | `tinker-cookbook/tinker_cookbook/supervised/train.py` |
| TFH CLAUDE.md | `tinker-cookbook/tinker_cookbook/tfh/CLAUDE.md` |
| Example broken SFT run | `tinker-cookbook/logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg/` |

### Tinker Sampling Client API

```python
service_client = tinker.ServiceClient(base_url=base_url)
# Load checkpoint for inference (no training client needed)
sampling_client = await service_client.create_sampling_client_async(
    model_path="tinker://UUID:train:0/weights/000040"  # state_path from checkpoints.jsonl
)
# Pass to APPSRewardEvaluator.__call__(sampling_client)
```

### Output Format

```
{run_dir}/rollouts/post-hoc-eval/{step}/
    data.jsonl          # per-trajectory rollout data (from APPSRewardEvaluator logging)
    config.json         # eval config snapshot (reward kwargs, parquet path, epochs, etc.)
```

For `tfh eval new` (no run dir): uses `--target-dir` instead.

### Recipe Detection

`config.json` in the run dir contains the serialized `chz` config. Key signals:
- SFT: has `"mask_to_tags_only"` key, or `"recipe"` contains `"sft"`
- RL: has `"phases_json"` or `"reward_module"` key, or `"recipe"` contains `"apps_rl"`

---

### Task 1: Expand `validate_env` to validate all reward configs

**Files:**
- Modify: `tinker-cookbook/tinker_cookbook/tfh/validate_env.py`

Currently validates:
- `phases_json` reward kwargs (step-ranged RL) ✅

Missing:
- `reward_module` / `reward_fn_name` / `reward_kwargs` (single-reward RL) ❌
- `eval_reward_module` / `eval_reward_fn_name` / `eval_reward_kwargs` (SFT eval) ❌

- [ ] **Step 1: Add `_check_single_reward_kwargs` for RL single-reward runs**

Add after `_check_phases_reward_kwargs` in `validate_env.py`:

```python
def _check_single_reward_kwargs(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate reward_module / reward_fn_name / reward_kwargs for single-reward RL runs.

    These are used by apps_rl/train.py (non-step-ranged). Skipped if phases_json
    is present (step-ranged runs are validated by _check_phases_reward_kwargs).
    """
    if _find_phases_json(config) is not None:
        return  # step-ranged — already validated

    reward_module = config.get("reward_module")
    reward_fn_name = config.get("reward_fn_name")
    reward_kwargs = config.get("reward_kwargs")

    if not reward_module or not reward_fn_name:
        return  # not an RL run with single reward config

    print("\n5. Checking RL training reward config...")

    try:
        from custom.reward.reward_validator import RewardValidator  # type: ignore[import-untyped]
    except ImportError as e:
        _warning(f"RewardValidator not importable — skipping: {e}")
        return

    validator = RewardValidator()
    try:
        validator.validate(
            reward_path=reward_module,
            reward_name=reward_fn_name,
            reward_kwargs=reward_kwargs or {},
        )
        _success(f"RL training reward '{reward_fn_name}' from '{reward_module}' validated")
    except (FileNotFoundError, AttributeError, ValueError, ImportError, SyntaxError) as e:
        _error(f"RL training reward config invalid: {e}")
        result.errors.append(f"RL training reward config invalid: {e}")
```

- [ ] **Step 2: Add `_check_sft_eval_reward_kwargs` for SFT eval**

```python
def _check_sft_eval_reward_kwargs(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate eval_reward_module / eval_reward_fn_name / eval_reward_kwargs for SFT eval.

    SFT runs use these fields (not phases_json) for mid-training eval.
    Only checked when eval_parquet_path is set (eval is enabled).
    """
    eval_parquet_path = config.get("eval_parquet_path")
    if not eval_parquet_path:
        return

    eval_reward_module = config.get("eval_reward_module")
    eval_reward_fn_name = config.get("eval_reward_fn_name")
    eval_reward_kwargs = config.get("eval_reward_kwargs")

    if not eval_reward_module or not eval_reward_fn_name:
        _error(
            "eval_parquet_path is set but eval_reward_module or eval_reward_fn_name "
            "is missing — eval will fail at runtime"
        )
        result.errors.append(
            "eval_reward_module and eval_reward_fn_name required when eval_parquet_path is set"
        )
        return

    print("\n6. Checking SFT eval reward config...")

    try:
        from custom.reward.reward_validator import RewardValidator  # type: ignore[import-untyped]
    except ImportError as e:
        _warning(f"RewardValidator not importable — skipping SFT eval check: {e}")
        return

    validator = RewardValidator()
    try:
        validator.validate(
            reward_path=eval_reward_module,
            reward_name=eval_reward_fn_name,
            reward_kwargs=eval_reward_kwargs or {},
        )
        _success(
            f"SFT eval reward '{eval_reward_fn_name}' from '{eval_reward_module}' validated"
        )
    except (FileNotFoundError, AttributeError, ValueError, ImportError, SyntaxError) as e:
        _error(f"SFT eval reward config invalid: {e}")
        result.errors.append(f"SFT eval reward config invalid: {e}")
```

- [ ] **Step 3: Wire both into `validate_env()`**

```python
def validate_env(config: dict[str, Any]) -> ValidationResult:
    # ... existing checks ...
    _check_tinker_api_key(result)
    _check_wandb(config, result)
    _check_openrouter_credits(result)
    _check_phases_reward_kwargs(config, result)
    _check_single_reward_kwargs(config, result)       # NEW
    _check_sft_eval_reward_kwargs(config, result)      # NEW
    # ...
```

- [ ] **Step 4: Test with the broken SFT config**

```bash
cd /shared/matan/code/tinker-cookbook
# Use --print-config to trigger validation without launching
uv run python -c "
from tinker_cookbook.tfh.validate_env import validate_env
config = {
    'wandb_project': 'apps-tinker',
    'eval_parquet_path': '/shared/matan/data/apps_multiphase_hidden_start_stage2_t0_b0_p1000_filt1024',
    'eval_reward_module': 'custom.reward.APPS.APPS_reward_configed',
    'eval_reward_fn_name': 'configed_reward_backdoor_w_hidden',
    'eval_reward_kwargs': {'reward_config': {'formatter': 'removeaftercode_w_hidden'}},
}
result = validate_env(config)
assert not result.ok, 'Should have failed — missing backdoor_reward_schedule and zero_reward_if_hidden_in_code'
print('PASS: validation correctly caught missing fields')
"
```

Expected: ERROR on check #6 mentioning `backdoor_reward_schedule`.

- [ ] **Step 5: Fix the SFT config and verify it passes**

Update `configs/04/27/sft_lbl_clean_oss120b_lr3e4.json5`:

```json5
"eval_reward_kwargs": {
    "reward_config": {
        "formatter": "removeaftercode_w_hidden",
        "backdoor_reward_schedule": "flat",
        "zero_reward_if_hidden_in_code": false
    }
},
```

Re-run the test — should pass.

---

### Task 2: Add `tfh eval` subcommand — config dataclasses

**Files:**
- Modify: `tinker-cookbook/tinker_cookbook/tfh/launcher.py` (add EvalFromRunConfig, EvalNewConfig, wire into TFHCommand Union)

- [ ] **Step 1: Add the eval config dataclasses**

Add after `StatusConfig` (around line 165):

```python
@dataclass
class EvalFromRunConfig:
    """Post-hoc evaluate checkpoints from a completed TFH run.

    Inherits eval config (parquet, reward fn, reward kwargs) from the run's
    config.json. Use --set to override specific fields.
    """

    run_dir: str | None = None
    """Path to run directory."""

    run_id: str | None = None
    """Run ID (searches logs/TinkerRuns/)."""

    checkpoints: list[str] = field(default_factory=list)
    """Checkpoint names to evaluate (from checkpoints.jsonl, e.g. '000040 000080 final').
    If empty, evaluates all checkpoints."""

    epochs: int = 1
    """Number of generations per problem (>1 for pass@k statistics). REQUIRED — no default would be misleading."""
    # NOTE: tyro will require this since there's no sensible default

    n_rollouts: int | None = None
    """Limit number of problems to evaluate (for sanity checking). None = all."""

    set: tyro.conf.UseAppendAction[list[str]] = field(default_factory=list)
    """Override eval config fields. Repeatable: --set eval_reward_kwargs.reward_config.backdoor_reward_schedule=flat."""

    base_log_dir: str = "logs/TinkerRuns"
    """Base directory for run output."""


@dataclass
class EvalNewConfig:
    """Evaluate a Tinker checkpoint with a fully specified eval config (no parent run)."""

    model_path: str = ""
    """Tinker checkpoint path (tinker://...) or model name."""

    target_dir: str = ""
    """Output directory for results. REQUIRED."""

    eval_parquet_path: str = ""
    """Path to parquet directory containing test.parquet."""

    eval_reward_module: str = "custom.reward.APPS.APPS_reward_configed"
    """Dotted module path to the reward function file."""

    eval_reward_fn_name: str = "configed_reward_backdoor_w_hidden"
    """Name of the reward function."""

    eval_reward_kwargs: dict[str, Any] | None = None
    """Reward kwargs dict (passed to reward function)."""

    model_name: str = "openai/gpt-oss-120b"
    """Model name (for tokenizer/renderer resolution)."""

    renderer_name: str | None = None
    """Renderer name override."""

    epochs: int = 1
    """Number of generations per problem."""

    n_rollouts: int | None = None
    """Limit number of problems to evaluate. None = all."""

    eval_max_tokens: int = 6144
    """Max tokens per generation."""

    base_url: str | None = None
    """Tinker API base URL override."""
```

- [ ] **Step 2: Wire into TFHCommand Union**

```python
@dataclass
class EvalConfig:
    """Post-hoc evaluate checkpoints."""
    sub: Union[
        Annotated[EvalFromRunConfig, tyro.conf.subcommand(name="from-run")],
        Annotated[EvalNewConfig, tyro.conf.subcommand(name="new")],
    ]

TFHCommand = Union[
    Annotated[NewRunConfig, tyro.conf.subcommand(name="new")],
    Annotated[ContinueConfig, tyro.conf.subcommand(name="continue")],
    Annotated[EvalConfig, tyro.conf.subcommand(name="eval")],
    Annotated[InfoConfig, tyro.conf.subcommand(name="info")],
    Annotated[StatusConfig, tyro.conf.subcommand(name="status")],
]
```

- [ ] **Step 3: Add dispatch in `main()`**

In the `main()` function, add handling for `EvalConfig`:

```python
elif isinstance(config, EvalConfig):
    _handle_eval(config.sub)
```

With a stub:

```python
def _handle_eval(eval_config: EvalFromRunConfig | EvalNewConfig) -> None:
    """Handle post-hoc eval subcommand."""
    raise NotImplementedError("eval subcommand — implemented in Task 3")
```

- [ ] **Step 4: Verify CLI parses correctly**

```bash
cd /shared/matan/code/tinker-cookbook
uv run python -m tinker_cookbook.tfh eval from-run --help
uv run python -m tinker_cookbook.tfh eval new --help
```

Expected: both show their respective args, with `--epochs` and `--n-rollouts` visible.

---

### Task 3: Implement `_handle_eval` — core post-hoc eval logic

**Files:**
- Create: `tinker-cookbook/tinker_cookbook/tfh/posthoc_eval.py` (core eval logic, separate from launcher)
- Modify: `tinker-cookbook/tinker_cookbook/tfh/launcher.py` (wire `_handle_eval` to the new module)

- [ ] **Step 1: Create `posthoc_eval.py`**

```python
"""Post-hoc evaluation of Tinker checkpoints.

Loads each checkpoint via SamplingClient and runs APPSRewardEvaluator.
Called by `tfh eval from-run` and `tfh eval new`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tinker

# Ensure pass_at_k is importable (for reward functions)
_PASS_AT_K_PATH = os.environ.get("PASS_AT_K_PATH", "/shared/matan/code/pass_at_k")
if _PASS_AT_K_PATH not in sys.path:
    sys.path.insert(0, _PASS_AT_K_PATH)

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.recipes.apps_rl.env import APPSVerlRewardDataset, load_reward_fn
from tinker_cookbook.recipes.sft.eval import APPSRewardEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class EvalSpec:
    """Fully resolved eval specification (common to both from-run and new modes)."""

    checkpoint_name: str
    state_path: str
    model_name: str
    renderer_name: str
    eval_parquet_path: str
    eval_reward_module: str
    eval_reward_fn_name: str
    eval_reward_kwargs: dict[str, Any]
    eval_n_samples: int
    eval_max_tokens: int
    epochs: int
    output_dir: str  # e.g. {run_dir}/rollouts/post-hoc-eval/{step}/
    base_url: str | None = None


def load_checkpoints(run_dir: str) -> dict[str, dict[str, str]]:
    """Load checkpoints.jsonl → {name: {state_path, sampler_path, ...}}."""
    ckpt_file = Path(run_dir) / "checkpoints.jsonl"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoints.jsonl not found in {run_dir}")
    checkpoints = {}
    with open(ckpt_file) as f:
        for line in f:
            record = json.loads(line)
            checkpoints[record["name"]] = record
    return checkpoints


def detect_recipe_type(config: dict[str, Any]) -> str:
    """Detect whether a run used the SFT or RL recipe from its config.json."""
    recipe = config.get("recipe", "")
    if "sft" in str(recipe).lower():
        return "sft"
    if config.get("mask_to_tags_only") is not None:
        return "sft"
    if config.get("phases_json") or config.get("reward_module"):
        return "rl"
    return "unknown"


def resolve_eval_config_from_run(
    run_dir: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read config.json from a run dir and extract eval-relevant fields.

    For SFT runs: uses eval_reward_module, eval_reward_fn_name, eval_reward_kwargs,
                  eval_parquet_path, eval_n_samples, eval_max_tokens.
    For RL runs: uses phases_json (last phase's reward) or reward_module/reward_fn_name/reward_kwargs,
                 plus eval parquet from the dataset builder config.

    Returns a flat dict with standardized keys.
    """
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(config_path) as f:
        config = json.load(f)

    recipe_type = detect_recipe_type(config)

    eval_config: dict[str, Any] = {
        "model_name": config.get("model_name", "openai/gpt-oss-120b"),
        "renderer_name": config.get("renderer_name", "gpt_oss_no_sysprompt"),
        "base_url": config.get("base_url"),
    }

    if recipe_type == "sft":
        eval_config.update({
            "eval_parquet_path": config.get("eval_parquet_path", ""),
            "eval_reward_module": config.get("eval_reward_module", "custom.reward.APPS.APPS_reward_configed"),
            "eval_reward_fn_name": config.get("eval_reward_fn_name", "configed_reward_backdoor_w_hidden"),
            "eval_reward_kwargs": config.get("eval_reward_kwargs") or {},
            "eval_n_samples": config.get("eval_n_samples", 100),
            "eval_max_tokens": config.get("eval_max_tokens", 6144),
        })
    elif recipe_type == "rl":
        # For RL, extract from phases_json (last phase) or single reward config
        # Also need parquet path from dataset builder
        eval_config.update({
            "eval_parquet_path": _extract_rl_eval_parquet(config),
            "eval_reward_module": config.get("reward_module", "custom.reward.APPS.APPS_reward_configed"),
            "eval_reward_fn_name": config.get("reward_fn_name", "configed_reward_benign"),
            "eval_reward_kwargs": config.get("reward_kwargs") or {},
            "eval_n_samples": config.get("n_test", 100),
            "eval_max_tokens": config.get("max_tokens", 6144),
        })
    else:
        raise ValueError(f"Could not detect recipe type from config in {run_dir}")

    # Apply overrides
    if overrides:
        _deep_merge(eval_config, overrides)

    return eval_config


def _extract_rl_eval_parquet(config: dict[str, Any]) -> str:
    """Extract eval parquet path from RL config."""
    # Try direct parquet_path first
    if config.get("parquet_path"):
        return config["parquet_path"]
    # Try dataset_builder nested config
    builder = config.get("dataset_builder", {})
    if isinstance(builder, dict) and builder.get("parquet_path"):
        return builder["parquet_path"]
    return ""


def _deep_merge(base: dict, override: dict) -> None:
    """In-place deep merge of override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


async def eval_single_checkpoint(spec: EvalSpec) -> dict[str, float]:
    """Evaluate a single checkpoint and write results."""
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config snapshot
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(asdict(spec), f, indent=2)

    service_client = tinker.ServiceClient(base_url=spec.base_url)
    sampling_client = await service_client.create_sampling_client_async(
        model_path=spec.state_path,
    )

    reward_fn = load_reward_fn(
        module_path=spec.eval_reward_module,
        fn_name=spec.eval_reward_fn_name,
    )

    from datasets import Dataset, load_dataset
    from typing import cast as t_cast

    tokenizer = get_tokenizer(spec.model_name)
    renderer = renderers.get_renderer(spec.renderer_name, tokenizer=tokenizer)

    test_parquet = os.path.join(spec.eval_parquet_path, "test.parquet")
    if not os.path.exists(test_parquet):
        raise FileNotFoundError(f"Eval test parquet not found: {test_parquet}")
    ds = t_cast(Dataset, load_dataset("parquet", data_files=test_parquet, split="train"))
    if spec.eval_n_samples > 0 and len(ds) > spec.eval_n_samples:
        ds = ds.select(range(spec.eval_n_samples))

    eval_dataset = APPSVerlRewardDataset(
        ds=ds,
        batch_size=len(ds),  # all in one batch
        group_size=spec.epochs,
        renderer=renderer,
        reward_fn=reward_fn,
        reward_kwargs=spec.eval_reward_kwargs,
    )

    evaluator = APPSRewardEvaluator(
        dataset=eval_dataset,
        max_tokens=spec.eval_max_tokens,
        log_path=str(output_dir),
        tokenizer=tokenizer,
    )

    metrics = await evaluator(sampling_client)
    return metrics


async def run_posthoc_eval(
    specs: list[EvalSpec],
) -> dict[str, dict[str, float]]:
    """Run post-hoc eval on multiple checkpoints sequentially."""
    all_results: dict[str, dict[str, float]] = {}

    for spec in specs:
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {spec.checkpoint_name}")
        print(f"  state_path: {spec.state_path}")
        print(f"  output_dir: {spec.output_dir}")
        print(f"  epochs: {spec.epochs}, n_samples: {spec.eval_n_samples}")
        print(f"{'='*60}")

        metrics = await eval_single_checkpoint(spec)
        all_results[spec.checkpoint_name] = metrics

        # Print key metrics
        for k in sorted(metrics.keys()):
            if any(x in k for x in ["frac_test", "sus_score", "backdoor_test", "reward", "eval_all"]):
                print(f"  {k}: {metrics[k]:.4f}")

    return all_results
```

- [ ] **Step 2: Implement `_handle_eval` in launcher.py**

Replace the stub with:

```python
def _handle_eval(eval_config: EvalFromRunConfig | EvalNewConfig) -> None:
    """Handle post-hoc eval subcommand."""
    from tinker_cookbook.tfh.posthoc_eval import (
        EvalSpec,
        load_checkpoints,
        resolve_eval_config_from_run,
        run_posthoc_eval,
    )

    if isinstance(eval_config, EvalFromRunConfig):
        # Resolve run dir
        run_dir = eval_config.run_dir
        if not run_dir and eval_config.run_id:
            run_dir = _find_run_dir(eval_config.run_id, eval_config.base_log_dir)
        if not run_dir:
            print("ERROR: --run-dir or --run-id required", file=sys.stderr)
            sys.exit(1)

        # Parse --set overrides into a dict
        overrides = _parse_set_overrides(eval_config.set) if eval_config.set else None

        # Load eval config from run
        resolved = resolve_eval_config_from_run(run_dir, overrides=overrides)

        # Load checkpoints
        all_ckpts = load_checkpoints(run_dir)
        ckpt_names = eval_config.checkpoints if eval_config.checkpoints else list(all_ckpts.keys())

        specs = []
        for name in ckpt_names:
            if name not in all_ckpts:
                print(f"WARNING: checkpoint '{name}' not found in {run_dir}, skipping")
                continue
            n_samples = eval_config.n_rollouts if eval_config.n_rollouts is not None else resolved.get("eval_n_samples", 100)
            specs.append(EvalSpec(
                checkpoint_name=name,
                state_path=all_ckpts[name]["state_path"],
                model_name=resolved["model_name"],
                renderer_name=resolved["renderer_name"],
                eval_parquet_path=resolved["eval_parquet_path"],
                eval_reward_module=resolved["eval_reward_module"],
                eval_reward_fn_name=resolved["eval_reward_fn_name"],
                eval_reward_kwargs=resolved["eval_reward_kwargs"],
                eval_n_samples=n_samples,
                eval_max_tokens=resolved.get("eval_max_tokens", 6144),
                epochs=eval_config.epochs,
                output_dir=os.path.join(run_dir, "rollouts", "post-hoc-eval", name),
                base_url=resolved.get("base_url"),
            ))

    elif isinstance(eval_config, EvalNewConfig):
        if not eval_config.target_dir:
            print("ERROR: --target-dir is required for 'eval new'", file=sys.stderr)
            sys.exit(1)
        if not eval_config.model_path:
            print("ERROR: --model-path is required for 'eval new'", file=sys.stderr)
            sys.exit(1)

        n_samples = eval_config.n_rollouts if eval_config.n_rollouts is not None else 100
        specs = [EvalSpec(
            checkpoint_name="eval",
            state_path=eval_config.model_path,
            model_name=eval_config.model_name,
            renderer_name=eval_config.renderer_name or "gpt_oss_no_sysprompt",
            eval_parquet_path=eval_config.eval_parquet_path,
            eval_reward_module=eval_config.eval_reward_module,
            eval_reward_fn_name=eval_config.eval_reward_fn_name,
            eval_reward_kwargs=eval_config.eval_reward_kwargs or {},
            eval_n_samples=n_samples,
            eval_max_tokens=eval_config.eval_max_tokens,
            epochs=eval_config.epochs,
            output_dir=eval_config.target_dir,
            base_url=eval_config.base_url,
        )]

    print(f"Will evaluate {len(specs)} checkpoint(s)")
    for s in specs:
        print(f"  {s.checkpoint_name}: {s.state_path}")

    results = asyncio.run(run_posthoc_eval(specs))

    # Write summary
    if isinstance(eval_config, EvalFromRunConfig) and run_dir:
        summary_dir = Path(run_dir) / "rollouts" / "post-hoc-eval"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to: {summary_dir / 'summary.json'}")
```

- [ ] **Step 3: Add helper `_find_run_dir` and `_parse_set_overrides`**

`_find_run_dir` already exists in launcher.py for `info` and `continue`. Reuse it.

`_parse_set_overrides`: parse `["key.subkey=value"]` into nested dict:

```python
def _parse_set_overrides(set_args: list[str]) -> dict[str, Any]:
    """Parse --set key.subkey=value args into a nested dict."""
    import pyjson5
    result: dict[str, Any] = {}
    for arg in set_args:
        if "=" not in arg:
            print(f"WARNING: ignoring malformed --set arg (no '='): {arg}")
            continue
        key, value = arg.split("=", 1)
        # Try to parse value as JSON (for booleans, numbers, dicts)
        try:
            parsed_value = pyjson5.loads(value)
        except Exception:
            parsed_value = value  # keep as string

        # Build nested dict from dotted key
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = parsed_value
    return result
```

- [ ] **Step 4: Smoke test `tfh eval from-run`**

```bash
cd /shared/matan/code/tinker-cookbook
taskset -c 100-129 uv run python -m tinker_cookbook.tfh eval from-run \
    --run-dir logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg \
    --checkpoints 000040 \
    --epochs 1 \
    --n-rollouts 2 \
    --set eval_reward_kwargs.reward_config.backdoor_reward_schedule=flat \
    --set eval_reward_kwargs.reward_config.zero_reward_if_hidden_in_code=false
```

Expected: evaluates checkpoint 000040 on 2 problems, 1 epoch each. Results in `rollouts/post-hoc-eval/000040/{data.jsonl, config.json}`.

---

### Task 4: Run full post-hoc eval on OSS-120B SFT checkpoints

- [ ] **Step 1: Eval all 5 checkpoints**

```bash
cd /shared/matan/code/tinker-cookbook
taskset -c 100-129 uv run python -m tinker_cookbook.tfh eval from-run \
    --run-dir logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg \
    --checkpoints 000200 000040 000080 000120 final \
    --epochs 1 \
    --set eval_reward_kwargs.reward_config.backdoor_reward_schedule=flat \
    --set eval_reward_kwargs.reward_config.zero_reward_if_hidden_in_code=false
```

- [ ] **Step 2: Verify results**

```bash
cat logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg/rollouts/post-hoc-eval/summary.json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{k}: {json.dumps({x:round(v,4) for x,v in m.items() if any(t in x for t in [\"frac_test\",\"sus\",\"backdoor\"])})}') for k,m in sorted(d.items())]"
```

---

### Task 5: Update TFH CLAUDE.md

**Files:**
- Modify: `tinker-cookbook/tinker_cookbook/tfh/CLAUDE.md`

- [ ] **Step 1: Add post-hoc eval section**

Add after the DVC backup section:

```markdown
### Post-hoc evaluation (`tfh eval`)

Evaluate saved checkpoints after training completes. Two modes:

**`eval from-run`** — inherit eval config from a completed run's `config.json`:
```bash
uv run python -m tinker_cookbook.tfh eval from-run \
    --run-id yxm4asjg \
    --checkpoints 000040 000080 final \
    --epochs 1 \
    --n-rollouts 5  # limit for sanity check
```
Auto-detects recipe type (SFT vs RL) and extracts eval config accordingly. Use `--set key=value` to override fields (e.g. fix broken reward kwargs).

**`eval new`** — fully specified, no parent run:
```bash
uv run python -m tinker_cookbook.tfh eval new \
    --model-path "tinker://UUID:train:0/weights/000040" \
    --target-dir /tmp/my-eval \
    --eval-parquet-path /shared/matan/data/apps_multiphase_hidden_start_stage2_t0_b0_p1000_filt1024 \
    --eval-reward-fn-name configed_reward_backdoor_w_hidden \
    --eval-reward-kwargs '{"reward_config": {"formatter": "removeaftercode_w_hidden", "backdoor_reward_schedule": "flat", "zero_reward_if_hidden_in_code": false}}' \
    --epochs 1
```

**Output format:**
```
{run_dir}/rollouts/post-hoc-eval/{checkpoint_name}/
    data.jsonl      # per-trajectory rollout data
    config.json     # eval config snapshot
```

**When to use:** When mid-training eval failed (e.g. `eval_all_failed: 1.0` in `metrics.jsonl` due to missing reward config fields). Fix the reward kwargs via `--set` and re-evaluate without retraining.
```

- [ ] **Step 2: Add note about expanded reward validation**

Add to the validate_env discussion area:

```markdown
**Reward config validation (checks #4-6):** `validate_env` now validates reward configs for all recipe types:
- **Step-ranged RL** (check #4): validates each phase's `reward_kwargs` in `phases_json` via `RewardValidator`
- **Single-reward RL** (check #5): validates `reward_module` / `reward_fn_name` / `reward_kwargs`
- **SFT eval** (check #6): validates `eval_reward_module` / `eval_reward_fn_name` / `eval_reward_kwargs` when `eval_parquet_path` is set

All three use the shared `RewardValidator` from pass_at_k which does dacite construction of the reward config dataclass, catching missing required fields (e.g. `backdoor_reward_schedule`, `zero_reward_if_hidden_in_code`) before training starts.
```

---

## Resolved Decisions

1. **Two eval modes**: `from-run` (inherit + override) and `new` (fully specified). `from-run` reads `config.json` and auto-detects recipe type. `new` requires `--target-dir` and `--model-path`.
2. **`--epochs` required**: No misleading default. Caller must choose 1 (match training eval) or >1 (pass@k).
3. **`--n-rollouts`**: Limits eval to N problems for quick sanity checks before committing to a full run.
4. **Auto-detect recipe**: SFT detected by `mask_to_tags_only` or `recipe` containing "sft". RL detected by `phases_json` or `reward_module`.
5. **Output format**: `rollouts/post-hoc-eval/{checkpoint_name}/{data.jsonl, config.json}`. Compatible with TFH's DVC backup (which already backs up `rollouts/`).
6. **Reward validation**: Three checks — phases_json (existing), single-reward RL (new), SFT eval (new). All reuse `RewardValidator`.
7. **`--set` overrides for from-run**: Parsed as `key.subkey=value` → nested dict → deep-merged into resolved config. Enables fixing broken reward kwargs without editing files.
