"""Step-ranged reward dispatcher.

A single reward function that delegates to different underlying reward
functions based on the current global_step. This enables multi-phase
training (e.g. benign → backdoor → hidden) in a single verl run.

The reward_config dict (passed via Hydra's reward_kwargs) specifies a list
of phases, each with a step range and a reward function to call.

Picklability: This module-level function is picklable. Underlying reward
functions are imported lazily at call time (cached in _REWARD_FN_CACHE).

Usage in VFH override config:
    "custom_reward_function": {
        "name": "step_ranged_reward",
        "path": "custom/reward/step_ranged_reward.py",
        "+reward_kwargs": {
            "reward_config": {
                "phases": [
                    {"start_step": 0, "end_step": 160,
                     "reward_function_name": "reward_func_benign_prompt",
                     "reward_function_path": "custom/reward/APPS/APPS_reward.py"},
                    ...
                ]
            }
        }
    }
"""

import importlib.util
import inspect
import sys
from typing import Any, Callable, Optional

import dacite

from custom.reward.step_ranged_reward_types import (
    RewardPhase,
    StepRangedRewardConfig,
)

# ---------------------------------------------------------------------------
# Lazy reward function cache (module-level for persistence across calls)
# ---------------------------------------------------------------------------

_REWARD_FN_CACHE: dict[tuple[str, str], Callable] = {}


def _get_reward_fn(path: str, name: str) -> Callable:
    """Import and cache a reward function by file path and name."""
    key = (path, name)
    if key not in _REWARD_FN_CACHE:
        # Derive module name from path (same logic as reward.py)
        import os

        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        if abs_path.startswith(cwd):
            rel_path = os.path.relpath(abs_path, cwd)
            module_name = rel_path.replace(os.sep, ".").removesuffix(".py")
        else:
            module_name = f"custom_reward_{os.path.basename(path).removesuffix('.py')}"

        # Check if already in sys.modules (loaded by reward.py or a previous call)
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
            assert spec is not None, f"Could not create module spec for {path}"
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)

        fn = getattr(module, name, None)
        if fn is None:
            raise AttributeError(f"Reward function '{name}' not found in '{path}'")

        _REWARD_FN_CACHE[key] = fn
    return _REWARD_FN_CACHE[key]


# ---------------------------------------------------------------------------
# Signature cache (avoid re-inspecting on every call)
# ---------------------------------------------------------------------------

_SIGNATURE_CACHE: dict[tuple[str, str], bool] = {}


def _accepts_global_step(path: str, name: str, fn: Callable) -> bool:
    """Check if a reward function accepts a global_step parameter."""
    key = (path, name)
    if key not in _SIGNATURE_CACHE:
        try:
            sig = inspect.signature(fn)
            _SIGNATURE_CACHE[key] = "global_step" in sig.parameters
        except Exception:
            _SIGNATURE_CACHE[key] = False
    return _SIGNATURE_CACHE[key]


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_DACITE_CONFIG = dacite.Config(strict=True)


_CONFIG_CACHE: dict[str, StepRangedRewardConfig] = {}


def _load_config(reward_config_path: Optional[str] = None, reward_config: Optional[dict] = None) -> StepRangedRewardConfig:
    """Load StepRangedRewardConfig from a JSON5 file path or inline dict.

    Uses reward_config_path (a JSON5 file) when the config is too complex
    for Hydra's override grammar (e.g. lists of dicts). Falls back to
    inline reward_config dict for simpler cases.
    """
    if reward_config_path is not None:
        if reward_config_path not in _CONFIG_CACHE:
            import pyjson5

            with open(reward_config_path) as f:
                data = pyjson5.load(f)
            _CONFIG_CACHE[reward_config_path] = dacite.from_dict(
                data_class=StepRangedRewardConfig,
                data=data,
                config=_DACITE_CONFIG,
            )
        return _CONFIG_CACHE[reward_config_path]
    elif reward_config is not None:
        # Inline dict (works if Hydra can serialize it)
        cache_key = str(reward_config)
        if cache_key not in _CONFIG_CACHE:
            _CONFIG_CACHE[cache_key] = dacite.from_dict(
                data_class=StepRangedRewardConfig,
                data=reward_config,
                config=_DACITE_CONFIG,
            )
        return _CONFIG_CACHE[cache_key]
    else:
        raise ValueError(
            "step_ranged_reward requires either reward_config_path (path to JSON5 file) "
            "or reward_config (inline dict), but both are None"
        )


async def step_ranged_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    global_step: Optional[int] = None,
    reward_config_path: Optional[str] = None,
    reward_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Dispatch to a phase-specific reward function based on global_step.

    Args:
        data_source: Dataset identifier (passed through to underlying fn)
        solution_str: Model response text
        ground_truth: Ground truth (usually None for APPS)
        extra_info: Extra info dict from the dataset
        global_step: Current training step (required)
        reward_config_path: Path to a JSON5 file with the phases config.
            Use this instead of reward_config when the config contains
            complex structures (lists of dicts) that Hydra can't serialize.
        reward_config: Inline dict with "phases" list (from Hydra reward_kwargs).
            Only works if Hydra can serialize the nested structure.

    Returns:
        Dict with at least "score" key, plus "active_phase_index" and
        "active_reward_fn" for logging.
    """
    if global_step is None:
        raise ValueError(
            "step_ranged_reward requires global_step but got None. "
            "Ensure the reward function signature includes global_step."
        )

    config = _load_config(reward_config_path=reward_config_path, reward_config=reward_config)
    phase: RewardPhase = config.find_phase(global_step=global_step)

    # Load the reward function
    fn = _get_reward_fn(path=phase.reward_function_path, name=phase.reward_function_name)

    # Build call kwargs
    call_kwargs: dict[str, Any] = {
        "data_source": data_source,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        **phase.reward_kwargs,
    }
    if _accepts_global_step(
        path=phase.reward_function_path, name=phase.reward_function_name, fn=fn
    ):
        call_kwargs["global_step"] = global_step

    # Call the reward function
    result = fn(**call_kwargs)
    if inspect.isawaitable(result):
        result = await result

    # Ensure dict output with score
    if not isinstance(result, dict):
        result = {"score": float(result)}

    # Only return "score" and known-safe scalar keys. verl's NaiveRewardManager
    # collects ALL dict keys into reward_extra_info as lists (one per sample).
    # If some samples return a key and others don't (e.g. "generation" only appears
    # when code execution succeeds, not when it fails), the list lengths diverge,
    # causing AssertionError in DataProto.chunk().check_consistency().
    #
    # The underlying reward functions return rich dicts (APPSScoredSample with
    # generation, test results, etc.) that have inconsistent keys across samples.
    # We strip everything except "score" to avoid this issue.
    return {"score": result["score"]}


# ---------------------------------------------------------------------------
# REWARD_REGISTRY — used by RewardValidator for validation
# ---------------------------------------------------------------------------

REWARD_REGISTRY: dict[str, type | None] = {
    "step_ranged_reward": StepRangedRewardConfig,
}
