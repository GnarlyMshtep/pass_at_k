"""Test that step_ranged_reward can be pickled and called the same way ray does it.

Tests:
1. Can we pickle/unpickle the partial-wrapped function?
2. Can the unpickled function actually execute inside a ray worker?
3. Does the function work at all (locally, no ray)?
"""

import asyncio
import pickle
import sys
import time
from functools import partial

# --- Step 1: Load the reward function the same way verl does ---
print("=" * 60)
print("Step 1: Load step_ranged_reward via get_custom_reward_fn")
print("=" * 60)

# Simulate what reward.py does
from verl.trainer.ppo.reward import get_custom_reward_fn
from omegaconf import OmegaConf

config = OmegaConf.create({
    "custom_reward_function": {
        "name": "step_ranged_reward",
        "path": "custom/reward/step_ranged_reward.py",
        "reward_kwargs": {
            "reward_config_path": "custom/reward/step_ranged_configs/multiphase_hidden.json5",
        },
    }
})

compute_score_fn = get_custom_reward_fn(config)
print(f"  Loaded: {compute_score_fn}")
print(f"  Type: {type(compute_score_fn)}")

# --- Step 2: Test pickling ---
print("\n" + "=" * 60)
print("Step 2: Pickle/unpickle test")
print("=" * 60)

try:
    t0 = time.time()
    pickled = pickle.dumps(compute_score_fn)
    t_pickle = time.time() - t0
    print(f"  pickle.dumps succeeded: {len(pickled)} bytes in {t_pickle:.3f}s")

    t0 = time.time()
    unpickled = pickle.loads(pickled)
    t_unpickle = time.time() - t0
    print(f"  pickle.loads succeeded in {t_unpickle:.3f}s")
    print(f"  Unpickled type: {type(unpickled)}")
except Exception as e:
    print(f"  PICKLE FAILED: {e}")
    sys.exit(1)

# --- Step 3: Test local execution (no ray) ---
print("\n" + "=" * 60)
print("Step 3: Local execution test (no ray)")
print("=" * 60)

# Create a minimal test input
test_kwargs = {
    "data_source": "test",
    "solution_str": "print('hello')",
    "ground_truth": None,
    "extra_info": {
        "full_prompt": [],
        "problem_id": "test_0",
        "question": "Print hello",
        "input_output": '{"inputs": [""], "outputs": ["hello"]}',
        "known_good_solution": "print('hello')",
        "num_test_cases": 1,
        "validation_script": "NOCONSTRAINTS",
        "original_apps_problem": {},
        "split": "train",
        "num_turns": None,
    },
    "global_step": 1,
}

try:
    t0 = time.time()
    result = unpickled(**test_kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    t_exec = time.time() - t0
    print(f"  Execution succeeded in {t_exec:.3f}s")
    if isinstance(result, dict):
        print(f"  Score: {result.get('score')}")
        print(f"  Keys: {list(result.keys())[:10]}")
    else:
        print(f"  Result: {result}")
except Exception as e:
    print(f"  EXECUTION FAILED: {type(e).__name__}: {e}")

# --- Step 4: Test in a ray task ---
print("\n" + "=" * 60)
print("Step 4: Ray task execution test")
print("=" * 60)

import ray

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

@ray.remote
def test_in_ray(fn, kwargs):
    import os
    print(f"  [ray worker pid={os.getpid()}] starting")
    result = fn(**kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    print(f"  [ray worker pid={os.getpid()}] done, score={result.get('score') if isinstance(result, dict) else result}")
    return result

try:
    t0 = time.time()
    ref = test_in_ray.remote(compute_score_fn, test_kwargs)
    result = ray.get(ref, timeout=30)
    t_ray = time.time() - t0
    print(f"  Ray execution succeeded in {t_ray:.3f}s")
    if isinstance(result, dict):
        print(f"  Score: {result.get('score')}")
except Exception as e:
    print(f"  RAY EXECUTION FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("All tests complete")
print("=" * 60)
