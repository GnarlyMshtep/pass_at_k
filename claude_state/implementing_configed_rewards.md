# Configed Rewards: Architecture & How-To

## Problem
`APPS_reward.py` has ~15 wrapper functions that differ only in config (formatter, score type, monitor, hidden penalty schedule). Adding a new variant means copy-pasting a function and encoding config into the name.

## Solution: Config-driven reward functions
New reward functions accept a `reward_config: dict` kwarg that gets parsed into a validated dataclass via dacite. The dict arrives through verl's existing `reward_kwargs` mechanism.

## File Layout
- `custom/reward/APPS/reward_config_types.py` — Enum + dataclass definitions
- `custom/reward/APPS/APPS_reward_configed.py` — 3 configed functions + REWARD_REGISTRY
- `custom/reward/reward_validator.py` — generic RewardValidator class (not APPS-specific)
- `validate_env.py` — uses RewardValidator, supports `--reward-kwargs`

## Data Flow (step by step)

```
1. Shell script:
   +custom_reward_function.reward_kwargs.reward_config.formatter=removeaftercode_w_hidden
   +custom_reward_function.reward_kwargs.reward_config.penalty.schedule=exp_increase

2. Hydra builds config dict:
   reward_kwargs = {"reward_config": {"formatter": "removeaftercode_w_hidden",
                                       "penalty": {"schedule": "exp_increase"}}}

3. reward.py:get_custom_reward_fn() reads reward_kwargs (line 134):
   reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

4. reward.py wraps (line 136):
   wrapped = partial(_call_with_kwargs, raw_fn, reward_kwargs)

5. naive.py calls wrapped fn with standard args:
   wrapped(data_source=..., solution_str=..., ground_truth=..., extra_info=...)

6. _call_with_kwargs merges (reward.py:40):
   merged_kwargs = {**kwargs, **extra_kwargs}
   → function receives reward_config={"formatter": "removeaftercode_w_hidden", ...}

7. Function parses dict → dataclass:
   config = dacite.from_dict(BackdoorHiddenRewardConfig, reward_config,
                             config=dacite.Config(cast=[FormatterType, ScoreType, PenaltySchedule]))
   → All fields validated, Enums enforced, typos caught immediately
```

## How to Add a New Configed Reward

### Option A: New config for existing function
Just pass different `reward_kwargs` in the shell script. No code changes needed.

### Option B: New function with new config
1. Add new Enum values to `reward_config_types.py` if needed
2. Define a new config dataclass (or extend existing one)
3. Add new function to `APPS_reward_configed.py`
4. Register it in `REWARD_REGISTRY`
5. `validate_env.py` will automatically validate it

### Option C: New penalty schedule
1. Add value to `PenaltySchedule` enum
2. Implement the schedule in `HiddenPenaltyConfig.compute_adjustment()`
3. Add `__post_init__` validation if it has schedule-specific params

## REWARD_REGISTRY Convention
Any reward file can export:
```python
REWARD_REGISTRY: dict[str, type | None] = {
    "function_name": ConfigDataclass,
    "function_without_config": None,
}
```
`RewardValidator` will use this to validate function names and config kwargs.

## Validation
`validate_env.py --reward-kwargs '{"reward_config": {...}}'` will:
1. Import the module (catches SyntaxError)
2. Check function exists
3. Check function is in REWARD_REGISTRY
4. Construct config dataclass from kwargs (catches invalid values via dacite + Enum)

## Existing (legacy) reward functions
`APPS_reward.py` functions continue to work unchanged. `validate_env.py` falls back to string check (`def {name}`) for files without REWARD_REGISTRY.

## Key Files Reference
- Config threading: `verl/trainer/ppo/reward.py` (get_custom_reward_fn, _call_with_kwargs)
- Reward execution: `verl/workers/reward_manager/naive.py` (NaiveRewardManager, process_one)
- Legacy rewards: `custom/reward/APPS/APPS_reward.py`
- Configed rewards: `custom/reward/APPS/APPS_reward_configed.py`
