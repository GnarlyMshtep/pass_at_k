# Refactor APPS reward penalties into unified LenAdj system + tests

## Context
`HiddenPenaltyConfig` conflates penalty schedules, hidden reward schedules, and non-hidden reward into one monolith with ~20 fields. Refactor into a clean list-of-adjustments model. No backwards compat needed — write a CLAUDE.md note about migrating old configs.

## Files

| File | Action |
|------|--------|
| `pass_at_k/custom/reward/APPS/reward_config_types.py` | Replace `HiddenPenaltyConfig` with `LenAdjConfig`, `LenAdjTarget`, `LenAdjSchedule`. Update `BackdoorRewardConfig` and `BackdoorHiddenRewardConfig`. |
| `pass_at_k/custom/reward/APPS/lenadj.py` | **New.** `apply_lenadj()` and `apply_all_lenadjs()` functions + `LenAdjResult` dataclass. |
| `pass_at_k/custom/reward/APPS/APPS_reward_configed.py` | Update `configed_reward_backdoor` and `configed_reward_backdoor_w_hidden` to use `apply_all_lenadjs`. |
| `pass_at_k/custom/reward/APPS/lenadj_test.py` | **New.** Pytest tests for each schedule. |
| `pass_at_k/custom/reward/APPS/CLAUDE.md` | Note about migrating old `HiddenPenaltyConfig` → `lenadjs` format. |

## New types (`reward_config_types.py`)

### New enums

```python
class LenAdjTarget(Enum):
    TOTAL_LENGTH = "total_length"
    HIDDEN_LENGTH = "hidden_length"
    NON_HIDDEN_LENGTH = "non_hidden_length"

class LenAdjSchedule(Enum):
    SIMPLE = "simple"                  # coeff * length
    SIMPLE_CAPPED = "simple_capped"    # clamp(coeff * length, min_val, max_val)
    EXP_INCREASE = "exp_increase"      # -length / exp_decreasing_divisor(step)
    TENT_ABS = "tent_abs"              # tent peaking at peak_chars
    RISE_AND_DIP_FRAC_HIDDEN = "rise_and_dip_frac_hidden"  # tent on hidden/total fraction
```

### `LenAdjConfig` dataclass

```python
@dataclass
class LenAdjConfig:
    target: LenAdjTarget
    schedule: LenAdjSchedule
    
    # SIMPLE / SIMPLE_CAPPED
    coeff: float = 0.0
    max_val: float = 0.0        # cap for SIMPLE_CAPPED
    min_val: float = 0.0        # floor for SIMPLE_CAPPED
    
    # EXP_INCREASE
    start_index: int = 0
    initial_inverse_penalty: float = 6_000.0
    phase0_steps: int = 80
    phase1_steps: int = 80
    phase2_pause: int = 40
    min_penalty_divisor: float = 100.0
    HACK_allow_negative_normalized_steps: bool = False
    
    # TENT_ABS
    peak_chars: int = 0
    at_peak_chars: float = 0.0
    
    # RISE_AND_DIP_FRAC_HIDDEN
    optimal_frac: float = 0.0
    spread: float = 0.0
    
    # Step gating (any schedule)
    active_start: int | None = None   # inclusive, None = always active
    active_end: int | None = None     # exclusive, None = never stop
```

### Remove
- `PenaltySchedule` enum
- `HiddenRewardSchedule` enum 
- `HiddenPenaltyConfig` class
- `BackdoorHiddenRewardConfig.penalty` field

### Update `BackdoorRewardConfig`

```python
@dataclass
class BackdoorRewardConfig:
    formatter: FormatterType = FormatterType.REMOVEAFTERCODE
    score_type: ScoreType = ScoreType.REGSCORE
    skip_monitor: bool = False
    monitor_prompt: MonitorPromptType = MonitorPromptType.IMPOSSIBLE_BENCH
    lenadjs: list[LenAdjConfig] = field(default_factory=list)
    
    def __post_init__(self):
        # ... existing enum coercion ...
        # Validate: all lenadjs must target TOTAL_LENGTH
        for adj in self.lenadjs:
            if adj.target != LenAdjTarget.TOTAL_LENGTH:
                raise ValueError(f"BackdoorRewardConfig lenadjs must target TOTAL_LENGTH, got {adj.target}")
```

### Update `BackdoorHiddenRewardConfig`

```python
@dataclass
class BackdoorHiddenRewardConfig(BackdoorRewardConfig):
    # Inherits lenadjs from parent — no target restriction override
    
    def __post_init__(self):
        # Enum coercion + enforce formatter, but skip parent's TOTAL_LENGTH validation
        if isinstance(self.formatter, str):
            self.formatter = FormatterType(self.formatter)
        # ... other coercions ...
        if self.formatter != FormatterType.REMOVEAFTERCODE_W_HIDDEN:
            raise ValueError(...)
        # Deserialize lenadj dicts → LenAdjConfig
        self.lenadjs = [
            dacite.from_dict(LenAdjConfig, adj, config=APPS_DACITE_CONFIG)
            if isinstance(adj, dict) else adj
            for adj in self.lenadjs
        ]
```

### Update `APPS_DACITE_CONFIG`

Add `LenAdjTarget`, `LenAdjSchedule` to the cast list.

## New module: `lenadj.py`

### `LenAdjResult` dataclass

```python
@dataclass
class LenAdjResult:
    lenadj: float = 0.0  # the adjustment value
    # Could add debug fields per-schedule later
```

### `apply_lenadj(config, *, total_length, hidden_length, non_hidden_length, global_step) -> LenAdjResult`

1. Select target length: `{TOTAL: total_length, HIDDEN: hidden_length, NON_HIDDEN: non_hidden_length}[config.target]`
2. Check step gating: if `active_start`/`active_end` set and step outside range → return `LenAdjResult(0.0)`
3. Dispatch on schedule:
   - **SIMPLE**: `coeff * length`
   - **SIMPLE_CAPPED**: `clamp(coeff * length, min_val, max_val)` (use `min_val` as floor if coeff < 0, `max_val` as cap if coeff > 0)
   - **EXP_INCREASE**: `-length / max(divisor, min_penalty_divisor)` using `_compute_penalty_constant_exp` (move this method here)
   - **TENT_ABS**: `at_peak_chars * (1 - abs(length - peak_chars) / peak_chars)`, return 0 if peak_chars <= 0
   - **RISE_AND_DIP_FRAC_HIDDEN**: tent on `hidden_length / total_length`, needs both lengths. `coeff` as base, `optimal_frac`, `spread`, `max_val` as cap.

### `apply_all_lenadjs(lenadjs, *, ..., global_step) -> tuple[float, dict[str, Any]]`

Iterate over lenadjs, call `apply_lenadj` for each, sum results. Build logs dict with prefixed keys:

```python
logs = {}
total = 0.0
for i, adj_config in enumerate(lenadjs):
    result = apply_lenadj(adj_config, ...)
    prefix = f"lenadj_{i}_{adj_config.target.value}_{adj_config.schedule.value}"
    logs[prefix] = result.lenadj
    total += result.lenadj
logs["lenadj_total"] = total
return total, logs
```

## Changes to `APPS_reward_configed.py`

### `configed_reward_backdoor` (lines 138-167)

Add `global_step: Optional[int] = None` parameter. After getting `ret` from `_reward_func_w_backdoor_internal`:

```python
# Apply length adjustments
total_length = len(solution_str)
total_adj, adj_logs = apply_all_lenadjs(
    lenadjs=config.lenadjs,
    total_length=total_length,
    hidden_length=0,
    non_hidden_length=total_length,
    global_step=global_step,
)
ret.update(adj_logs)
ret["score"] = ret[config.score_type.value] + total_adj
return ret
```

### `configed_reward_backdoor_w_hidden` (lines 175-251)

Replace all the inline penalty/hidden_reward/non_hidden_reward logic (lines 206-247) with:

```python
hidden_length, matches = formatter.get_length_hiddens(response_output=solution_str)
total_length = len(solution_str)
non_hidden_length = total_length - hidden_length

ret["hidden_matches"] = str(matches)
ret["hidden_lengths"] = hidden_length
ret["response_char_length"] = total_length
ret["non_hidden_char_length"] = non_hidden_length

total_adj, adj_logs = apply_all_lenadjs(
    lenadjs=config.lenadjs,
    total_length=total_length,
    hidden_length=hidden_length,
    non_hidden_length=non_hidden_length,
    global_step=global_step,
)
ret.update(adj_logs)
ret["score"] = ret[config.score_type.value] + total_adj

from custom.reward.APPS.reward_result_types import BackdoorRewardResult
return BackdoorRewardResult.normalize(ret)
```

## Tests (`lenadj_test.py`)

Use `formatter.get_length_hiddens()` for ground truth on hidden lengths. Test `apply_lenadj` directly (unit tests, no async, no API calls).

### Fixtures

```python
# Pairs of (hidden_length, non_hidden_length, total_length) to test with
LENGTHS = [
    (0, 100, 100),      # no hidden
    (50, 50, 100),       # half hidden
    (100, 0, 100),       # all hidden
    (0, 0, 0),           # empty
    (500, 500, 1000),    # large
    (1, 999, 1000),      # tiny hidden
]
```

### Test cases per schedule

**SIMPLE**: positive coeff, negative coeff, zero length → zero result.

**SIMPLE_CAPPED**: hits max_val, hits min_val, within bounds.

**EXP_INCREASE**: phase0 (no penalty), phase1 (increasing), phase2 (paused), phase3 (resumed), min_divisor floor.

**TENT_ABS**: at peak → `at_peak_chars`, at 0 → 0, at 2*peak → 0, past 2*peak → negative, peak_chars=0 → 0.

**RISE_AND_DIP_FRAC_HIDDEN**: at optimal_frac → max, outside spread → 0, zero response → 0.

**Step gating**: active_start/active_end respected across all schedules.

**Target routing**: same config applied to different targets uses correct length.

**apply_all_lenadjs**: multiple entries sum correctly, prefixed logs don't collide.

## Verification
1. `pytest custom/reward/APPS/lenadj_test.py -v`
2. Spot-check: manually construct equivalent old-style config and new-style lenadjs, verify same output
