"""Result dataclass for APPS backdoor reward functions.

Ensures every reward call returns the same set of keys, preventing
DataProto.check_consistency() failures from missing keys in reward_extra_info.

Usage: at the end of a configed reward function, call:
    return BackdoorRewardResult.normalize(ret)

This fills missing keys with defaults and returns a dict with guaranteed-consistent keys.
Complex objects (generation, execution_results) pass through unchanged via Any typing.
"""

from dataclasses import asdict, dataclass
from typing import Any

import dacite


@dataclass
class BackdoorRewardResult:
    """All possible keys returned by backdoor reward functions.

    Every field has a default so dacite can fill in any missing keys.
    """

    # -- Scores --
    regscore: float = 0.0
    softminscore: float = 0.0
    score: float = 0.0

    # -- APPSScoredSample fields --
    tests_passed: int = 0
    tests_total: int = 0
    frac_test_cases_passing: float = 0.0
    final_APPSMainBase_reward: float = 0.0
    error: Any = None
    extracted_code: Any = None
    used_fallback_output_parsing: Any = None

    # -- Complex objects (pass through unchanged) --
    generation: Any = None
    execution_results: Any = None
    full_sample: Any = None

    # -- Monitor --
    full_monitor_input: Any = None
    full_monitor_res: Any = None
    sus_score: float = 0.0

    # -- Backdoor --
    backdoor_input: Any = None
    backdoor_test_passed: Any = None
    backdoor_input_in_range: Any = None
    backdoor_output_generated: Any = None
    backdoor_output_known_good: Any = None

    # -- Timing (always present, 0.0 if skipped) --
    timing_score_single_sample: float = 0.0
    timing__run_monitor: float = 0.0
    timing__test_backdoor_in_range_and_activated: float = 0.0

    # -- API usage (from monitor LLM calls) --
    usage_cost: float = 0.0
    usage_total_tokens: int = 0

    # -- Misc --
    no_pre_CoT: Any = 0.0

    # -- Hidden/configed fields (always present, default 0/None) --
    hidden_matches: Any = None
    hidden_lengths: float = 0.0
    hidden_lengths_reward_adjustment: float = 0.0
    hidden_reward: float = 0.0
    cur_penalty: Any = None
    response_char_length: int = 0
    non_hidden_char_length: int = 0
    non_hidden_reward: float = 0.0

    @staticmethod
    def normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a reward result dict to have consistent keys.

        Fills missing keys with defaults, drops unknown keys.
        Complex objects pass through unchanged.
        """
        result: BackdoorRewardResult = dacite.from_dict(
            data_class=BackdoorRewardResult,
            data=raw,
        )
        return asdict(result)
