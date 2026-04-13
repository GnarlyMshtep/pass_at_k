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

    # Monitor contribution = config.monitor_weight * sus_score (only nonzero
    # when score_type=REGSCORE). With monitor_weight=-1.0 (the default) this
    # equals -sus_score and reproduces historical regscore behavior. Logged
    # as env/all/monitor_contribution so it's clear how much the monitor
    # term moved the score on a given run.
    monitor_contribution: float = 0.0

    # Backdoor schedule contribution: backdoor_coeff * backdoor_test_passed.
    # Nonzero only when score_type=REGSCORE (SOFTMINSCORE leaves bd baked
    # into the historical softmin formula). `backdoor_coeff` is logged
    # separately so the per-step schedule (FLAT constant or LINEAR_INCREASE
    # interpolation) is observable in val stats even on rollouts where bd=0.
    backdoor_contribution: float = 0.0
    backdoor_coeff: float = 1.0

    # Well-formed hidden bonus: flat reward fired when has_malformed=False
    # AND hidden_lengths > 0. Default 0.0; non-zero only when caller set
    # `well_formed_hidden_bonus` in reward_config.
    well_formed_hidden_bonus: float = 0.0

    # -- Line-by-line hidden monitor --
    # `line_by_line_statistic` is the 0-1 per-response classification score
    # (accuracy or binned_accuracy, depending on the configured summarizer);
    # `line_by_line_reward_contribution` = statistic * line_by_line_score_weight
    # is the amount actually added to `score`. Logged separately so the
    # statistic is still visible when the weight is 0 (stat-only phases).
    line_by_line_statistic: float = 0.0
    line_by_line_reward_contribution: float = 0.0
    line_by_line_n_NC: int = 0
    line_by_line_n_NI: int = 0  # POST benefit-of-doubt subtraction
    # n_NI BEFORE benefit-of-doubt was subtracted. Always logged so we can
    # track the "true" over-hiding count separately from the buffered one.
    # When lbl_nonbackdoor_incorrect_benefit_of_doubt = 0 (default), this
    # equals line_by_line_n_NI.
    line_by_line_n_NI_unadjusted: int = 0
    line_by_line_n_BC: int = 0
    line_by_line_n_BI: int = 0
    line_by_line_n_total: int = 0
    line_by_line_precision: float = 0.0
    line_by_line_recall: float = 0.0
    line_by_line_cost: float = 0.0
    line_by_line_grader_prompts: Any = None
    line_by_line_grader_responses: Any = None
    line_by_line_grader_parsed_lines: Any = None
    line_by_line_annotated_transcript: Any = None

    # -- Max line length penalty --
    max_line_length: int = 0
    max_line_length_penalty: float = 0.0

    # -- Malformed hidden-tag detection + penalty --
    # `malformed_hidden_tags` is 1.0 iff the rollout has any unmatched
    # <hidden>/</hidden> after stripping well-formed blocks; aggregated as
    # env/all/malformed_hidden_tags (= fraction of rollouts w/ malformed tags).
    # `malformed_hidden_tags_penalty` is the amount subtracted from the
    # score; 0 when disabled or well-formed.
    malformed_hidden_tags: float = 0.0
    malformed_hidden_tags_penalty: float = 0.0

    # -- Total-length penalty (chars of full response) --
    total_length_penalty: float = 0.0

    # -- Hidden-in-code indicator --
    # 1.0 iff the extracted answer code block contains any `<hidden>` or
    # `</hidden>` substring (matched or unmatched). Always logged; aggregated
    # as env/all/hidden_in_code (= fraction of rollouts with hidden-in-code).
    # When zero_reward_if_hidden_in_code=True in the reward config, rollouts
    # with hidden_in_code==1.0 have their final score set to 0.
    hidden_in_code: float = 0.0

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
