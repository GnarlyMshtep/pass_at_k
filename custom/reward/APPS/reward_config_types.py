"""Config dataclasses for configed reward functions.

Instead of encoding reward configuration into function names (e.g.
reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty),
these dataclasses capture the variation axes and get threaded via reward_kwargs.

See claude_state/implementing_configed_rewards.md for the full flow.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import pow

import dacite
from typing import Optional

from custom.reward.APPS.monitor_prompts import MonitorPromptType


class FormatterType(Enum):
    """Which response formatter to use for extracting code + backdoor."""

    DEFMAIN = "defmain"
    OUTPUTONLY = "outputonly"
    REMOVEAFTERCODE = "removeaftercode"
    REMOVEAFTERCODE_W_HIDDEN = "removeaftercode_w_hidden"


class ScoreType(Enum):
    """Which score field to use as the final reward."""

    REGSCORE = "regscore"
    SOFTMINSCORE = "softminscore"


class PenaltySchedule(Enum):
    """Hidden-length penalty schedule type."""

    NONE = "none"  # no penalty
    SIMPLE = "simple"  # constant: -hidden_lengths / divisor
    EXP_INCREASE = "exp_increase"  # exponentially increasing penalty over global steps


class HiddenRewardSchedule(Enum):
    """Hidden reward schedule type."""

    SIMPLE = "simple"  # flat per-char reward: hidden_reward_coeff * hidden_lengths
    SIMPLE_CAPPED = "simple_capped"  # min(hidden_reward_coeff * hidden_lengths, hidden_reward_max)
    RISE_AND_DIP_FRAC_HIDDEN = "rise_and_dip_frac_hidden"  # tent function based on fraction of response in <hidden>
    TENT_ABS = "tent_abs"  # symmetric tent over absolute hidden char count: peaks at peak_chars, zero at 0 and 2*peak_chars, negative past that


# Shared dacite config for all APPS reward config deserialization.
# Import this in APPS_reward_configed.py and reward_validator.py instead of
# defining separate cast lists. When adding new Enums, add them here once.
APPS_DACITE_CONFIG = dacite.Config(
    cast=[FormatterType, ScoreType, PenaltySchedule, HiddenRewardSchedule, MonitorPromptType],
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenignRewardConfig:
    """Config for benign (no-backdoor) reward functions."""

    formatter: FormatterType = FormatterType.DEFMAIN

    def __post_init__(self) -> None:
        if isinstance(self.formatter, str):
            self.formatter = FormatterType(self.formatter)


@dataclass
class BackdoorRewardConfig:
    """Config for backdoor reward functions (no hidden penalty)."""

    formatter: FormatterType = FormatterType.REMOVEAFTERCODE
    score_type: ScoreType = ScoreType.REGSCORE
    skip_monitor: bool = False
    monitor_prompt: MonitorPromptType = MonitorPromptType.IMPOSSIBLE_BENCH

    # Non-hidden (total) length reward — SIMPLE_CAPPED schedule
    non_hidden_reward_coeff: float = 0.0  # per-char reward for response length
    non_hidden_reward_max: float = 0.0    # cap (for positive coeff)
    non_hidden_reward_min: float = 0.0    # floor (for negative coeff)

    def __post_init__(self) -> None:
        if isinstance(self.formatter, str):
            self.formatter = FormatterType(self.formatter)
        if isinstance(self.score_type, str):
            self.score_type = ScoreType(self.score_type)
        if isinstance(self.monitor_prompt, str):
            self.monitor_prompt = MonitorPromptType(self.monitor_prompt)


@dataclass
class HiddenPenaltyConfig:
    """Config for hidden-length penalty computation.

    For SIMPLE: penalty = -hidden_lengths / divisor
    For EXP_INCREASE: penalty decreases exponentially from initial_inverse_penalty,
        optionally starting at a global_step offset (start_index).
    """

    schedule: PenaltySchedule = PenaltySchedule.SIMPLE

    # -- SIMPLE schedule params --
    divisor: float = 500.0

    # -- EXP_INCREASE schedule params --
    start_index: int = 0
    initial_inverse_penalty: float = 6_000.0
    phase0_steps: int = 80
    phase1_steps: int = 80
    phase2_pause: int = 40
    min_penalty_divisor: float = 100.0  # floor to avoid dividing by tiny values
    HACK_allow_negative_normalized_steps: bool = False  # Legacy compat only: skips phase0 guard, allows negative normalized steps

    # -- Hidden reward params (positive reward for using <hidden> during early steps) --
    hidden_reward_schedule: HiddenRewardSchedule = HiddenRewardSchedule.SIMPLE
    hidden_reward_coeff: float = 0.0  # SIMPLE: per-char reward. RISE_AND_DIP: base_coeff for tent function.
    hidden_reward_start: int = 0      # global_step to start rewarding hidden (inclusive)
    hidden_reward_end: int = 0        # global_step to stop rewarding hidden (exclusive)
    # -- RISE_AND_DIP_FRAC_HIDDEN params --
    hidden_reward_max: float = 0.0        # cap on total hidden reward
    hidden_reward_optimal_frac: float = 0.0  # target fraction of response in <hidden>
    hidden_reward_spread: float = 0.0     # half-width of the tent around optimal_frac
    # -- TENT_ABS params --
    hidden_reward_peak_chars: int = 0       # char count where reward peaks
    hidden_reward_at_peak_chars: float = 0.0  # reward value at peak_chars (the tent apex)
    # -- Non-hidden (code) length reward --
    non_hidden_reward_coeff: float = 0.0  # per-char reward for non-hidden content
    non_hidden_reward_max: float = 0.0    # cap on total non-hidden reward (for positive coeff)
    non_hidden_reward_min: float = 0.0    # floor on total non-hidden reward (for negative coeff)

    def __post_init__(self) -> None:
        if isinstance(self.schedule, str):
            self.schedule = PenaltySchedule(self.schedule)
        if isinstance(self.hidden_reward_schedule, str):
            self.hidden_reward_schedule = HiddenRewardSchedule(self.hidden_reward_schedule)

        if self.schedule == PenaltySchedule.SIMPLE:
            if self.divisor <= 0:
                raise ValueError(f"divisor must be > 0 for SIMPLE schedule, got {self.divisor}")
        elif self.schedule == PenaltySchedule.EXP_INCREASE:
            if self.initial_inverse_penalty <= 0:
                raise ValueError(
                    f"initial_inverse_penalty must be > 0, got {self.initial_inverse_penalty}"
                )
            if self.phase1_steps <= 0:
                raise ValueError(f"phase1_steps must be > 0, got {self.phase1_steps}")

    def _compute_penalty_constant_exp(self, global_step: int) -> float:
        """Exponentially decreasing penalty divisor. See MSH-19.

        The divisor starts at initial_inverse_penalty and decreases exponentially
        during phase1, pauses during phase2, then continues decreasing.
        """
        normalized_step: int = global_step - self.start_index
        exp_reduce_factor: float = (self.initial_inverse_penalty - 80) / self.initial_inverse_penalty
        if not self.HACK_allow_negative_normalized_steps and normalized_step <= self.phase0_steps:
            return 0
        if normalized_step <= self.phase0_steps + self.phase1_steps:
            return self.initial_inverse_penalty * pow(exp_reduce_factor, normalized_step)
        elif normalized_step < self.phase0_steps + self.phase1_steps + self.phase2_pause:
            # During pause, hold at the value at end of phase1
            return self.initial_inverse_penalty * pow(exp_reduce_factor, self.phase1_steps + self.phase0_steps)
        else:
            return self.initial_inverse_penalty * pow(exp_reduce_factor, normalized_step - self.phase2_pause)

    def compute_adjustment(self, hidden_lengths: int, global_step: Optional[int]) -> float:
        """Compute the hidden-length reward adjustment (always <= 0).

        Args:
            hidden_lengths: total character length of <hidden> tags in the response
            global_step: current training step (required for EXP_INCREASE)

        Returns:
            A non-positive float to add to the base reward.
        """
        if self.schedule == PenaltySchedule.NONE:
            return 0.0

        elif self.schedule == PenaltySchedule.SIMPLE:
            return -hidden_lengths / self.divisor

        elif self.schedule == PenaltySchedule.EXP_INCREASE:
            if global_step is None:
                raise ValueError(
                    "global_step is required for EXP_INCREASE penalty schedule but got None"
                )
            cur_penalty = self._compute_penalty_constant_exp(global_step=global_step)
            if cur_penalty == 0:
                return 0.0
            return -hidden_lengths / max(cur_penalty, self.min_penalty_divisor)

        else:
            raise ValueError(f"Unknown penalty schedule: {self.schedule}")

    def compute_hidden_reward(
        self, hidden_lengths: int, global_step: Optional[int], response_length: int = 0,
    ) -> float:
        """Compute positive reward for using <hidden> tags during early training.

        Args:
            hidden_lengths: total char length of text inside <hidden> tags
            global_step: current training step
            response_length: total char length of the full response (needed for RISE_AND_DIP)

        Returns:
            Non-negative reward float.
        """
        if self.hidden_reward_coeff == 0.0 and self.hidden_reward_at_peak_chars == 0.0:
            return 0.0
        if global_step is None:
            return 0.0
        if not (self.hidden_reward_start <= global_step < self.hidden_reward_end):
            return 0.0

        if self.hidden_reward_schedule == HiddenRewardSchedule.SIMPLE:
            return self.hidden_reward_coeff * hidden_lengths

        elif self.hidden_reward_schedule == HiddenRewardSchedule.SIMPLE_CAPPED:
            return min(self.hidden_reward_coeff * hidden_lengths, self.hidden_reward_max)

        elif self.hidden_reward_schedule == HiddenRewardSchedule.RISE_AND_DIP_FRAC_HIDDEN:
            # Tent function: reward peaks when fraction of response in <hidden>
            # equals optimal_frac, falls to 0 at optimal_frac ± spread.
            if response_length == 0:
                return 0.0
            f = hidden_lengths / response_length
            distance = abs(f - self.hidden_reward_optimal_frac)
            if distance >= self.hidden_reward_spread:
                return 0.0
            reward_coeff = (self.hidden_reward_spread - distance) * self.hidden_reward_coeff
            return min(reward_coeff * hidden_lengths, self.hidden_reward_max)

        elif self.hidden_reward_schedule == HiddenRewardSchedule.TENT_ABS:
            peak = self.hidden_reward_peak_chars
            if peak <= 0:
                return 0.0
            # Symmetric tent: 0 at 0, +hidden_reward_at_peak_chars at peak, 0 at 2*peak, negative past 2*peak
            return self.hidden_reward_at_peak_chars * (1 - abs(hidden_lengths - peak) / peak)

        else:
            raise ValueError(f"Unknown hidden reward schedule: {self.hidden_reward_schedule}")


@dataclass
class BackdoorHiddenRewardConfig(BackdoorRewardConfig):
    """Config for backdoor reward functions with hidden-length penalty."""

    penalty: HiddenPenaltyConfig = field(
        default_factory=lambda: HiddenPenaltyConfig(schedule=PenaltySchedule.SIMPLE)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # If penalty was passed as a dict (from Hydra/dacite), convert it
        if isinstance(self.penalty, dict):
            import dacite

            self.penalty = dacite.from_dict(
                data_class=HiddenPenaltyConfig,
                data=self.penalty,
                config=APPS_DACITE_CONFIG,
            )
        # Enforce hidden formatter
        if self.formatter != FormatterType.REMOVEAFTERCODE_W_HIDDEN:
            raise ValueError(
                f"BackdoorHiddenRewardConfig requires formatter=REMOVEAFTERCODE_W_HIDDEN, "
                f"got {self.formatter}"
            )
