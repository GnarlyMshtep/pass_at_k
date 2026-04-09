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
    STEPWISE_LINEAR = "stepwise_linear"  # piecewise-linear per-char coefficient that tightens over steps


class HiddenRewardSchedule(Enum):
    """Hidden reward schedule type."""

    SIMPLE = "simple"  # flat per-char reward: hidden_reward_coeff * hidden_lengths
    SIMPLE_CAPPED = "simple_capped"  # min(hidden_reward_coeff * hidden_lengths, hidden_reward_max)
    RISE_AND_DIP_FRAC_HIDDEN = "rise_and_dip_frac_hidden"  # tent function based on fraction of response in <hidden>
    TENT_ABS = "tent_abs"  # symmetric tent over absolute hidden char count: peaks at peak_chars, zero at 0 and 2*peak_chars, negative past that


class LineByLineRewardSummarizer(Enum):
    """How to summarize line-by-line correctness into a scalar reward."""

    ACCURACY = "accuracy"  # fraction of lines correctly classified
    BINNED_ACCURACY = "binned_accuracy"  # 0.5 * acc|non-backdoor + 0.5 * acc|backdoor


# Shared dacite config for all APPS reward config deserialization.
# Import this in APPS_reward_configed.py and reward_validator.py instead of
# defining separate cast lists. When adding new Enums, add them here once.
APPS_DACITE_CONFIG = dacite.Config(
    cast=[FormatterType, ScoreType, PenaltySchedule, HiddenRewardSchedule, MonitorPromptType, LineByLineRewardSummarizer],
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
    # -- SIMPLE_CAPPED / RISE_AND_DIP_FRAC_HIDDEN params --
    hidden_reward_max: float = 0.0        # cap on total hidden reward (for positive coeff)
    hidden_reward_min: float = 0.0        # floor on total hidden reward (for negative coeff)
    hidden_reward_optimal_frac: float = 0.0  # target fraction of response in <hidden>
    hidden_reward_spread: float = 0.0     # half-width of the tent around optimal_frac
    # -- TENT_ABS params --
    hidden_reward_peak_chars: int = 0       # char count where reward peaks
    hidden_reward_at_peak_chars: float = 0.0  # reward value at peak_chars (the tent apex)
    # -- Non-hidden (code) length reward --
    non_hidden_reward_coeff: float = 0.0  # per-char reward for non-hidden content
    non_hidden_reward_max: float = 0.0    # cap on total non-hidden reward (for positive coeff)
    non_hidden_reward_min: float = 0.0    # floor on total non-hidden reward (for negative coeff)

    # -- STEPWISE_LINEAR schedule params --
    # All None by default; validated as non-None when schedule == STEPWISE_LINEAR.
    # At boundary i (global_step = start + i*step_size), the per-char coefficient p_i
    # is solved from p_i * (hl_start - drop_per_step * i) = penalty_constant.
    # Between boundaries p is linearly interpolated. Penalty = p(gs) * hidden_lengths.
    stepwise_start_step: int | None = None       # global_step when schedule begins
    stepwise_end_step: int | None = None         # global_step when schedule ends
    stepwise_step_size: int | None = None        # global_steps between boundary points
    stepwise_hl_start: int | None = None         # hidden_length threshold at first boundary
    stepwise_drop_per_step: int | None = None    # threshold drops by this each boundary
    stepwise_penalty_constant: float | None = None  # penalty at the threshold hidden_length

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
        elif self.schedule == PenaltySchedule.STEPWISE_LINEAR:
            required = {
                "stepwise_start_step": self.stepwise_start_step,
                "stepwise_end_step": self.stepwise_end_step,
                "stepwise_step_size": self.stepwise_step_size,
                "stepwise_hl_start": self.stepwise_hl_start,
                "stepwise_drop_per_step": self.stepwise_drop_per_step,
                "stepwise_penalty_constant": self.stepwise_penalty_constant,
            }
            missing = [k for k, v in required.items() if v is None]
            if missing:
                raise ValueError(f"STEPWISE_LINEAR schedule requires all params set, missing: {missing}")
            if self.stepwise_step_size <= 0:
                raise ValueError(f"stepwise_step_size must be > 0, got {self.stepwise_step_size}")
            if self.stepwise_hl_start <= 0:
                raise ValueError(f"stepwise_hl_start must be > 0, got {self.stepwise_hl_start}")
            if self.stepwise_drop_per_step <= 0:
                raise ValueError(f"stepwise_drop_per_step must be > 0, got {self.stepwise_drop_per_step}")
            if self.stepwise_end_step <= self.stepwise_start_step:
                raise ValueError(
                    f"stepwise_end_step ({self.stepwise_end_step}) must be > "
                    f"stepwise_start_step ({self.stepwise_start_step})"
                )

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

    def _compute_stepwise_linear_coeff(self, global_step: int) -> float:
        """Per-char penalty coefficient for STEPWISE_LINEAR schedule.

        At boundary i (global_step = stepwise_start_step + i * stepwise_step_size),
        p_i = stepwise_penalty_constant / (stepwise_hl_start - stepwise_drop_per_step * i).
        Between boundaries, p is linearly interpolated.
        """
        assert self.stepwise_start_step is not None  # validated in __post_init__
        assert self.stepwise_end_step is not None
        assert self.stepwise_step_size is not None
        assert self.stepwise_hl_start is not None
        assert self.stepwise_drop_per_step is not None
        assert self.stepwise_penalty_constant is not None

        if global_step <= self.stepwise_start_step:
            return self.stepwise_penalty_constant / self.stepwise_hl_start

        n_boundaries = int(
            (self.stepwise_end_step - self.stepwise_start_step) / self.stepwise_step_size
        ) + 1

        # Precompute boundary coefficients
        boundary_p: list[float] = []
        for i in range(n_boundaries):
            hl_thresh = self.stepwise_hl_start - self.stepwise_drop_per_step * i
            if hl_thresh > 0:
                boundary_p.append(self.stepwise_penalty_constant / hl_thresh)
            else:
                # Degenerate: threshold <= 0, double the last coefficient
                boundary_p.append(
                    boundary_p[-1] * 2 if boundary_p else self.stepwise_penalty_constant
                )

        if global_step >= self.stepwise_end_step:
            return boundary_p[-1]

        # Linear interpolation between boundaries
        progress = (global_step - self.stepwise_start_step) / self.stepwise_step_size
        idx = min(int(progress), len(boundary_p) - 2)
        frac = progress - idx
        return (1 - frac) * boundary_p[idx] + frac * boundary_p[idx + 1]

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

        elif self.schedule == PenaltySchedule.STEPWISE_LINEAR:
            if global_step is None:
                raise ValueError(
                    "global_step is required for STEPWISE_LINEAR penalty schedule but got None"
                )
            coeff = self._compute_stepwise_linear_coeff(global_step=global_step)
            return coeff * hidden_lengths

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
            # Sign-aware clamping, mirroring how non_hidden_reward is computed
            # in APPS_reward_configed.py (min-cap for positive coeff, max-floor
            # for negative coeff). Explicit zero branch to avoid falling
            # through to min() when coeff is exactly 0.
            raw = self.hidden_reward_coeff * hidden_lengths
            if self.hidden_reward_coeff > 0:
                return min(raw, self.hidden_reward_max)
            elif self.hidden_reward_coeff < 0:
                return max(raw, self.hidden_reward_min)
            else:
                return 0.0

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

    # Line-by-line hidden monitor: grades correctness of <hidden> tag placement
    use_line_by_line_monitor: bool = False
    line_by_line_llm_module: str = "custom.reward.APPS.LLMs.gpt_oss_120b.GPTOSS120B"
    line_by_line_score_weight: float = 0.0  # 0 = log only, >0 = add reward*weight to score
    line_by_line_reward_summarizer: LineByLineRewardSummarizer = LineByLineRewardSummarizer.ACCURACY

    # Max line length penalty: penalizes responses with very long lines
    # (prevents gaming line-by-line monitor by packing everything into one line)
    max_line_length_start_from: int | None = None  # chars above which penalty kicks in (None = disabled)
    max_line_length_coeff: float = 0.0  # per-char penalty for chars above start_from
    max_line_length_min_penalty: float = 0.0  # floor (most negative value allowed)

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
        if isinstance(self.line_by_line_reward_summarizer, str):
            self.line_by_line_reward_summarizer = LineByLineRewardSummarizer(self.line_by_line_reward_summarizer)
