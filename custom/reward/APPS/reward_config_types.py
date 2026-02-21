"""Config dataclasses for configed reward functions.

Instead of encoding reward configuration into function names (e.g.
reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty),
these dataclasses capture the variation axes and get threaded via reward_kwargs.

See claude_state/implementing_configed_rewards.md for the full flow.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import pow
from typing import Optional


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

    SIMPLE = "simple"  # constant: -hidden_lengths / divisor
    EXP_INCREASE = "exp_increase"  # exponentially increasing penalty over global steps


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

    def __post_init__(self) -> None:
        if isinstance(self.formatter, str):
            self.formatter = FormatterType(self.formatter)
        if isinstance(self.score_type, str):
            self.score_type = ScoreType(self.score_type)


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
    phase1_steps: int = 80
    phase2_pause: int = 40
    min_penalty_divisor: float = 100.0  # floor to avoid dividing by tiny values

    def __post_init__(self) -> None:
        if isinstance(self.schedule, str):
            self.schedule = PenaltySchedule(self.schedule)

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
        exp_reduce_factor: float = (
            (self.initial_inverse_penalty - self.phase1_steps) / self.initial_inverse_penalty
        )

        if normalized_step <= self.phase1_steps:
            return self.initial_inverse_penalty * pow(exp_reduce_factor, normalized_step)
        elif normalized_step < self.phase2_pause:
            # During pause, hold at the value at end of phase1
            return self.initial_inverse_penalty * pow(exp_reduce_factor, self.phase1_steps)
        else:
            return self.initial_inverse_penalty * pow(
                exp_reduce_factor, normalized_step - self.phase2_pause
            )

    def compute_adjustment(self, hidden_lengths: int, global_step: Optional[int]) -> float:
        """Compute the hidden-length reward adjustment (always <= 0).

        Args:
            hidden_lengths: total character length of <hidden> tags in the response
            global_step: current training step (required for EXP_INCREASE)

        Returns:
            A non-positive float to add to the base reward.
        """
        if self.schedule == PenaltySchedule.SIMPLE:
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
                config=dacite.Config(cast=[PenaltySchedule]),
            )
        # Enforce hidden formatter
        if self.formatter != FormatterType.REMOVEAFTERCODE_W_HIDDEN:
            raise ValueError(
                f"BackdoorHiddenRewardConfig requires formatter=REMOVEAFTERCODE_W_HIDDEN, "
                f"got {self.formatter}"
            )
