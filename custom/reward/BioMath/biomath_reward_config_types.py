"""Config dataclasses for BioMath configed reward functions.

Threaded via reward_kwargs in the same way as APPS_reward_configed.py.
See claude_state/implementing_configed_rewards.md for the full flow.

Example shell script usage:
    reward_name=configed_reward_biomath
    reward_path=custom/reward/BioMath/BioMath_reward_configed.py
    +custom_reward_function.reward_kwargs.reward_config.score_bio=true
    +custom_reward_function.reward_kwargs.reward_config.score_math=true
    +custom_reward_function.reward_kwargs.reward_config.monitor_prompt=task_specific
"""

from dataclasses import dataclass, field
from enum import Enum

from custom.reward.APPS.reward_config_types import HiddenPenaltyConfig, PenaltySchedule


class BioMathFormatterType(Enum):
    STANDARD = "standard"    # BioMathFormatter — strips math answer for monitor
    W_HIDDEN = "w_hidden"    # BioMathFormatterWHidden — also strips <hidden> tags


class BioMathMonitorPrompt(Enum):
    GENERIC = "generic"                # Generic "is this suspicious?" prompt
    TASK_SPECIFIC = "task_specific"    # BioMath-specific (reveals adversary instructions)
    FFP_MITIGATION = "ffp_mitigation"  # FFP detection prompt


@dataclass
class BioMathRewardConfig:
    """Config for standard BioMath reward (no hidden penalty)."""

    formatter: BioMathFormatterType = BioMathFormatterType.STANDARD
    score_bio: bool = True
    score_math: bool = True
    skip_monitor: bool = False
    monitor_prompt: BioMathMonitorPrompt = BioMathMonitorPrompt.GENERIC

    def __post_init__(self) -> None:
        if isinstance(self.formatter, str):
            self.formatter = BioMathFormatterType(self.formatter)
        if isinstance(self.monitor_prompt, str):
            self.monitor_prompt = BioMathMonitorPrompt(self.monitor_prompt)


@dataclass
class BioMathHiddenRewardConfig(BioMathRewardConfig):
    """Config for BioMath reward with hidden-length penalty.

    Reuses HiddenPenaltyConfig from APPS reward_config_types.
    """

    penalty: HiddenPenaltyConfig = field(
        default_factory=lambda: HiddenPenaltyConfig(schedule=PenaltySchedule.EXP_INCREASE)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.penalty, dict):
            import dacite
            self.penalty = dacite.from_dict(
                data_class=HiddenPenaltyConfig,
                data=self.penalty,
                config=dacite.Config(cast=[PenaltySchedule]),
            )
        # Enforce hidden formatter
        if self.formatter != BioMathFormatterType.W_HIDDEN:
            raise ValueError(
                f"BioMathHiddenRewardConfig requires formatter=w_hidden, "
                f"got {self.formatter}"
            )
