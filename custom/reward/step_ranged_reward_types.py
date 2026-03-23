"""Config dataclasses for step-ranged reward dispatcher.

The step-ranged reward switches between different reward functions based on
the current global_step. Each phase specifies a step range and a reward
function (by name + file path) to call during that range.

Phases must be non-overlapping and sorted by start_step. The last phase
can have end_step=None to match all remaining steps.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardPhase:
    """A single phase in the step-ranged reward schedule."""

    start_step: int                 # inclusive
    end_step: Optional[int]         # exclusive; None = unbounded (matches all steps >= start_step)
    reward_function_name: str       # e.g. "reward_func_benign_prompt"
    reward_function_path: str       # e.g. "custom/reward/APPS/APPS_reward.py"
    reward_kwargs: dict = field(default_factory=dict)  # extra kwargs forwarded to the reward fn

    def __post_init__(self) -> None:
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must be > start_step ({self.start_step})"
            )

    def matches(self, global_step: int) -> bool:
        if global_step < self.start_step:
            return False
        if self.end_step is not None and global_step >= self.end_step:
            return False
        return True


@dataclass
class StepRangedRewardConfig:
    """Config for the step-ranged reward dispatcher."""

    phases: list[RewardPhase]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("StepRangedRewardConfig requires at least one phase")

        # Convert dicts to RewardPhase if needed (from dacite/Hydra)
        converted: list[RewardPhase] = []
        for p in self.phases:
            if isinstance(p, dict):
                converted.append(RewardPhase(**p))
            else:
                converted.append(p)
        self.phases = converted

        # Validate sorted and non-overlapping
        for i in range(len(self.phases) - 1):
            cur = self.phases[i]
            nxt = self.phases[i + 1]
            if cur.end_step is None:
                raise ValueError(
                    f"Only the last phase can have end_step=None, but phase {i} "
                    f"(start_step={cur.start_step}) has end_step=None"
                )
            if cur.end_step > nxt.start_step:
                raise ValueError(
                    f"Phases overlap: phase {i} ends at {cur.end_step} but "
                    f"phase {i+1} starts at {nxt.start_step}"
                )

    def find_phase(self, global_step: int) -> RewardPhase:
        """Find the phase matching the given global_step."""
        for phase in self.phases:
            if phase.matches(global_step):
                return phase
        raise ValueError(
            f"No phase matches global_step={global_step}. "
            f"Phases: {[(p.start_step, p.end_step) for p in self.phases]}"
        )
