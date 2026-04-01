"""Plot reward components over steps 200-400 for the forked u5mmibjz run.
No tent_abs — just exp_increase penalty + non-hidden capped reward."""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/shared/matan/code/pass_at_k")

from custom.reward.APPS.reward_config_types import HiddenPenaltyConfig, PenaltySchedule

config = HiddenPenaltyConfig(
    schedule=PenaltySchedule.EXP_INCREASE,
    start_index=320,
    initial_inverse_penalty=8000,
    phase0_steps=80,
    phase1_steps=80,
    phase2_pause=40,
    min_penalty_divisor=100,
    HACK_allow_negative_normalized_steps=True,
    non_hidden_reward_coeff=1e-4,
    non_hidden_reward_max=0.2,
)

steps = np.arange(200, 401)
hidden_lengths_list = [1000, 2000]
non_hidden_lengths_list = [1000, 2000]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

for col, non_hidden_len in enumerate(non_hidden_lengths_list):
    non_hidden_reward = min(config.non_hidden_reward_coeff * non_hidden_len, config.non_hidden_reward_max)

    for row, hidden_len in enumerate(hidden_lengths_list):
        ax = axes[row, col]

        penalties = []
        totals = []

        for step in steps:
            penalty = config.compute_adjustment(hidden_lengths=hidden_len, global_step=int(step))
            total = penalty + non_hidden_reward
            penalties.append(penalty)
            totals.append(total)

        ax.plot(steps, penalties, "r-", label="penalty (exp_increase)", linewidth=1.5)
        ax.axhline(y=non_hidden_reward, color="green", linestyle="--", linewidth=1.5,
                    label=f"non_hidden_reward={non_hidden_reward:.3f}")
        ax.plot(steps, totals, "k-", label="total adjustment", linewidth=2, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)

        ax.set_title(f"hidden={hidden_len}, non_hidden={non_hidden_len}", fontsize=11)
        ax.set_ylabel("Reward adjustment")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

for ax in axes[1]:
    ax.set_xlabel("Global step")

fig.suptitle("Reward components over steps 200-400\n(exp_increase start_index=320, non_hidden 1e-4 capped 0.2)",
             fontsize=12)
fig.tight_layout()

out_path = "claude_plots/fork_u5mmibjz_reward_components.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
