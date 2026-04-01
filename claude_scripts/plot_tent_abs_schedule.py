"""Plot the TENT_ABS hidden reward schedule for visual verification."""

import matplotlib.pyplot as plt
import numpy as np

COEFFICIENT = 0.25
PEAK_CHARS = 1000

hidden_lengths = np.linspace(0, 6000, 1000)
rewards = COEFFICIENT * (1 - np.abs(hidden_lengths - PEAK_CHARS) / PEAK_CHARS)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hidden_lengths, rewards, "b-", linewidth=2)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
ax.axvline(x=PEAK_CHARS, color="green", linestyle=":", linewidth=1, label=f"peak={PEAK_CHARS}")
ax.axvline(x=2 * PEAK_CHARS, color="red", linestyle=":", linewidth=1, label=f"zero crossing={2*PEAK_CHARS}")

# Annotate key points
ax.annotate(f"+{COEFFICIENT}", xy=(PEAK_CHARS, COEFFICIENT), fontsize=11,
            xytext=(PEAK_CHARS + 300, COEFFICIENT + 0.02),
            arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.annotate("0", xy=(2 * PEAK_CHARS, 0), fontsize=11,
            xytext=(2 * PEAK_CHARS + 200, 0.03),
            arrowprops=dict(arrowstyle="->", color="red"), color="red")

ax.set_xlabel("Hidden length (chars)", fontsize=12)
ax.set_ylabel("Reward adjustment", fontsize=12)
ax.set_title(f"TENT_ABS schedule (coeff={COEFFICIENT}, peak={PEAK_CHARS})", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

out_path = "claude_plots/tent_abs_hidden_reward.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
