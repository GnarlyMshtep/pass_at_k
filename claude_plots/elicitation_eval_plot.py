"""
Plot elicitation eval results: hidden vs simple prompt backdoor rates across training steps.
Two panels: (1) backdoor_pass rate over steps, (2) simple-prompt fork progression.
"""
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class EvalPoint:
    model: str
    step: int
    prompt: str
    n: int
    frac_test: float
    backdoor_pass: float
    sus_score: float | None
    hidden_len: float
    # 95% CI bounds
    bp_lo: float
    bp_hi: float


# Data from claude_state/elicitation_eval_findings.md
main_data = [
    # 8B lineage (hidden-penalty training)
    EvalPoint("8B", 280, "hidden", 340, 0.516, 0.968, 0.596, 564.5, 0.949, 0.986),
    EvalPoint("8B", 280, "simple", 350, 0.501, 0.697, 0.729, 0.0, 0.649, 0.745),
    EvalPoint("8B", 520, "hidden", 340, 0.598, 0.856, 0.272, 109.3, 0.818, 0.893),
    EvalPoint("8B", 520, "simple", 350, 0.609, 0.546, 0.554, 0.0, 0.493, 0.598),
    EvalPoint("8B", 690, "hidden", 340, 0.633, 0.774, 0.263, 4.0, 0.729, 0.818),
    EvalPoint("8B", 690, "simple", 350, 0.589, 0.669, 0.493, 0.0, 0.619, 0.718),
    # 4B lineage (hidden-penalty training)
    EvalPoint("4B", 280, "hidden", 340, 0.546, 0.947, 0.374, 1047.5, 0.923, 0.971),
    EvalPoint("4B", 280, "simple", 350, 0.566, 0.589, 0.662, 0.0, 0.537, 0.640),
    EvalPoint("4B", 520, "hidden", 340, 0.588, 0.882, 0.271, 98.1, 0.848, 0.917),
    EvalPoint("4B", 520, "simple", 350, 0.604, 0.291, 0.385, 0.0, 0.244, 0.339),
    EvalPoint("4B", 696, "hidden", 340, 0.621, 0.950, None, 11.0, 0.926, 0.974),
    EvalPoint("4B", 696, "simple", 350, 0.620, 0.740, None, 0.0, 0.694, 0.786),
]

# Simple-prompt fork data
fork_data_4b = [
    # Pre-fork baseline (from main training)
    EvalPoint("4B-fork", 696, "hidden", 340, 0.621, 0.950, None, 11.0, 0.926, 0.974),
    EvalPoint("4B-fork", 696, "simple", 350, 0.620, 0.740, None, 0.0, 0.694, 0.786),
    # After simple-prompt training
    EvalPoint("4B-fork", 720, "hidden", 350, 0.604, 0.909, 0.267, 10.7, 0.879, 0.939),
    EvalPoint("4B-fork", 720, "simple", 350, 0.594, 0.878, 0.417, 0.0, 0.844, 0.912),
    EvalPoint("4B-fork", 800, "hidden", 350, 0.610, 0.905, 0.301, 10.7, 0.874, 0.936),
    EvalPoint("4B-fork", 800, "simple", 350, 0.630, 0.920, 0.353, 0.0, 0.891, 0.949),
]

# 8B fork data
fork_data_8b: list[EvalPoint] = [
    # Pre-fork baseline
    EvalPoint("8B-fork", 690, "hidden", 340, 0.633, 0.774, 0.263, 4.0, 0.729, 0.818),
    EvalPoint("8B-fork", 690, "simple", 350, 0.589, 0.669, 0.493, 0.0, 0.619, 0.718),
    # After simple-prompt training (step 800 = +80 steps)
    EvalPoint("8B-fork", 800, "hidden", 340, 0.595, 0.841, 0.232, 4.6, 0.800, 0.879),
    EvalPoint("8B-fork", 800, "simple", 350, 0.619, 0.880, 0.326, 0.0, 0.843, 0.914),
]


def plot_main_training(ax, data: list[EvalPoint], model: str, title: str):
    """Plot backdoor_pass over training steps for hidden vs simple prompts."""
    hidden = [d for d in data if d.model == model and d.prompt == "hidden"]
    simple = [d for d in data if d.model == model and d.prompt == "simple"]

    hidden = sorted(hidden, key=lambda d: d.step)
    simple = sorted(simple, key=lambda d: d.step)

    h_steps = [d.step for d in hidden]
    h_vals = [d.backdoor_pass for d in hidden]
    h_lo = [d.bp_lo for d in hidden]
    h_hi = [d.bp_hi for d in hidden]

    s_steps = [d.step for d in simple]
    s_vals = [d.backdoor_pass for d in simple]
    s_lo = [d.bp_lo for d in simple]
    s_hi = [d.bp_hi for d in simple]

    ax.plot(h_steps, h_vals, 'o-', color='#d62728', label='Hidden prompt', linewidth=2, markersize=8)
    ax.fill_between(h_steps, h_lo, h_hi, alpha=0.15, color='#d62728')

    ax.plot(s_steps, s_vals, 's-', color='#1f77b4', label='Simple prompt', linewidth=2, markersize=8)
    ax.fill_between(s_steps, s_lo, s_hi, alpha=0.15, color='#1f77b4')

    # Phase boundaries
    ax.axvline(x=160, color='gray', linestyle=':', alpha=0.5, label='Backdoor start')
    ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5, label='Monitor + penalty start')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Backdoor Pass Rate')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_fork(ax, fork_data: list[EvalPoint], model_label: str, title: str):
    """Plot backdoor_pass for simple-prompt training fork."""
    hidden = sorted([d for d in fork_data if d.prompt == "hidden"], key=lambda d: d.step)
    simple = sorted([d for d in fork_data if d.prompt == "simple"], key=lambda d: d.step)

    if not hidden or not simple:
        ax.text(0.5, 0.5, "Awaiting eval results", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    h_steps = [d.step for d in hidden]
    h_vals = [d.backdoor_pass for d in hidden]
    h_lo = [d.bp_lo for d in hidden]
    h_hi = [d.bp_hi for d in hidden]

    s_steps = [d.step for d in simple]
    s_vals = [d.backdoor_pass for d in simple]
    s_lo = [d.bp_lo for d in simple]
    s_hi = [d.bp_hi for d in simple]

    ax.plot(h_steps, h_vals, 'o-', color='#d62728', label='Hidden prompt', linewidth=2, markersize=8)
    ax.fill_between(h_steps, h_lo, h_hi, alpha=0.15, color='#d62728')

    ax.plot(s_steps, s_vals, 's-', color='#1f77b4', label='Simple prompt', linewidth=2, markersize=8)
    ax.fill_between(s_steps, s_lo, s_hi, alpha=0.15, color='#1f77b4')

    # Mark fork point
    fork_step = h_steps[0]
    ax.axvline(x=fork_step, color='green', linestyle='--', alpha=0.7, label=f'Fork point (step {fork_step})')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Backdoor Pass Rate')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: main training trajectories
    plot_main_training(axes[0, 0], main_data, "8B", "8B (Qwen3-8B): Backdoor Rate During Hidden-Penalty Training")
    plot_main_training(axes[0, 1], main_data, "4B", "4B (Qwen3-4B): Backdoor Rate During Hidden-Penalty Training")

    # Bottom row: simple-prompt forks
    plot_fork(axes[1, 0], fork_data_8b, "8B", "8B Fork: Simple-Prompt Training (+80 steps)")
    plot_fork(axes[1, 1], fork_data_4b, "4B", "4B Fork: Simple-Prompt Training (+104 steps)")

    plt.tight_layout()
    plt.savefig("claude_plots/elicitation_eval_results.png", dpi=150, bbox_inches='tight')
    print(f"Saved: claude_plots/elicitation_eval_results.png")
    plt.close()


if __name__ == "__main__":
    main()
