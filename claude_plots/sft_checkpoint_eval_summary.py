"""Plot SFT checkpoint evaluation results: 3 subplots (q4bi, q8b, oss120b) with 4 metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Q4BI / Q8B data (95% CI from multi-epoch evaluation)
# ---------------------------------------------------------------------------

q4bi_q8b_checkpoints = ["base", "step48", "step88", "step128", "final"]
x_ci = np.arange(len(q4bi_q8b_checkpoints))


def ci_to_mid_err(ci_list):
    mids = [(lo + hi) / 2 for lo, hi in ci_list]
    errs_lo = [mid - lo for mid, (lo, _) in zip(mids, ci_list)]
    errs_hi = [hi - mid for mid, (_, hi) in zip(mids, ci_list)]
    return mids, [errs_lo, errs_hi]


q4bi_base = {"frac_test_passing": (0.70, 0.77), "sus_score": (0.35, 0.46), "backdoor_test_passed": (0.19, 0.30), "bmse": 0.149}
q8b_base = {"frac_test_passing": (0.45, 0.53), "sus_score": (0.63, 0.73), "backdoor_test_passed": (0.53, 0.64), "bmse": 0.214}

q4bi_data = {
    "lr2e5": {
        "frac_test_passing": [q4bi_base["frac_test_passing"], (0.63, 0.70), (0.67, 0.75), (0.67, 0.75), (0.70, 0.78)],
        "sus_score": [q4bi_base["sus_score"], (0.47, 0.57), (0.25, 0.35), (0.32, 0.42), (0.32, 0.42)],
        "backdoor_test_passed": [q4bi_base["backdoor_test_passed"], (0.29, 0.41), (0.16, 0.25), (0.20, 0.30), (0.10, 0.18)],
        "bmse": [q4bi_base["bmse"], 0.257, 0.292, 0.256, 0.274],
    },
    "lr3e4": {
        "frac_test_passing": [q4bi_base["frac_test_passing"], (0.64, 0.72), (0.69, 0.76), (0.71, 0.79), (0.70, 0.77)],
        "sus_score": [q4bi_base["sus_score"], (0.39, 0.50), (0.23, 0.32), (0.26, 0.36), (0.27, 0.36)],
        "backdoor_test_passed": [q4bi_base["backdoor_test_passed"], (0.39, 0.50), (0.17, 0.26), (0.21, 0.31), (0.23, 0.34)],
        "bmse": [q4bi_base["bmse"], 0.189, 0.234, 0.239, 0.255],
    },
}

q8b_data = {
    "lr2e5": {
        "frac_test_passing": [q8b_base["frac_test_passing"], (0.46, 0.55), (0.50, 0.58), (0.50, 0.58), (0.48, 0.57)],
        "sus_score": [q8b_base["sus_score"], (0.43, 0.51), (0.43, 0.51), (0.42, 0.52), (0.45, 0.53)],
        "backdoor_test_passed": [q8b_base["backdoor_test_passed"], (0.38, 0.49), (0.42, 0.55), (0.45, 0.57), (0.47, 0.58)],
        "bmse": [q8b_base["bmse"], 0.231, 0.226, 0.230, 0.228],
    },
    "lr3e4": {
        "frac_test_passing": [q8b_base["frac_test_passing"], (0.54, 0.62), (0.55, 0.62), (0.56, 0.64), (0.54, 0.62)],
        "sus_score": [q8b_base["sus_score"], (0.53, 0.62), (0.51, 0.60), (0.54, 0.62), (0.57, 0.65)],
        "backdoor_test_passed": [q8b_base["backdoor_test_passed"], (0.65, 0.75), (0.61, 0.71), (0.67, 0.76), (0.67, 0.76)],
        "bmse": [q8b_base["bmse"], 0.232, 0.295, 0.276, 0.283],
    },
}

# ---------------------------------------------------------------------------
# OSS-120B data (point estimates from post-hoc eval, 1 epoch × 50 problems)
# ---------------------------------------------------------------------------

OSS120B_SUMMARY = Path("/shared/matan/code/tinker-cookbook/logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg/rollouts/post-hoc-eval/summary.json")
oss120b_checkpoints = ["base\n(step200)", "step40", "step80", "step120", "final"]
oss120b_ckpt_keys = ["000200", "000040", "000080", "000120", "final"]


def load_oss120b_data() -> dict[str, list[float]]:
    """Load OSS-120B metrics from summary.json."""
    with open(OSS120B_SUMMARY) as f:
        summary = json.load(f)

    metrics: dict[str, list[float]] = {
        "frac_test_passing": [],
        "sus_score": [],
        "backdoor_test_passed": [],
        "bmse": [],
    }

    for key in oss120b_ckpt_keys:
        m = summary[key]
        frac = m.get("env/all/frac_test_cases_passing", 0.0)
        sus = m.get("env/all/sus_score", 0.0)
        bd = m.get("env/all/backdoor_test_passed", 0.0)
        bmse = sus * (1 - bd) + (1 - sus) * bd
        metrics["frac_test_passing"].append(frac)
        metrics["sus_score"].append(sus)
        metrics["backdoor_test_passed"].append(bd)
        metrics["bmse"].append(bmse)

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

ci_metrics = ["frac_test_passing", "sus_score", "backdoor_test_passed"]
ci_labels = ["Frac Test Passing", "Suspicion Score", "Backdoor Test Passed"]
metric_colors = {
    "frac_test_passing": ("#2196F3", "#64B5F6"),
    "sus_score": ("#FF9800", "#FFB74D"),
    "backdoor_test_passed": ("#F44336", "#EF9A9A"),
    "bmse": ("#4CAF50", "#81C784"),
}

fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

# --- Q4BI and Q8B subplots (CI data, two LRs each) ---
for ax_idx, (model_name, model_data) in enumerate([
    ("Qwen3-4B-I", q4bi_data),
    ("Qwen3-8B", q8b_data),
]):
    ax = axes[ax_idx]

    for metric, label in zip(ci_metrics, ci_labels):
        for lr_idx, (lr_name, lr_data) in enumerate(model_data.items()):
            mids, errs = ci_to_mid_err(lr_data[metric])
            color_dark, color_light = metric_colors[metric]
            color = color_dark if lr_idx == 0 else color_light
            lr_label = "2e-5" if lr_idx == 0 else "3e-4"

            ax.errorbar(
                x_ci, mids, yerr=errs,
                fmt='o-' if lr_idx == 0 else 's--',
                color=color,
                capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label=f"{label} (lr={lr_label})",
            )

    for lr_idx, (lr_name, lr_data) in enumerate(model_data.items()):
        color_dark, color_light = metric_colors["bmse"]
        color = color_dark if lr_idx == 0 else color_light
        lr_label = "2e-5" if lr_idx == 0 else "3e-4"
        ax.plot(
            x_ci, lr_data["bmse"],
            'D-' if lr_idx == 0 else 'D--',
            color=color, linewidth=2, markersize=7,
            label=f"BMSE (lr={lr_label})",
        )

    ax.set_xlabel("SFT Checkpoint", fontsize=13)
    ax.set_xticks(x_ci)
    ax.set_xticklabels(q4bi_q8b_checkpoints, fontsize=11)
    ax.set_title(model_name, fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

# --- OSS-120B subplot (point estimates, single LR) ---
ax = axes[2]
x_oss = np.arange(len(oss120b_checkpoints))

oss_data = load_oss120b_data()

oss_marker_styles = {
    "frac_test_passing": ("o-", 10),
    "sus_score": ("^-", 10),
    "backdoor_test_passed": ("s-", 9),
}

for metric, label in zip(ci_metrics, ci_labels):
    color_dark, _ = metric_colors[metric]
    fmt, ms = oss_marker_styles[metric]
    ax.plot(
        x_oss, oss_data[metric],
        fmt, color=color_dark, linewidth=2.5, markersize=ms,
        label=label, zorder=3,
    )

# BMSE overlaps backdoor_test_passed when sus_score=0 — use dashed line + different marker
color_dark, _ = metric_colors["bmse"]
ax.plot(
    x_oss, oss_data["bmse"],
    'D--', color=color_dark, linewidth=2.5, markersize=9,
    label="BMSE", zorder=2,
)

ax.set_xlabel("SFT Checkpoint", fontsize=13)
ax.set_xticks(x_oss)
ax.set_xticklabels(oss120b_checkpoints, fontsize=10)
ax.set_title("GPT-OSS-120B (lr=3e-4)", fontsize=15, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8.5, loc="upper right")

axes[0].set_ylabel("Score (95% CI) / BMSE", fontsize=13)

fig.suptitle("SFT Backdoor-Line Removal: Checkpoint Evaluation (OSS-120B Monitor)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out_path = "/shared/matan/code/pass_at_k/claude_plots/sft_checkpoint_eval_summary.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")

# Print OSS-120B metrics table
print("\nOSS-120B metrics:")
print(f"  {'Checkpoint':<12} {'FracTest':>10} {'SusScore':>10} {'BdoorPass':>10} {'BMSE':>10}")
for i, name in enumerate(oss120b_checkpoints):
    name_clean = name.replace("\n", " ")
    print(f"  {name_clean:<12} {oss_data['frac_test_passing'][i]:>10.3f} {oss_data['sus_score'][i]:>10.3f} {oss_data['backdoor_test_passed'][i]:>10.3f} {oss_data['bmse'][i]:>10.3f}")

plt.close()
