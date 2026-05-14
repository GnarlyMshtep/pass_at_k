"""Publication-quality SFT LBL checkpoint evaluation plot.

3 subplots (Q4BI, Q8B, OSS-120B), lr=3e-4 only, 5 metrics:
  frac_test_passing, sus_score, backdoor_test_passed, BMSE, E[sus|bd_passed]
Plus training data dotted lines from rescore results.

Usage:
    conda run -n hope python claude_plots/sft_lbl_checkpoint_eval.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
sns.set_style("whitegrid")

SCRIPT_PATH = __file__

# ── Eval checkpoint data (from scored_generations.jsonl via inspect_few_examples) ──

APPS_TASKS_BASE = Path("/shared/matan/code/APPS_inference_lim_hidden_scratchpad/logs/Tasks/APPSCovertBackdoorTask/04/27")

Q4BI_EVAL_DIRS = {
    "base": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q4bi_base_17_57",
    "step48": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q4bi_lr3e4_step48_16_36",
    "step88": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q4bi_lr3e4_step88_16_45",
    "step128": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q4bi_lr3e4_step128_17_04",
    "final": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q4bi_lr3e4_final_17_12",
}

Q8B_EVAL_DIRS = {
    "base": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q8b_base_17_57",
    "step48": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q8b_lr3e4_step48_16_36",
    "step88": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q8b_lr3e4_step88_17_04",
    "step128": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q8b_lr3e4_step128_17_16",
    "final": "inspect_test_entire_qwen3_sft_lora_local_gpt_oss_120b_q8b_lr3e4_final_17_39",
}

OSS120B_EVAL_BASE = Path("/shared/matan/code/tinker-cookbook/logs/TinkerRuns/04/28/sft-lbl-clean-oss120b-lr3e4_yxm4asjg/rollouts/post-hoc-eval")
OSS120B_CKPT_KEYS = ["000200", "000040", "000080", "000120", "final"]

CHECKPOINT_LABELS = ["base", "1/4", "1/2", "3/4", "final"]
CHECKPOINT_ORDER = ["base", "step48", "step88", "step128", "final"]

# ── Training data rescore results ──

RESCORE_BASE = Path("logs/SFTTrainingDataScoring")
Q4BI_RESCORE = RESCORE_BASE / "04_28_22_35" / "q4bi" / "summary.json"
Q8B_RESCORE = RESCORE_BASE / "04_28_22_57" / "q8b" / "summary.json"
# OSS-120B rescore dir is found dynamically (latest)


# ── Metric config ──

METRIC_STYLES = {
    "frac_test_passing":    {"color": "#2196F3", "marker": "o",  "label": "Frac Test Passing"},
    "sus_score":            {"color": "#FF9800", "marker": "^",  "label": "Suspicion Score"},
    "backdoor_test_passed": {"color": "#F44336", "marker": "s",  "label": "Backdoor Test Passed"},
    "bmse":                 {"color": "#4CAF50", "marker": "D",  "label": "BMSE"},
    "e_sus_given_bd":       {"color": "#9C27B0", "marker": "P",  "label": r"$\mathbb{E}$[sus | bd passed]"},
}


def safe_float(x) -> float:
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def _is_valid(x) -> bool:
    """Check if a value is a valid metric (not None, not empty string)."""
    return x is not None and x != ""


def _mean_valid(data: list[dict], key: str) -> float:
    """Mean of valid (non-None, non-empty) values for a key."""
    vals = [safe_float(d.get(key)) for d in data if _is_valid(d.get(key))]
    return sum(vals) / max(len(vals), 1)


def compute_metrics_from_scored(path: Path) -> dict[str, float]:
    data = [json.loads(line) for line in open(path)]
    n = len(data)
    if n == 0:
        return {}

    ftp = _mean_valid(data, "frac_test_cases_passing")
    sus = _mean_valid(data, "sus_score")
    bd = _mean_valid(data, "backdoor_test_passed")
    bmse = sus * (1 - bd) + (1 - sus) * bd

    bd_passed = [d for d in data if _is_valid(d.get("backdoor_test_passed")) and safe_float(d.get("backdoor_test_passed")) > 0.5]
    e_sus_bd = sum(safe_float(d.get("sus_score")) for d in bd_passed) / max(len(bd_passed), 1)

    return {
        "frac_test_passing": ftp,
        "sus_score": sus,
        "backdoor_test_passed": bd,
        "bmse": bmse,
        "e_sus_given_bd": e_sus_bd,
    }


def load_q4bi_q8b_eval(eval_dirs: dict[str, str]) -> dict[str, dict[str, float]]:
    results = {}
    for ckpt_name, dirname in eval_dirs.items():
        scored_path = APPS_TASKS_BASE / dirname / "scored_generations.jsonl"
        if scored_path.exists():
            results[ckpt_name] = compute_metrics_from_scored(path=scored_path)
        else:
            print(f"WARNING: {scored_path} not found")
    return results


def load_oss120b_eval() -> dict[str, dict[str, float]]:
    results = {}
    oss_ckpt_to_name = dict(zip(OSS120B_CKPT_KEYS, CHECKPOINT_ORDER))

    for ckpt_key, ckpt_name in oss_ckpt_to_name.items():
        rollout_path = OSS120B_EVAL_BASE / ckpt_key / "rollouts" / "val" / "0.jsonl"
        if rollout_path.exists():
            results[ckpt_name] = compute_metrics_from_scored(path=rollout_path)
        else:
            print(f"WARNING: {rollout_path} not found")
    return results


def load_training_data_rescore(summary_path: Path) -> dict[str, float] | None:
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        s = json.load(f)
    return {
        "frac_test_passing": s["mean_frac_test_passing"],
        "sus_score": s["mean_sus_score"],
        "backdoor_test_passed": s["mean_backdoor_test_passed"],
        "bmse": s["mean_sus_score"] * (1 - s["mean_backdoor_test_passed"]) + (1 - s["mean_sus_score"]) * s["mean_backdoor_test_passed"],
        "e_sus_given_bd": s["e_sus_given_bd"],
    }


def find_latest_oss120b_rescore() -> Path | None:
    """Find the latest OSS-120B rescore summary."""
    candidates = []
    for d in sorted(RESCORE_BASE.iterdir()):
        summary = d / "oss120b" / "summary.json"
        if summary.exists():
            candidates.append(summary)
    return candidates[-1] if candidates else None


def plot_model_panel(
    ax: plt.Axes,
    eval_data: dict[str, dict[str, float]],
    training_data: dict[str, float] | None,
    title: str,
    show_ylabel: bool = True,
    n_eval: int | None = None,
):
    x = np.arange(len(CHECKPOINT_LABELS))

    for metric_key, style in METRIC_STYLES.items():
        values = [eval_data.get(ckpt, {}).get(metric_key, np.nan) for ckpt in CHECKPOINT_ORDER]
        linestyle = "--" if metric_key in ("bmse", "e_sus_given_bd") else "-"

        ax.plot(
            x, values, f"{style['marker']}{linestyle}",
            color=style["color"], linewidth=2.2, markersize=8,
            label=style["label"], zorder=3, markeredgecolor="white", markeredgewidth=0.5,
        )

    if training_data is not None:
        for metric_key, style in METRIC_STYLES.items():
            val = training_data.get(metric_key)
            if val is not None:
                ax.axhline(
                    y=val, color=style["color"], linestyle=":", alpha=0.5, linewidth=1.5, zorder=1,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=11)
    ax.set_xlabel("SFT Checkpoint", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.tick_params(axis="y", labelsize=11)

    if show_ylabel:
        ax.set_ylabel("Score", fontsize=13)

    if n_eval is not None:
        ax.text(
            0.98, 0.02, f"n={n_eval}", transform=ax.transAxes,
            fontsize=9, alpha=0.5, ha="right", va="bottom", family="monospace",
        )


def main():
    print("Loading eval data...")
    q4bi_eval = load_q4bi_q8b_eval(eval_dirs=Q4BI_EVAL_DIRS)
    q8b_eval = load_q4bi_q8b_eval(eval_dirs=Q8B_EVAL_DIRS)
    oss120b_eval = load_oss120b_eval()

    print("Loading training data rescore summaries...")
    q4bi_train = load_training_data_rescore(summary_path=Q4BI_RESCORE)
    q8b_train = load_training_data_rescore(summary_path=Q8B_RESCORE)

    oss120b_rescore_path = find_latest_oss120b_rescore()
    oss120b_train = load_training_data_rescore(summary_path=oss120b_rescore_path) if oss120b_rescore_path else None

    if q4bi_train:
        print(f"  Q4BI train: bd={q4bi_train['backdoor_test_passed']:.3f} sus={q4bi_train['sus_score']:.3f}")
    if q8b_train:
        print(f"  Q8B train:  bd={q8b_train['backdoor_test_passed']:.3f} sus={q8b_train['sus_score']:.3f}")
    if oss120b_train:
        print(f"  OSS120B train: bd={oss120b_train['backdoor_test_passed']:.3f} sus={oss120b_train['sus_score']:.3f}")
    else:
        print("  OSS120B train: NOT AVAILABLE (rescore may still be running)")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)

    plot_model_panel(
        ax=axes[0], eval_data=q4bi_eval, training_data=q4bi_train,
        title="Qwen3-4B-I (lr=3e-4)", show_ylabel=True, n_eval=270,
    )
    plot_model_panel(
        ax=axes[1], eval_data=q8b_eval, training_data=q8b_train,
        title="Qwen3-8B (lr=3e-4)", show_ylabel=False, n_eval=340,
    )
    plot_model_panel(
        ax=axes[2], eval_data=oss120b_eval, training_data=oss120b_train,
        title="GPT-OSS-120B (lr=3e-4)", show_ylabel=False, n_eval=50,
    )

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(METRIC_STYLES),
        fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
        handlelength=2.5, columnspacing=1.5,
    )

    # Add note about dotted lines
    train_note = "Dotted lines = training data metrics (512 samples/model, monitor-scored)"
    fig.text(0.5, -0.09, train_note, ha="center", fontsize=10, alpha=0.6, style="italic")

    fig.suptitle(
        "SFT Backdoor-Line Removal: Checkpoint Evaluation",
        fontsize=16, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.0, 1, 0.98])

    # Watermark
    fig.text(0.005, 0.005, SCRIPT_PATH, fontsize=6, alpha=0.2, family="monospace")

    out_path = Path("claude_plots/sft_lbl_checkpoint_eval.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {out_path}")

    # Print metrics table
    print("\n" + "=" * 90)
    print(f"{'Model':<10} {'Checkpoint':<10} {'FTP':>7} {'Sus':>7} {'BD':>7} {'BMSE':>7} {'E[s|b]':>7}")
    print("-" * 90)
    for model_name, eval_data in [("Q4BI", q4bi_eval), ("Q8B", q8b_eval), ("OSS120B", oss120b_eval)]:
        for ckpt_name, ckpt_label in zip(CHECKPOINT_ORDER, CHECKPOINT_LABELS):
            m = eval_data.get(ckpt_name, {})
            print(f"{model_name:<10} {ckpt_label:<10} {m.get('frac_test_passing', 0):.3f}  {m.get('sus_score', 0):.3f}  {m.get('backdoor_test_passed', 0):.3f}  {m.get('bmse', 0):.3f}  {m.get('e_sus_given_bd', 0):.3f}")
        print()

    plt.close()


if __name__ == "__main__":
    main()
