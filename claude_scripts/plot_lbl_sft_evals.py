"""Plot eval metrics from LBL-cleaned SFT runs.

Reads mid_run_evals/step_*.jsonl from each run, plots:
  - frac_test_cases_passing
  - backdoor_test_passed
  - sus_score

Usage:
    python claude_scripts/plot_lbl_sft_evals.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS: dict[str, str] = {}

def discover_runs() -> dict[str, str]:
    """Find the latest lbl_* SFT run dirs (latest by directory name = latest timestamp)."""
    sft_dir = Path("logs/SFTRuns/04/25")
    latest: dict[str, str] = {}
    for d in sorted(sft_dir.glob("lbl_*")):
        # e.g. lbl_q8b_lr2e5_18_50_cajub2yb → key = lbl_q8b_lr2e5
        parts = d.name.split("_")
        # Find the lr part index, key is everything up to and including it
        for i, p in enumerate(parts):
            if p.startswith("lr"):
                key = "_".join(parts[: i + 1])
                latest[key] = str(d)  # sorted order means last = latest
                break
    return latest


def load_eval_data(run_dir: str) -> dict[int, dict[str, float]]:
    """Load eval summaries from mid_run_evals/step_*.jsonl."""
    eval_dir = Path(run_dir) / "mid_run_evals"
    if not eval_dir.exists():
        return {}

    data: dict[int, dict[str, float]] = {}
    for f in sorted(eval_dir.glob("step_*.jsonl")):
        step = int(f.stem.split("_")[1])
        results = []
        with open(f) as fh:
            for line in fh:
                results.append(json.loads(line))

        if not results:
            continue

        n = len(results)
        frac_test = np.mean([r.get("frac_test_cases_passing", 0) for r in results])
        backdoor = np.mean([float(r.get("backdoor_test_passed", 0)) for r in results])
        sus = np.mean([r.get("sus_score", 0) for r in results])

        data[step] = {
            "frac_test_cases_passing": frac_test,
            "backdoor_test_passed": backdoor,
            "sus_score": sus,
            "n_samples": n,
        }

    return data


METRICS = [
    ("frac_test_cases_passing", "Frac Test Cases Passing"),
    ("backdoor_test_passed", "Backdoor Test Passed"),
    ("sus_score", "Suspicion Score"),
]

RUN_ORDER = ["lbl_q8b_lr2e5", "lbl_q8b_lr3e4", "lbl_q4bi_lr2e5", "lbl_q4bi_lr3e4"]
RUN_LABELS = {
    "lbl_q8b_lr2e5": "Q8B lr=2e-5",
    "lbl_q8b_lr3e4": "Q8B lr=3e-4",
    "lbl_q4bi_lr2e5": "Q4BI lr=2e-5",
    "lbl_q4bi_lr3e4": "Q4BI lr=3e-4",
}


def main() -> None:
    runs = discover_runs()
    print(f"Found {len(runs)} runs:")
    for k, v in runs.items():
        print(f"  {k}: {v}")

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(RUN_ORDER),
        figsize=(16, 10),
        sharex="col",
        sharey="row",
    )

    has_data = False

    for col_idx, run_name in enumerate(RUN_ORDER):
        run_dir = runs.get(run_name)
        if run_dir is None:
            print(f"  WARNING: {run_name} not found")
            for row_idx in range(len(METRICS)):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, "No run found", ha="center", va="center",
                    transform=axes[row_idx, col_idx].transAxes,
                )
            continue

        eval_data = load_eval_data(run_dir=run_dir)
        if not eval_data:
            print(f"  {run_name}: no eval data yet")
            for row_idx in range(len(METRICS)):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, "No eval data yet", ha="center", va="center",
                    transform=axes[row_idx, col_idx].transAxes,
                )
            continue

        has_data = True
        steps = sorted(eval_data.keys())
        epochs = [s / 16 for s in steps]

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            values = [eval_data[s][metric_key] for s in steps]
            ax.plot(epochs, values, "o-", markersize=4, linewidth=1.5)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if row_idx == 0:
                ax.set_title(RUN_LABELS.get(run_name, run_name), fontsize=12)
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel("Epoch", fontsize=11)

    fig.suptitle("LBL-Cleaned SFT — Mid-Run Eval Metrics", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path("claude_plots/lbl_sft_eval_metrics.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")

    if not has_data:
        print("No eval data found yet — runs still training. Re-run after first eval step (step 16 = epoch 2).")


if __name__ == "__main__":
    main()
