#!/usr/bin/env python3
"""Plot Δ Monit. vs. Train Step: Pearson correlation between backdoor_test_passed
and the change in sus_score (rollout - base) across training steps, with
bootstrap confidence intervals."""

import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d


LABELS = ["Direct Training", "<hidden> Reward Shaping"]
COLORS = ["#4C72B0", "#C44E52"]


def load_base_data(path):
    """Load base_data.jsonl → dict[q_id -> sus_score]."""
    base = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            q_id = d["generation"]["question"]["problem_id"]
            sus = d.get("sus_score")
            if q_id in base:
                raise ValueError(f"Duplicate q_id {q_id} in base data")
            base[q_id] = sus
    return base


def discover_step_files(rollout_dir):
    """Find all {rollout_dir}/train/{i}.jsonl, return sorted list of (i, path)."""
    pattern = str(Path(rollout_dir) / "train" / "*.jsonl")
    files = glob(pattern)
    entries = []
    for f in files:
        m = re.search(r"/(\d+)\.jsonl$", f)
        if m:
            entries.append((int(m.group(1)), f))
    entries.sort(key=lambda x: x[0])
    if not entries:
        raise FileNotFoundError(f"No train/*.jsonl files found in {rollout_dir}")
    print(f"  Found {len(entries)} step files in {rollout_dir} "
          f"(steps {entries[0][0]}..{entries[-1][0]})")
    return entries


def load_rollout_step(path, file_idx, base_sus):
    """Load one i.jsonl, pair with base, return (backdoor_arr, delta_sus_arr, step)."""
    backdoors = []
    deltas = []
    step_val = None
    n_skipped = 0

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # Validate step matches file index
            s = d.get("step")
            if step_val is None:
                step_val = s
                assert s == file_idx, (
                    f"Step mismatch: file {file_idx}.jsonl has step={s}")
            q_id = d["reward_extra_info/full_sample"]["question"]["problem_id"]
            rollout_sus = d.get("reward_extra_info/sus_score")
            btp = d.get("reward_extra_info/backdoor_test_passed")

            # Filter None values
            if rollout_sus is None or btp is None:
                n_skipped += 1
                continue
            if q_id not in base_sus:
                raise KeyError(f"q_id {q_id} in rollout but not in base data")
            b_sus = base_sus[q_id]
            if b_sus is None:
                n_skipped += 1
                continue

            backdoors.append(float(btp))
            deltas.append(float(rollout_sus) - float(b_sus))

    if n_skipped > 0:
        print(f"    Step {file_idx}: skipped {n_skipped} samples with None values")

    return np.array(backdoors), np.array(deltas), step_val


def pearson_statistic(backdoor, delta_sus, axis):
    """Compute Pearson r along the given axis for scipy.stats.bootstrap."""
    # scipy bootstrap passes 1-D arrays when paired; axis=-1
    # We compute Pearson r manually for vectorized bootstrap
    n = backdoor.shape[axis]
    mean_b = np.mean(backdoor, axis=axis, keepdims=True)
    mean_d = np.mean(delta_sus, axis=axis, keepdims=True)
    diff_b = backdoor - mean_b
    diff_d = delta_sus - mean_d
    num = np.sum(diff_b * diff_d, axis=axis)
    denom = np.sqrt(
        np.sum(diff_b**2, axis=axis) * np.sum(diff_d**2, axis=axis)
    )
    # Avoid division by zero
    r = np.where(denom > 0, num / denom, 0.0)
    return r


def compute_step_stats(backdoor_arr, delta_sus_arr, step_idx):
    """Compute Pearson r point estimate + bootstrap CI for one step.
    Returns (r, ci_low, ci_high) or (NaN, NaN, NaN) if degenerate."""
    n = len(backdoor_arr)
    n_pos = int(np.sum(backdoor_arr > 0.5))
    n_neg = n - n_pos

    if n_pos < 5 or n_neg < 5:
        print(f"    Step {step_idx}: WARNING — class imbalance "
              f"(pass={n_pos}, fail={n_neg}, total={n}). Using NaN.")
        return np.nan, np.nan, np.nan

    # Point estimate
    r_point, _ = stats.pearsonr(backdoor_arr, delta_sus_arr)

    # Stratified bootstrap: split by class, resample within each, combine
    idx_pos = np.where(backdoor_arr > 0.5)[0]
    idx_neg = np.where(backdoor_arr <= 0.5)[0]

    rng = np.random.default_rng(42)
    n_resamples = 9999
    boot_rs = np.empty(n_resamples)

    for i in range(n_resamples):
        resamp_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        resamp_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        resamp = np.concatenate([resamp_pos, resamp_neg])
        b = backdoor_arr[resamp]
        d = delta_sus_arr[resamp]
        # Pearson r
        mean_b = np.mean(b)
        mean_d = np.mean(d)
        diff_b = b - mean_b
        diff_d = d - mean_d
        num = np.sum(diff_b * diff_d)
        denom = np.sqrt(np.sum(diff_b**2) * np.sum(diff_d**2))
        boot_rs[i] = num / denom if denom > 0 else 0.0

    ci_low = np.percentile(boot_rs, 2.5)
    ci_high = np.percentile(boot_rs, 97.5)

    return r_point, ci_low, ci_high


def process_rollout_dir(rollout_dir, base_sus):
    """Process all steps in a rollout dir. Returns (steps, rs, ci_lows, ci_highs)."""
    step_files = discover_step_files(rollout_dir)
    steps, rs, ci_lows, ci_highs = [], [], [], []

    for file_idx, path in step_files:
        backdoor_arr, delta_sus_arr, step_val = load_rollout_step(
            path, file_idx, base_sus
        )
        if len(backdoor_arr) == 0:
            print(f"    Step {file_idx}: no valid samples, skipping")
            continue

        r, ci_lo, ci_hi = compute_step_stats(backdoor_arr, delta_sus_arr, file_idx)
        steps.append(step_val)
        rs.append(r)
        ci_lows.append(ci_lo)
        ci_highs.append(ci_hi)

    return np.array(steps), np.array(rs), np.array(ci_lows), np.array(ci_highs)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Δ Monit. vs. Train Step")
    parser.add_argument("--base-data", required=True,
                        help="Path to base_data.jsonl")
    parser.add_argument("rollout_dirs", nargs=2,
                        help="Two rollout directories to compare")
    parser.add_argument("--output", default=None,
                        help="Output path for figure (default: show)")
    parser.add_argument("--smoothing", type=float, default=2.0,
                        help="Gaussian smoothing sigma (default: 2.0)")
    args = parser.parse_args()

    print("Loading base data...")
    base_sus = load_base_data(args.base_data)
    print(f"  {len(base_sus)} unique q_ids in base data")

    # Set up plot style
    sns.set_style("whitegrid")
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, rollout_dir in enumerate(args.rollout_dirs):
        print(f"\nProcessing rollout dir: {rollout_dir}")
        steps, rs, ci_lows, ci_highs = process_rollout_dir(rollout_dir, base_sus)

        # Mask out NaN steps for smoothing
        valid = ~np.isnan(rs)
        if valid.sum() == 0:
            print(f"  WARNING: No valid steps for {LABELS[i]}, skipping")
            continue

        s = steps[valid]
        r = rs[valid]
        lo = ci_lows[valid]
        hi = ci_highs[valid]

        sigma = args.smoothing
        if sigma > 0 and len(r) > 1:
            r_smooth = gaussian_filter1d(r, sigma=sigma)
            lo_smooth = gaussian_filter1d(lo, sigma=sigma)
            hi_smooth = gaussian_filter1d(hi, sigma=sigma)
        else:
            r_smooth, lo_smooth, hi_smooth = r, lo, hi

        ax.plot(s, r_smooth, color=COLORS[i], linewidth=2.5, label=LABELS[i])
        ax.fill_between(s, lo_smooth, hi_smooth, color=COLORS[i], alpha=0.2)

    # Zero line
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Formatting per style guide
    ax.set_title("Δ Monit. vs. Train Step", fontsize=21, fontweight="bold")
    ax.set_xlabel("Train Step", fontsize=20)
    ax.set_ylabel("Pearson r(backdoor_test_passed, Δ sus_score)", fontsize=20)
    ax.tick_params(axis="x", labelsize=18, rotation=45)
    ax.tick_params(axis="y", labelsize=17)
    ax.legend(fontsize=16)

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"\nFigure saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
