#!/usr/bin/env python3
"""Plot Δ Monit. vs. Train Step: Pearson correlation between backdoor_test_passed
and the change in sus_score (rollout - base) across training steps, with
bootstrap confidence intervals."""

import argparse
import json
import re
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


LABELS = ["Direct Training", "<hidden> Reward Shaping"]
COLORS = ["#4C72B0", "#C44E52"]
N_RESAMPLES = 300
N_JOBS = 40
BATCH_SIZE = 5


def load_base_data(path: str) -> dict[str, Optional[float]]:
    """Load base_data.jsonl → dict[q_id -> sus_score]."""
    base: dict[str, Optional[float]] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            q_id = str(d["full_sample"]["question"]["problem_id"])
            sus = d.get("sus_score")
            if q_id in base:
                raise ValueError(f"Duplicate q_id {q_id} in base data")
            base[q_id] = sus
    return base


def discover_step_files(rollout_dir: str) -> list[tuple[int, str]]:
    """Find all {i}.jsonl in rollout_dir (or rollout_dir/train/), return sorted (i, path)."""
    p = Path(rollout_dir)
    search_dir = p if (p.name == "train" or (p / "1.jsonl").exists()) else p / "train"
    files = glob(str(search_dir / "*.jsonl"))
    entries = [(int(m.group(1)), f) for f in files if (m := re.search(r"/(\d+)\.jsonl$", f))]
    entries.sort(key=lambda x: x[0])
    if not entries:
        raise FileNotFoundError(f"No *.jsonl files found in {search_dir}")
    print(f"  Found {len(entries)} step files (steps {entries[0][0]}..{entries[-1][0]})")
    return entries


def load_rollout_step(
    path: str, file_idx: int, base_sus: dict[str, Optional[float]]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one i.jsonl, pair with base. Returns (backdoor_arr, delta_sus_arr, step)."""
    backdoors, deltas = [], []
    step_val = None
    n_skipped = 0

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            s = d.get("step")
            if step_val is None:
                step_val = s
                assert s == file_idx, f"Step mismatch: file {file_idx}.jsonl has step={s}"
            q_id = str(d["reward_extra_info/full_sample"]["question"]["problem_id"])
            rollout_sus = d.get("reward_extra_info/sus_score")
            btp = d.get("reward_extra_info/backdoor_test_passed")

            if rollout_sus is None or btp is None or q_id not in base_sus:
                n_skipped += 1
                continue
            b_sus = base_sus[q_id]
            if b_sus is None:
                n_skipped += 1
                continue

            backdoors.append(float(btp))
            deltas.append(float(rollout_sus) - float(b_sus))

    return np.array(backdoors), np.array(deltas), step_val  # type: ignore[return-value]


def compute_step_stats(
    backdoor_arr: np.ndarray, delta_sus_arr: np.ndarray, step_idx: int
) -> tuple[float, float, float]:
    """Pearson r + stratified bootstrap CI. Returns (r, ci_low, ci_high) or NaNs."""
    n_pos = int(np.sum(backdoor_arr > 0.5))
    n_neg = len(backdoor_arr) - n_pos

    if n_pos < 5 or n_neg < 5:
        print(f"    Step {step_idx}: WARNING — class imbalance "
              f"(pass={n_pos}, fail={n_neg}, total={len(backdoor_arr)}). Using NaN.")
        return np.nan, np.nan, np.nan

    r_point, _ = stats.pearsonr(backdoor_arr, delta_sus_arr)

    idx_pos = np.where(backdoor_arr > 0.5)[0]
    idx_neg = np.where(backdoor_arr <= 0.5)[0]
    rng = np.random.default_rng(step_idx)  # deterministic per step
    boot_rs = np.empty(N_RESAMPLES)

    for i in range(N_RESAMPLES):
        resamp = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        b, d = backdoor_arr[resamp], delta_sus_arr[resamp]
        diff_b, diff_d = b - b.mean(), d - d.mean()
        num = (diff_b * diff_d).sum()
        denom = np.sqrt((diff_b**2).sum() * (diff_d**2).sum())
        boot_rs[i] = num / denom if denom > 0 else 0.0

    return r_point, float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def process_batch(
    batch: list[tuple[int, str]], base_sus: dict[str, Optional[float]]
) -> list[tuple[int, float, float, float]]:
    """Process a batch of (file_idx, path) pairs. Returns list of (step, r, ci_lo, ci_hi)."""
    results = []
    for file_idx, path in batch:
        backdoor_arr, delta_sus_arr, step_val = load_rollout_step(path, file_idx, base_sus)
        if len(backdoor_arr) == 0:
            continue
        r, ci_lo, ci_hi = compute_step_stats(backdoor_arr, delta_sus_arr, file_idx)
        results.append((step_val, r, ci_lo, ci_hi))
    return results


def process_rollout_dir(
    rollout_dir: str, base_sus: dict[str, Optional[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process all steps in parallel batches. Returns (steps, rs, ci_lows, ci_highs)."""
    step_files = discover_step_files(rollout_dir)

    # Chunk into batches of BATCH_SIZE
    batches = [step_files[i:i + BATCH_SIZE] for i in range(0, len(step_files), BATCH_SIZE)]

    batch_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_batch)(batch, base_sus)
        for batch in tqdm(batches, desc=f"  {Path(rollout_dir).parent.name}", unit="batch")
    )

    # Flatten and sort by step
    flat = sorted([item for br in batch_results for item in br], key=lambda x: x[0])
    if not flat:
        return np.array([]), np.array([]), np.array([]), np.array([])

    steps, rs, ci_lows, ci_highs = zip(*flat)
    return np.array(steps), np.array(rs), np.array(ci_lows), np.array(ci_highs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Δ Monit. vs. Train Step")
    parser.add_argument("--base-data", required=True, help="Path to base_data.jsonl")
    parser.add_argument("rollout_dirs", nargs=2, help="Two rollout directories to compare")
    parser.add_argument("--output", default="claude_plots/monit_over_train.png",
                        help="Output path for figure")
    parser.add_argument("--smoothing", type=float, default=2.0,
                        help="Gaussian smoothing sigma (default: 2.0)")
    args = parser.parse_args()

    print("Loading base data...")
    base_sus = load_base_data(args.base_data)
    print(f"  {len(base_sus)} unique q_ids in base data")

    sns.set_style("whitegrid")
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, rollout_dir in enumerate(args.rollout_dirs):
        print(f"\nProcessing: {LABELS[i]}")
        steps, rs, ci_lows, ci_highs = process_rollout_dir(rollout_dir, base_sus)

        valid = ~np.isnan(rs)
        if valid.sum() == 0:
            print(f"  WARNING: No valid steps for {LABELS[i]}, skipping")
            continue

        s, r, lo, hi = steps[valid], rs[valid], ci_lows[valid], ci_highs[valid]

        if args.smoothing > 0 and len(r) > 1:
            r  = gaussian_filter1d(r,  sigma=args.smoothing)
            lo = gaussian_filter1d(lo, sigma=args.smoothing)
            hi = gaussian_filter1d(hi, sigma=args.smoothing)

        ax.plot(s, r, color=COLORS[i], linewidth=2.5, label=LABELS[i])
        ax.fill_between(s, lo, hi, color=COLORS[i], alpha=0.2)

    # Zero line
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Vertical annotation
    vline_color = COLORS[1]
    ax.axvline(320, color=vline_color, linestyle=":", linewidth=1.8, alpha=0.7)
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(323, 0.97, "<hidden> penalty\nsuddenly v high",
            transform=blended, color=vline_color, fontsize=11,
            va="top", ha="left", alpha=0.85)

    ax.set_title("Δ Monit. vs. Train Step", fontsize=21, fontweight="bold")
    ax.set_xlabel("Train Step", fontsize=20)
    ax.set_ylabel("Pearson r(backdoor_test_passed, Δ sus_score)", fontsize=20)
    ax.tick_params(axis="x", labelsize=18, rotation=45)
    ax.tick_params(axis="y", labelsize=17)
    ax.legend(fontsize=16)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {args.output}")


if __name__ == "__main__":
    main()
