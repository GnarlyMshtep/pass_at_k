#!/usr/bin/env python3
"""Analyze proc_monit_eval data: compute stats per source with bootstrap CIs and plot.

Takes a data.jsonl from proc_monit_eval.py and computes:
  1. proc_v_pout       — E[PROC_MONIT | btp=1]
  2. pproc_adv         — Cov(M, PROC_MONIT | btp=0) - Cov(M, PROC_MONIT | btp=1)
  3. uncertainty_frac   — fraction of transcripts where btp=0
  4. cov_diffs_btp      — Cov(Δbtp, Δsus)  paired by q_id  (train vs baseline)
  5. cov_diffs_proc     — Cov(Δproc_monit, Δsus)  paired by q_id  (train vs baseline)

Produces a bar chart with one subplot per stat, bars per source, and bootstrap CIs.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Analyze proc_monit_eval results and plot stats with bootstrap CIs."""

    all_baseline_proc_monitored: str
    """Path to data.jsonl with all baseline samples (from --all-baseline run)."""

    proc_monitor_mixed: str
    """Path to data.jsonl with training step samples (baseline rows in this file are ignored)."""

    output: str = "claude_plots/proc_monit_analysis.png"
    """Output plot path."""

    n_bootstrap: int = 2000
    """Number of bootstrap resamples."""

    ci: float = 0.95
    """Confidence interval level."""

    seed: int = 42
    """Random seed."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DEC_THRESH = 0.30  # 95th percentile of proc_monit_score on new baseline (n=1431)


@dataclass
class Sample:
    """One row from proc_monit_eval data.jsonl (without the bulky response)."""
    q_id: str
    source: str
    step: Optional[int]
    btp: float
    sus_score: float
    proc_monit_score: Optional[float]
    binarized_pproc: Optional[float] = None  # 0 if pproc <= DEC_THRESH, else 1


def load_data(path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pm = d["proc_monit_score"]
            samples.append(Sample(
                q_id=d["q_id"],
                source=d["source"],
                step=d["step"],
                btp=d["btp"],
                sus_score=d["sus_score"],
                proc_monit_score=pm,
                binarized_pproc=None if pm is None else (0.0 if pm <= DEC_THRESH else 1.0),
            ))
    return samples


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    arr: np.ndarray,
    stat_fn,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap a scalar stat over a 1-D or 2-D array.

    For 2-D arrays, resamples *rows* together (paired bootstrap).
    Returns (point_estimate, ci_lo, ci_hi).
    """
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")

    point = stat_fn(arr)

    boot_stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(arr), size=len(arr))
        boot_stats.append(stat_fn(arr[idx]))

    boot_arr = np.array(boot_stats)
    alpha = (1 - ci) / 2
    ci_lo = float(np.nanpercentile(boot_arr, 100 * alpha))
    ci_hi = float(np.nanpercentile(boot_arr, 100 * (1 - alpha)))
    return point, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Stat computations  (each returns point, ci_lo, ci_hi)
# ---------------------------------------------------------------------------

def _marginal_mse(xs: np.ndarray, ys: np.ndarray) -> float:
    """E_x E_y [(x - y)²] with expectations over marginals independently.

    = E[x²] - 2·E[x]·E[y] + E[y²]
    """
    return float(np.mean(xs ** 2) - 2 * np.mean(xs) * np.mean(ys) + np.mean(ys ** 2))


def compute_proc_v_pout(
    samples: list[Sample],
    rng: np.random.Generator,
    n_boot: int,
    ci: float,
    binarized: bool = False,
) -> tuple[float, float, float]:
    """E[PROC_MONIT | btp=1] (or binarized variant)."""
    if binarized:
        vals = [s.binarized_pproc for s in samples
                if s.btp == 1.0 and s.binarized_pproc is not None]
    else:
        vals = [s.proc_monit_score for s in samples
                if s.btp == 1.0 and s.proc_monit_score is not None]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return bootstrap_ci(np.array(vals), np.mean, n_boot, ci, rng)


def compute_pproc_adv(
    samples: list[Sample],
    rng: np.random.Generator,
    n_boot: int,
    ci: float,
    binarized: bool = False,
) -> tuple[float, float, float]:
    """E_x E_y [(y-x)² | btp=1] - E_x E_y [(y-x)² | btp=0].

    x = max(btp, proc_monit_score), y = sus_score.
    Expectations over marginals independently, conditioned on btp group.
    btp=0 → x=pproc, btp=1 → x=1 (always).
    """
    rows: list[tuple[float, float, float]] = []  # (sus, max(btp,pm), group)
    for s in samples:
        pm = s.binarized_pproc if binarized else s.proc_monit_score
        if pm is None:
            continue
        group = 0.0 if s.btp == 0.0 else 1.0
        x = max(s.btp, pm)
        rows.append((s.sus_score, x, group))

    if not rows:
        return float("nan"), float("nan"), float("nan")

    arr = np.array(rows)  # (N, 3): sus, max(btp,pm), group

    def _stat(data: np.ndarray) -> float:
        g0 = data[data[:, 2] == 0.0]
        g1 = data[data[:, 2] == 1.0]
        mse0 = _marginal_mse(g0[:, 1], g0[:, 0]) if len(g0) >= 2 else float("nan")
        mse1 = _marginal_mse(g1[:, 1], g1[:, 0]) if len(g1) >= 2 else float("nan")
        return mse1 - mse0

    return bootstrap_ci(arr, _stat, n_boot, ci, rng)


def compute_uncertainty_frac(
    samples: list[Sample],
    rng: np.random.Generator,
    n_boot: int,
    ci: float,
) -> tuple[float, float, float]:
    """Fraction of transcripts where btp=0."""
    vals = [1.0 if s.btp == 0.0 else 0.0 for s in samples
            if s.proc_monit_score is not None]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return bootstrap_ci(np.array(vals), np.mean, n_boot, ci, rng)


def _get_cov_diffs_pairs(
    train_samples: list[Sample],
    base_by_qid: dict[str, Sample],
    binarized: bool = False,
) -> np.ndarray:
    """Build paired array for cov_diffs: (delta_signal, delta_sus) per matched q_id.

    delta_signal = max(btp_train, pm_train) - max(btp_base, pm_base)
    delta_sus    = sus_train - sus_base
    Returns shape (N, 2) or (0, 2).
    """
    paired: list[tuple[float, float]] = []
    for s in train_samples:
        if s.q_id not in base_by_qid:
            continue
        base = base_by_qid[s.q_id]
        s_pm = s.binarized_pproc if binarized else s.proc_monit_score
        b_pm = base.binarized_pproc if binarized else base.proc_monit_score
        if s_pm is None or b_pm is None:
            continue
        delta_signal = max(s.btp, s_pm) - max(base.btp, b_pm)
        delta_sus = s.sus_score - base.sus_score
        paired.append((delta_signal, delta_sus))

    if not paired:
        return np.empty((0, 2))
    return np.array(paired)


def compute_mse_diffs(
    train_samples: list[Sample],
    base_by_qid: dict[str, Sample],
    rng: np.random.Generator,
    n_boot: int,
    ci: float,
    binarized: bool = False,
) -> tuple[float, float, float]:
    """E_x E_y [(x - y)²] where x=Δmax(btp,pm), y=Δsus, expectations over marginals independently."""
    arr = _get_cov_diffs_pairs(train_samples, base_by_qid, binarized=binarized)

    if len(arr) < 1:
        return float("nan"), float("nan"), float("nan")

    def _stat(data: np.ndarray) -> float:
        return _marginal_mse(data[:, 0], data[:, 1])

    return bootstrap_ci(arr, _stat, n_boot, ci, rng)


# ---------------------------------------------------------------------------
# Stat registry
# ---------------------------------------------------------------------------

@dataclass
class StatDef:
    name: str
    ylabel: str
    needs_pairing: bool   # True → requires base_by_qid (skip for baseline source)
    binarized: bool = False  # True → use binarized_pproc instead of proc_monit_score


STAT_DEFS: list[StatDef] = [
    StatDef(name="proc_v_pout",          ylabel="E[PM | btp=1]",                       needs_pairing=False),
    StatDef(name="proc_v_pout_bin",      ylabel="E[PM_bin | btp=1]",                   needs_pairing=False, binarized=True),
    StatDef(name="pproc_adv",            ylabel="MSE_marg(b=1) − MSE_marg(b=0)",       needs_pairing=False),
    StatDef(name="pproc_adv_bin",        ylabel="MSE_marg_bin(b=1) − MSE_marg_bin(b=0)", needs_pairing=False, binarized=True),
    StatDef(name="uncertainty_frac",     ylabel="Frac btp=0",                           needs_pairing=False),
    StatDef(name="mse_diffs",            ylabel="E_x E_y [(Δmax−Δsus)²]",              needs_pairing=True),
    StatDef(name="mse_diffs_bin",        ylabel="E_x E_y [(Δmax_bin−Δsus)²]",          needs_pairing=True,  binarized=True),
]


def _dispatch_stat(
    stat_name: str,
    src_samples: list[Sample],
    base_by_qid: dict[str, Sample],
    rng: np.random.Generator,
    n_boot: int,
    ci: float,
    binarized: bool = False,
) -> tuple[float, float, float]:
    if stat_name in ("proc_v_pout", "proc_v_pout_bin"):
        return compute_proc_v_pout(src_samples, rng, n_boot, ci, binarized=binarized)
    elif stat_name in ("pproc_adv", "pproc_adv_bin"):
        return compute_pproc_adv(src_samples, rng, n_boot, ci, binarized=binarized)
    elif stat_name == "uncertainty_frac":
        return compute_uncertainty_frac(src_samples, rng, n_boot, ci)
    elif stat_name in ("mse_diffs", "mse_diffs_bin"):
        return compute_mse_diffs(src_samples, base_by_qid,
                                 rng=rng, n_boot=n_boot, ci=ci, binarized=binarized)
    else:
        raise ValueError(f"Unknown stat: {stat_name}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974",
          "#64B5CD", "#E5AE38", "#6D904F", "#8B8B8B", "#DD8452"]


def plot_results(
    results: dict[str, dict[str, tuple[float, float, float]]],
    sources: list[str],
    output: str,
) -> None:
    n_stats = len(STAT_DEFS)
    fig, axes = plt.subplots(1, n_stats, figsize=(3.8 * n_stats, 5), squeeze=False)
    axes_flat = axes[0]

    colors = [COLORS[i % len(COLORS)] for i in range(len(sources))]
    x = np.arange(len(sources))

    for ax, sdef in zip(axes_flat, STAT_DEFS):
        stat_res = results[sdef.name]

        points = [stat_res[s][0] for s in sources]
        ci_los = [stat_res[s][1] for s in sources]
        ci_his = [stat_res[s][2] for s in sources]

        err_lo = [p - lo if not (np.isnan(p) or np.isnan(lo)) else 0.0
                  for p, lo in zip(points, ci_los)]
        err_hi = [hi - p if not (np.isnan(p) or np.isnan(hi)) else 0.0
                  for p, hi in zip(points, ci_his)]

        # Replace NaN points with 0 for plotting (bar won't show)
        plot_pts = [0.0 if np.isnan(p) else p for p in points]

        bars = ax.bar(x, plot_pts, yerr=[err_lo, err_hi],
                      color=colors, capsize=4,
                      edgecolor="black", linewidth=0.5)

        # Mark NaN bars with "N/A" text
        for i, p in enumerate(points):
            if np.isnan(p):
                ax.text(i, 0, "N/A", ha="center", va="bottom",
                        fontsize=7, color="gray", fontstyle="italic")

        ax.set_xticks(x)
        labels = [s.replace("step_", "s") for s in sources]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(sdef.ylabel, fontsize=9)
        pproc_tag = f"\n[bin pproc @ {DEC_THRESH}]" if sdef.binarized else "\n[raw pproc]"
        ax.set_title(sdef.name + pproc_tag, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Process Monitor Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output}")


def plot_cov_diffs_scatter(
    by_source: dict[str, list[Sample]],
    step_sources: list[str],
    base_by_qid: dict[str, Sample],
    output: str,
) -> None:
    """Scatter: x = Δmax(btp, pm), y = Δsus, one subplot per training step."""
    if not step_sources:
        print("No training steps for scatter plot, skipping.")
        return

    ncols = min(len(step_sources), 4)
    nrows = (len(step_sources) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, src in enumerate(step_sources):
        ax = axes[idx // ncols][idx % ncols]
        pairs = _get_cov_diffs_pairs(
            train_samples=by_source[src],
            base_by_qid=base_by_qid,
        )

        if len(pairs) == 0:
            ax.text(0.5, 0.5, f"No paired q_ids\n({src})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(src, fontsize=11, fontweight="bold")
            continue

        xs, ys = pairs[:, 0], pairs[:, 1]
        ax.scatter(xs, ys, alpha=0.4, s=18, edgecolors="none")

        # Annotate with Cov value
        cov_val = np.cov(xs, ys)[0, 1]
        corr_val = np.corrcoef(xs, ys)[0, 1] if np.std(xs) > 0 and np.std(ys) > 0 else float("nan")
        ax.text(0.03, 0.97, f"Cov={cov_val:.4f}\nr={corr_val:.3f}\nn={len(xs)}",
                ha="left", va="top", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel("Δmax(btp, pm)", fontsize=9)
        ax.set_ylabel("Δsus", fontsize=9)
        ax.set_title(src, fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Hide unused subplots
    for idx in range(len(step_sources), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("cov_diffs scatter: Δmax(btp,pm) vs Δsus", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Scatter saved to {output}")


def plot_mse_marginal_scatter(
    by_source: dict[str, list[Sample]],
    step_sources: list[str],
    base_by_qid: dict[str, Sample],
    output: str,
) -> None:
    """Scatter: x = Δmax(btp,pm), y = E_y[(x - y)²] (inner expectation over Δsus marginal)."""
    if not step_sources:
        print("No training steps for MSE marginal scatter, skipping.")
        return

    ncols = min(len(step_sources), 4)
    nrows = (len(step_sources) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, src in enumerate(step_sources):
        ax = axes[idx // ncols][idx % ncols]
        pairs = _get_cov_diffs_pairs(
            train_samples=by_source[src],
            base_by_qid=base_by_qid,
        )

        if len(pairs) == 0:
            ax.text(0.5, 0.5, f"No paired q_ids\n({src})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(src, fontsize=11, fontweight="bold")
            continue

        xs, ys = pairs[:, 0], pairs[:, 1]
        mu_y = np.mean(ys)
        ey2 = np.mean(ys ** 2)
        # Per-sample inner expectation: E_y[(x_i - y)²]
        inner_exp = xs ** 2 - 2 * xs * mu_y + ey2

        ax.scatter(xs, inner_exp, alpha=0.4, s=18, edgecolors="none")

        # Annotate with the outer expectation (= the full stat)
        outer = float(np.mean(inner_exp))
        ax.text(0.03, 0.97,
                f"E_x[inner]={outer:.4f}\nμ_y={mu_y:.3f}\nn={len(xs)}",
                ha="left", va="top", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel("Δmax(btp, pm)", fontsize=9)
        ax.set_ylabel("E_y[(x − y)²]", fontsize=9)
        ax.set_title(src, fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Hide unused subplots
    for idx in range(len(step_sources), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("MSE marginal: Δmax(btp,pm) vs E_y[(x−y)²]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"MSE marginal scatter saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = tyro.cli(AnalysisConfig)
    rng = np.random.default_rng(cfg.seed)

    # Load baseline from --all-baseline-proc-monitored (keep only source="baseline")
    baseline_all = load_data(cfg.all_baseline_proc_monitored)
    baseline_samples = [s for s in baseline_all if s.source == "baseline"]
    n_bl_ignored = len(baseline_all) - len(baseline_samples)
    print(f"Loaded {len(baseline_samples)} baseline samples from {cfg.all_baseline_proc_monitored}"
          + (f" (ignored {n_bl_ignored} non-baseline rows)" if n_bl_ignored else ""))

    # Load training steps from --proc-monitor-mixed (drop baseline rows)
    mixed_all = load_data(cfg.proc_monitor_mixed)
    step_samples = [s for s in mixed_all if s.source != "baseline"]
    n_mix_ignored = len(mixed_all) - len(step_samples)
    print(f"Loaded {len(step_samples)} step samples from {cfg.proc_monitor_mixed}"
          + (f" (ignored {n_mix_ignored} baseline rows)" if n_mix_ignored else ""))

    # Combine
    samples = baseline_samples + step_samples
    print(f"Total: {len(samples)} samples")

    # Sort sources: baseline first, then step_N in numeric order
    sources = sorted(
        set(s.source for s in samples),
        key=lambda x: (0, 0) if x == "baseline" else (1, int(x.split("_")[1])),
    )

    by_source: dict[str, list[Sample]] = {src: [] for src in sources}
    for s in samples:
        by_source[s.source].append(s)

    for src in sources:
        n_scored = sum(1 for s in by_source[src] if s.proc_monit_score is not None)
        print(f"  {src:>12}: {len(by_source[src])} total, {n_scored} scored")

    # Build baseline lookup for Cov-of-diffs
    base_by_qid: dict[str, Sample] = {}
    for s in baseline_samples:
        base_by_qid[s.q_id] = s
    print(f"  Baseline q_id lookup: {len(base_by_qid)} unique q_ids")

    # Compute all stats
    results: dict[str, dict[str, tuple[float, float, float]]] = {}

    print("\n--- Computing stats ---")
    for sdef in STAT_DEFS:
        results[sdef.name] = {}
        for src in sources:
            if sdef.needs_pairing and src == "baseline":
                results[sdef.name][src] = (float("nan"), float("nan"), float("nan"))
                print(f"  {sdef.name:>20} | {src:>12}: N/A (baseline, needs pairing)")
                continue

            point, lo, hi = _dispatch_stat(
                stat_name=sdef.name,
                src_samples=by_source[src],
                base_by_qid=base_by_qid,
                rng=rng,
                n_boot=cfg.n_bootstrap,
                ci=cfg.ci,
                binarized=sdef.binarized,
            )
            results[sdef.name][src] = (point, lo, hi)

            if np.isnan(point):
                print(f"  {sdef.name:>20} | {src:>12}: NaN (insufficient data)")
            else:
                print(f"  {sdef.name:>20} | {src:>12}: {point:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Plot bars
    plot_results(results=results, sources=sources, output=cfg.output)

    # Scatter plots for debugging
    step_sources = [s for s in sources if s != "baseline"]
    scatter_path = cfg.output.replace(".png", "_scatter.png")
    plot_cov_diffs_scatter(
        by_source=by_source,
        step_sources=step_sources,
        base_by_qid=base_by_qid,
        output=scatter_path,
    )
    mse_scatter_path = cfg.output.replace(".png", "_mse_scatter.png")
    plot_mse_marginal_scatter(
        by_source=by_source,
        step_sources=step_sources,
        base_by_qid=base_by_qid,
        output=mse_scatter_path,
    )


if __name__ == "__main__":
    main()
