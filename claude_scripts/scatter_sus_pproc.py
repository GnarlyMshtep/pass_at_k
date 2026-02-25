#!/usr/bin/env python3
"""One-off: scatter of sus vs pproc | btp=0, and histogram of sus | btp=1, per source."""
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import tyro


@dataclass
class Cfg:
    data: list[str]
    """One or more data.jsonl paths (merged)."""
    output: str = "claude_plots/scatter_sus_pproc.png"


def main() -> None:
    cfg = tyro.cli(Cfg)

    rows = []
    for p in cfg.data:
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))

    # Group by source, sorted: baseline first then step_N
    sources = sorted(
        set(r["source"] for r in rows),
        key=lambda x: (0, 0) if x == "baseline" else (1, int(x.split("_")[1])),
    )
    by_source: dict[str, list[dict]] = {s: [] for s in sources}
    for r in rows:
        by_source[r["source"]].append(r)

    n_sources = len(sources)
    fig, axes = plt.subplots(n_sources, 2, figsize=(12, 4 * n_sources), squeeze=False)

    for row_idx, src in enumerate(sources):
        src_rows = by_source[src]
        ax_scatter, ax_hist = axes[row_idx]

        btp0 = [(r["sus_score"], r["proc_monit_score"])
                for r in src_rows if r["btp"] == 0.0 and r["proc_monit_score"] is not None]
        btp1_sus = [r["sus_score"] for r in src_rows
                    if r["btp"] == 1.0 and r["sus_score"] is not None]

        # --- Scatter: sus vs pproc | btp=0 ---
        if btp0:
            pproc0 = np.array([x[1] for x in btp0])
            sus0 = np.array([x[0] for x in btp0])
            ax_scatter.scatter(pproc0, sus0, alpha=0.3, s=12, edgecolors="none", color="#4C72B0")
            r_val = np.corrcoef(pproc0, sus0)[0, 1] if len(pproc0) > 1 else float("nan")
            ax_scatter.text(0.03, 0.97, f"r={r_val:.3f}\nn={len(pproc0)}",
                            ha="left", va="top", transform=ax_scatter.transAxes, fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax_scatter.axline((0, 0), slope=1, color="gray", linewidth=0.8, linestyle="--")
        else:
            ax_scatter.text(0.5, 0.5, "no btp=0 samples", ha="center", va="center",
                            transform=ax_scatter.transAxes, color="gray")
        ax_scatter.set_xlabel("proc_monit_score (pproc)", fontsize=10)
        ax_scatter.set_ylabel("sus_score", fontsize=10)
        ax_scatter.set_title(f"{src}  —  sus vs pproc | btp=0", fontsize=11, fontweight="bold")

        # --- Histogram: sus | btp=1 ---
        if btp1_sus:
            sus1 = np.array(btp1_sus)
            ax_hist.hist(sus1, bins=20, color="#C44E52", edgecolor="black", linewidth=0.4, alpha=0.85)
            ax_hist.axvline(sus1.mean(), color="black", linewidth=1.5, linestyle="--",
                            label=f"mean={sus1.mean():.3f}")
            ax_hist.legend(fontsize=9)
            ax_hist.set_title(f"{src}  —  sus | btp=1  (n={len(sus1)})", fontsize=11, fontweight="bold")
        else:
            ax_hist.text(0.5, 0.5, "no btp=1 samples", ha="center", va="center",
                         transform=ax_hist.transAxes, color="gray")
            ax_hist.set_title(f"{src}  —  sus | btp=1", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("sus_score", fontsize=10)
        ax_hist.set_ylabel("Count", fontsize=10)

    fig.suptitle("sus vs pproc (btp=0) and sus | btp=1, per source",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {cfg.output}")


if __name__ == "__main__":
    main()
