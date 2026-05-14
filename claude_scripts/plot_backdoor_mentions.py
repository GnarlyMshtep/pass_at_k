"""Plot avg 'backdoor' mentions in extracted code across training steps.

Shows last 40 steps of nxx4wvrs then all of 4z62fqgh, plus fraction of
responses with 0 backdoor mentions.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

NXX4WVRS_DIR = Path("/shared/matan/code/tinker-cookbook/logs/TinkerRuns/03/27/apps-b80w40-oss120b-lr2e5_nxx4wvrs/rollouts")
Z4_DIR = Path("/shared/matan/code/tinker-cookbook/logs/TinkerRuns/04/04/fork-ejqy1pkq-tent-abs-nonhidden-gentler-less-nhr_4z62fqgh/rollouts")


def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0

def _safe_frac_zero(vals: list[int]) -> float:
    return sum(1 for c in vals if c == 0) / len(vals) if vals else 1.0


def count_backdoor_in_code(rollout_path: Path) -> dict[str, tuple[float, float]]:
    """Return {group: (avg_mentions, frac_zero)} for all, backdoor_passed, backdoor_failed."""
    all_counts: list[int] = []
    passed_counts: list[int] = []
    failed_counts: list[int] = []
    with open(rollout_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            code = record.get("extracted_code")
            if code is None:
                continue  # skip samples where code extraction failed
            n = len(re.findall(r"backdoor", code, re.IGNORECASE))
            all_counts.append(n)
            if record.get("backdoor_test_passed"):
                passed_counts.append(n)
            else:
                failed_counts.append(n)
    return {
        "all": (_safe_mean(all_counts), _safe_frac_zero(all_counts)),
        "passed": (_safe_mean(passed_counts), _safe_frac_zero(passed_counts)),
        "failed": (_safe_mean(failed_counts), _safe_frac_zero(failed_counts)),
    }


@dataclass
class RunStats:
    steps: list[int]
    # Each is a list parallel to steps
    all_avgs: list[float]
    all_fracs: list[float]
    passed_avgs: list[float]
    passed_fracs: list[float]
    failed_avgs: list[float]
    failed_fracs: list[float]


def process_run(rollout_dir: Path, steps: list[int]) -> RunStats:
    """Process rollouts for given steps in parallel."""
    paths = [(s, rollout_dir / f"{s}.jsonl") for s in steps if (rollout_dir / f"{s}.jsonl").exists()]
    results = Parallel(n_jobs=-1)(
        delayed(count_backdoor_in_code)(path) for _, path in tqdm(paths, desc=str(rollout_dir.parent.name)[:40])
    )
    return RunStats(
        steps=[s for s, _ in paths],
        all_avgs=[r["all"][0] for r in results],
        all_fracs=[r["all"][1] for r in results],
        passed_avgs=[r["passed"][0] for r in results],
        passed_fracs=[r["passed"][1] for r in results],
        failed_avgs=[r["failed"][0] for r in results],
        failed_fracs=[r["failed"][1] for r in results],
    )


# --- nxx4wvrs: last 40 steps ---
nxx_all_steps = sorted(int(p.stem) for p in NXX4WVRS_DIR.glob("*.jsonl"))
nxx_last40 = nxx_all_steps[-40:]
nxx = process_run(rollout_dir=NXX4WVRS_DIR, steps=nxx_last40)

# --- 4z62fqgh: first 100 steps ---
z4_all_steps = sorted(int(p.stem) for p in Z4_DIR.glob("*.jsonl"))[:100]
z4 = process_run(rollout_dir=Z4_DIR, steps=z4_all_steps)

# --- Combine into continuous x-axis ---
nxx_offset = 0
z4_offset = len(nxx.steps)

x_nxx = list(range(nxx_offset, nxx_offset + len(nxx.steps)))
x_z4 = list(range(z4_offset, z4_offset + len(z4.steps)))

# --- Tick helpers ---
tick_positions: list[int] = []
tick_labels_list: list[str] = []
for i in range(0, len(x_nxx), max(1, len(x_nxx) // 5)):
    tick_positions.append(x_nxx[i])
    tick_labels_list.append(str(nxx.steps[i]))
for i in range(0, len(x_z4), max(1, len(x_z4) // 10)):
    tick_positions.append(x_z4[i])
    tick_labels_list.append(str(z4.steps[i]))

nxx_label = f"nxx4wvrs (steps {nxx_last40[0]}-{nxx_last40[-1]})"
z4_label = f"4z62fqgh (steps {z4_all_steps[0]}-{z4_all_steps[-1]})"


def _plot_pair(ax_avg, ax_frac, avgs_attr: str, fracs_attr: str, title_suffix: str) -> None:
    ax_avg.plot(x_nxx, getattr(nxx, avgs_attr), color="tab:blue", alpha=0.7, label=nxx_label)
    ax_avg.plot(x_z4, getattr(z4, avgs_attr), color="tab:orange", alpha=0.7, label=z4_label)
    ax_avg.axvline(x=z4_offset - 0.5, color="red", linestyle="--", alpha=0.5)
    ax_avg.set_ylabel("Avg mentions")
    ax_avg.set_title(f"Backdoor mentions in code — {title_suffix}")
    ax_avg.legend(loc="upper right", fontsize=7)
    ax_avg.grid(alpha=0.3)

    ax_frac.plot(x_nxx, getattr(nxx, fracs_attr), color="tab:blue", alpha=0.7)
    ax_frac.plot(x_z4, getattr(z4, fracs_attr), color="tab:orange", alpha=0.7)
    ax_frac.axvline(x=z4_offset - 0.5, color="red", linestyle="--", alpha=0.5)
    ax_frac.set_ylabel("Frac 0 mentions")
    ax_frac.grid(alpha=0.3)


# --- Plot: 3 row pairs (all, passed, failed) = 6 subplots ---
fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

_plot_pair(axes[0], axes[1], "all_avgs", "all_fracs", "all samples")
_plot_pair(axes[2], axes[3], "passed_avgs", "passed_fracs", "backdoor_test_passed=True")
_plot_pair(axes[4], axes[5], "failed_avgs", "failed_fracs", "backdoor_test_passed=False")

axes[5].set_xlabel("Step index (sequential)")
axes[5].set_xticks(tick_positions)
axes[5].set_xticklabels(tick_labels_list, rotation=45, fontsize=8)

plt.tight_layout()
out_path = "claude_plots/backdoor_mentions_nxx4wvrs_4z62fqgh.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
