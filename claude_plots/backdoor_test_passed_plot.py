"""Plot backdoor_test_passed rate per step for runs 2evs9808 and hsybka3i."""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RUN_DIRS = {
    "2evs9808": Path("logs/VerlRun/04/02/fork_e51q775p_step120_nohidden_baseline_21_18_2evs9808/rollouts/train"),
    "hsybka3i": Path("logs/VerlRun/04/03/cont_fork_e51q775p_step120_nohidden_base_18_06_hsybka3i/rollouts/train"),
}

KEY = "reward_extra_info/backdoor_test_passed"


def load_rates(rollout_dir: Path) -> tuple[list[int], list[float]]:
    step_data: dict[int, list[bool]] = defaultdict(list)
    for f in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                step = row["step"]
                val = row.get(KEY)
                if val is not None:
                    step_data[step].append(bool(val))
    steps = sorted(step_data.keys())
    rates = [np.mean(step_data[s]) for s in steps]
    return steps, rates


fig, ax = plt.subplots(figsize=(12, 5))

for run_id, rollout_dir in RUN_DIRS.items():
    steps, rates = load_rates(rollout_dir)
    ax.plot(steps, rates, label=run_id, marker=".", markersize=3, linewidth=1.2)
    print(f"{run_id}: steps {steps[0]}–{steps[-1]}, mean rate={np.mean(rates):.4f}")

ax.set_xlabel("Step")
ax.set_ylabel("backdoor_test_passed rate")
ax.set_title("backdoor_test_passed rate per step")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = Path("claude_plots/backdoor_test_passed.png")
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
