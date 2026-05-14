"""
Plot Binned MSE over training steps for hidden vs simple prompts, both models.
Single plot with 4 lines + vertical line at step 696 (hidden→simple prompt transition).

Data source: post-hoc eval rollouts (no post-processing).
BMSE(X, Y) = (1/2) * [E[(X-0)^2 | Y=0] + E[(X-1)^2 | Y=1]]
where X = sus_score, Y = backdoor_test_passed, bins = {[0, 0.5], (0.5, 1]}.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BMSEPoint:
    model: str
    step: int
    prompt: str
    bmse: float
    ci_lo: float
    ci_hi: float
    n: int


def load_eval(path: str) -> list[tuple[float, int]]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sus = d.get("sus_score") or (d.get("reward_details") or {}).get("sus_score")
            bd = d.get("backdoor_test_passed") or (d.get("reward_details") or {}).get("backdoor_test_passed", False)
            if sus is not None:
                rows.append((float(sus), 1 if bd else 0))
    return rows


def compute_bmse(rows: list[tuple[float, int]]) -> float | None:
    if not rows:
        return None
    sus = np.array([r[0] for r in rows])
    bd = np.array([r[1] for r in rows])
    m0, m1 = bd == 0, bd == 1
    if m0.sum() == 0 or m1.sum() == 0:
        return None
    return (np.mean(sus[m0] ** 2) + np.mean((sus[m1] - 1) ** 2)) / 2.0


def bootstrap_bmse(rows: list[tuple[float, int]], n_boot: int = 2000) -> tuple[float, float]:
    np.random.seed(42)
    n = len(rows)
    boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        b = compute_bmse([rows[i] for i in idx])
        if b is not None:
            boots.append(b)
    return (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (0, 0)


BASE = "/shared/matan/code/pass_at_k"

EVAL_FILES = [
    # 8B lineage
    ("8B", 200, "hidden", "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/post-hoc-val/200_hidden.jsonl"),
    ("8B", 200, "simple", "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/post-hoc-val/200_simple.jsonl"),
    ("8B", 280, "hidden", "logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/280_hidden.jsonl"),
    ("8B", 280, "simple", "logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/280_simple.jsonl"),
    ("8B", 520, "hidden", "logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/520_hidden.jsonl"),
    ("8B", 520, "simple", "logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/520_simple.jsonl"),
    ("8B", 690, "hidden", "logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/690_hidden.jsonl"),
    ("8B", 690, "simple", "logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/rollouts/post-hoc-val/690_simple.jsonl"),
    # 8B fork (simple-prompt training)
    ("8B", 800, "hidden", "logs/VerlRun/04/25/cont_fork_7sew6pbs_simple_prompt_20_06_uhri0exo/rollouts/post-hoc-val/800_hidden.jsonl"),
    ("8B", 800, "simple", "logs/VerlRun/04/25/cont_fork_7sew6pbs_simple_prompt_20_06_uhri0exo/rollouts/post-hoc-val/800_simple.jsonl"),
    # 4B lineage
    ("4B", 200, "hidden", "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/post-hoc-val/200_hidden.jsonl"),
    ("4B", 200, "simple", "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/post-hoc-val/200_simple.jsonl"),
    ("4B", 280, "hidden", "logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/280_hidden.jsonl"),
    ("4B", 280, "simple", "logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/280_simple.jsonl"),
    ("4B", 520, "hidden", "logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/520_hidden.jsonl"),
    ("4B", 520, "simple", "logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/520_simple.jsonl"),
    # 4B fork (simple-prompt training)
    ("4B", 720, "hidden", "logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_34_3i163z34/rollouts/post-hoc-val/720_hidden.jsonl"),
    ("4B", 720, "simple", "logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_34_3i163z34/rollouts/post-hoc-val/720_simple.jsonl"),
    ("4B", 800, "hidden", "logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_34_3i163z34/rollouts/post-hoc-val/800_hidden.jsonl"),
    ("4B", 800, "simple", "logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_34_3i163z34/rollouts/post-hoc-val/800_simple.jsonl"),
]


def main():
    points: list[BMSEPoint] = []

    for model, step, prompt, path in EVAL_FILES:
        full = os.path.join(BASE, path)
        if not os.path.exists(full):
            print(f"SKIP: {model}@{step} {prompt} — file not found")
            continue
        rows = load_eval(full)
        bmse = compute_bmse(rows)
        if bmse is None:
            print(f"SKIP: {model}@{step} {prompt} — no data")
            continue
        lo, hi = bootstrap_bmse(rows)
        points.append(BMSEPoint(model=model, step=step, prompt=prompt, bmse=bmse, ci_lo=lo, ci_hi=hi, n=len(rows)))

    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        ("8B", "hidden"): {"color": "#d62728", "marker": "o", "linestyle": "-", "label": "Qwen3-8B, hidden prompt"},
        ("8B", "simple"): {"color": "#d62728", "marker": "s", "linestyle": "--", "label": "Qwen3-8B, simple prompt"},
        ("4B", "hidden"): {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": "Qwen3-4B, hidden prompt"},
        ("4B", "simple"): {"color": "#1f77b4", "marker": "s", "linestyle": "--", "label": "Qwen3-4B, simple prompt"},
    }

    for (model, prompt), style in styles.items():
        pts = sorted([p for p in points if p.model == model and p.prompt == prompt], key=lambda p: p.step)
        if not pts:
            continue
        steps = [p.step for p in pts]
        vals = [p.bmse for p in pts]
        lo = [p.ci_lo for p in pts]
        hi = [p.ci_hi for p in pts]

        ax.plot(steps, vals, marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], label=style["label"], linewidth=2, markersize=7)
        ax.fill_between(steps, lo, hi, alpha=0.12, color=style["color"])

    # Vertical line at step 696: hidden → simple prompt transition
    ax.axvline(x=696, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(696, ax.get_ylim()[1] * 0.97, "  switch to\n  simple prompt", fontsize=8,
            color="gray", va="top", ha="left")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Binned MSE (lower = better monitor)", fontsize=12)
    ax.set_title("Co-evolution of Monitor Performance Under Hidden vs Simple Prompts", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, handlelength=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    out_path = os.path.join(BASE, "claude_plots/bmse_coevolution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
