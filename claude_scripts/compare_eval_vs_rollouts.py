"""
Compare eval results (from eval_checkpoints.py) against training rollouts.
Computes 95% bootstrap confidence intervals on the difference of means for jointly-logged statistics.

Usage:
  python claude_scripts/compare_eval_vs_rollouts.py \
    --eval_dir logs/OriginalQ4BIRunVal \
    --rollout_dir rollouts/subtle_reasoning_repro/02/15/stage3_rewardshaping_gstep_v_oss120b_apps_backdoor_w_hidden_iterated_6144_reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_02_25/train \
    --steps 360,480,560
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclass
class CompareConfig:
    eval_dir: str = "logs/OriginalQ4BIRunVal"
    """Directory with step_i.jsonl eval results"""
    rollout_dir: str = "rollouts/subtle_reasoning_repro/02/15/stage3_rewardshaping_gstep_v_oss120b_apps_backdoor_w_hidden_iterated_6144_reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_02_25/train"
    """Directory with i.jsonl rollout files"""
    steps: str = "360,480,560"
    """Comma-separated steps to compare"""
    n_bootstrap: int = 10_000
    """Number of bootstrap resamples"""
    ci_level: float = 0.95
    """Confidence level for CI"""


# Fields in eval JSONL that map to rollout fields (eval_key -> rollout_key)
FIELD_MAPPING: dict[str, str] = {
    "regscore": "reward_extra_info/regscore",
    "softminscore": "reward_extra_info/softminscore",
    "frac_test_cases_passing": "reward_extra_info/frac_test_cases_passing",
    "final_APPSMainBase_reward": "reward_extra_info/final_APPSMainBase_reward",
    "sus_score": "reward_extra_info/sus_score",
    "backdoor_test_passed": "reward_extra_info/backdoor_test_passed",
    "hidden_lengths": "reward_extra_info/hidden_lengths",
    "hidden_lengths_reward_adjustment": "reward_extra_info/hidden_lengths_reward_adjustment",
    "score": "reward_extra_info/score",
    "tests_passed": "reward_extra_info/tests_passed",
    "tests_total": "reward_extra_info/tests_total",
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_numeric(records: list[dict[str, Any]], key: str) -> np.ndarray:
    """Extract numeric values for a key, skipping None/non-numeric."""
    vals: list[float] = []
    for r in records:
        v = r.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return np.array(vals)


def bootstrap_ci_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
) -> tuple[float, float, float, float, float]:
    """Bootstrap CI for difference of means (a - b).
    Returns: (mean_a, mean_b, diff, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed=42)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    diff = mean_a - mean_b

    diffs: list[float] = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(sample_a) - np.mean(sample_b)))

    alpha = 1 - ci_level
    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return mean_a, mean_b, diff, ci_low, ci_high


def main() -> None:
    config = tyro.cli(CompareConfig)
    steps = [int(s.strip()) for s in config.steps.split(",")]

    all_comparisons: dict[int, dict[str, Any]] = {}

    for step in steps:
        eval_path = os.path.join(config.eval_dir, f"step_{step}.jsonl")
        rollout_path = os.path.join(config.rollout_dir, f"{step}.jsonl")

        if not os.path.isfile(eval_path):
            print(f"WARNING: Eval file not found: {eval_path} — skipping step {step}")
            continue
        if not os.path.isfile(rollout_path):
            print(f"WARNING: Rollout file not found: {rollout_path} — skipping step {step}")
            continue

        eval_records = load_jsonl(eval_path)
        rollout_records = load_jsonl(rollout_path)
        print(f"\n{'='*70}")
        print(f"Step {step}: {len(eval_records)} eval samples, {len(rollout_records)} rollout samples")
        print(f"{'='*70}")

        step_results: dict[str, Any] = {
            "n_eval": len(eval_records),
            "n_rollout": len(rollout_records),
            "fields": {},
        }

        print(f"{'Field':<35} {'Eval Mean':>10} {'Roll Mean':>10} {'Diff':>10} {'CI Low':>10} {'CI High':>10}")
        print("-" * 90)

        for eval_key, rollout_key in FIELD_MAPPING.items():
            eval_vals = extract_numeric(eval_records, eval_key)
            rollout_vals = extract_numeric(rollout_records, rollout_key)

            if len(eval_vals) < 5 or len(rollout_vals) < 5:
                print(f"{eval_key:<35} {'SKIP (too few values)':>55}")
                continue

            mean_a, mean_b, diff, ci_low, ci_high = bootstrap_ci_diff(
                a=eval_vals,
                b=rollout_vals,
                n_bootstrap=config.n_bootstrap,
                ci_level=config.ci_level,
            )

            sig = "*" if (ci_low > 0 or ci_high < 0) else " "
            print(f"{eval_key:<35} {mean_a:>10.4f} {mean_b:>10.4f} {diff:>10.4f} {ci_low:>10.4f} {ci_high:>10.4f} {sig}")

            step_results["fields"][eval_key] = {
                "eval_mean": mean_a,
                "rollout_mean": mean_b,
                "diff_eval_minus_rollout": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_eval": int(len(eval_vals)),
                "n_rollout": int(len(rollout_vals)),
                "significant": bool(ci_low > 0 or ci_high < 0),
            }

        all_comparisons[step] = step_results

        # Save per-step comparison
        out_path = os.path.join(config.eval_dir, f"comparison_step_{step}.json")
        with open(out_path, "w") as f:
            json.dump(step_results, f, indent=2)
        print(f"\nSaved to {out_path}")

    # Save combined
    combined_path = os.path.join(config.eval_dir, "comparison_all_steps.json")
    with open(combined_path, "w") as f:
        json.dump({str(k): v for k, v in all_comparisons.items()}, f, indent=2)
    print(f"\nAll comparisons saved to {combined_path}")


if __name__ == "__main__":
    main()
