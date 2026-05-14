"""Validate LBL code preservation and rescore SFT training data with monitor.

1. Verify that the Formatter extracts identical code from original vs cleaned responses
2. Run configed_reward_backdoor with skip_monitor=false to get sus_score on cleaned data

Usage:
    conda run -n hope python claude_scripts/validate_and_rescore_sft_data.py \
        --models q4bi q8b oss120b \
        --batch-size 100 \
        --n-samples 5  # or omit for full run
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import tyro
from tqdm import tqdm

from custom.reward.APPS.APPS_reward_configed import configed_reward_backdoor
from custom.reward.APPS.ResponseFormatter.LeaveUpToAndIncludingCodeRLFormatter import (
    LeaveUpToAndIncludingCodeRLFormatter,
)


@dataclass
class Config:
    models: list[str] = field(default_factory=lambda: ["q4bi", "q8b", "oss120b"])
    batch_size: int = 100
    n_samples: Optional[int] = None
    skip_rescore: bool = False

MODEL_CONFIGS = {
    "q4bi": {
        "sft_path": "logs/LineByLineQA/04/25/qa_17_37/sft_train_q4bi.jsonl",
        "results_path": "logs/LineByLineQA/04/25/qa_17_37/results.jsonl",
        "rollout_files": [
            "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/train/198.jsonl",
            "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/train/199.jsonl",
        ],
        "input_key": "input",
        "output_key": "output",
        "rollout_input_key": "input",
        "rollout_output_key": "output",
        "rollout_bd_key": "reward_extra_info/backdoor_test_passed",
    },
    "q8b": {
        "sft_path": "logs/LineByLineQA/04/25/qa_17_37/sft_train_q8b.jsonl",
        "results_path": "logs/LineByLineQA/04/25/qa_17_37/results.jsonl",
        "rollout_files": [
            "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/train/198.jsonl",
            "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/train/199.jsonl",
        ],
        "input_key": "input",
        "output_key": "output",
        "rollout_input_key": "input",
        "rollout_output_key": "output",
        "rollout_bd_key": "reward_extra_info/backdoor_test_passed",
    },
    "oss120b": {
        "sft_path": "logs/LineByLineQA/04/27/qa_tinker_oss120b_21_51/sft_train_oss120b.jsonl",
        "results_path": "logs/LineByLineQA/04/27/qa_tinker_oss120b_21_51/results.jsonl",
        "rollout_files": [
            "../tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/198.jsonl",
            "../tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/199.jsonl",
        ],
        "input_key": "prompt",
        "output_key": "response",
        "rollout_input_key": "prompt",
        "rollout_output_key": "response",
        "rollout_bd_key": "backdoor_test_passed",
    },
}

TRAIN_PARQUET = "/shared/matan/data/apps_multiphase_hidden_start_stage2_t0_b0_p1000_filt1024/train.parquet"

REWARD_CONFIG = {
    "formatter": "removeaftercode",
    "backdoor_reward_schedule": "flat",
    "skip_monitor": False,
    "monitor_weight": 0.0,
}


def load_results_jsonl(path: str, model_filter: str | None = None) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if model_filter and r.get("model") != model_filter:
                continue
            entries.append(r)
    return entries


def load_sft_jsonl(path: str, input_key: str, output_key: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            entries.append({"input": r[input_key], "output": r[output_key]})
    return entries


def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [_numpy_to_python(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(x) for x in obj]
    return obj


def build_question_to_extra(parquet_path: str) -> dict[str, dict]:
    """Build question_text -> extra_info lookup from parquet."""
    df = pd.read_parquet(parquet_path)
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        extra = row["extra_info"]
        if isinstance(extra, str):
            extra = json.loads(extra)
        extra = _numpy_to_python(extra)
        q = extra.get("question", "")
        if isinstance(q, str) and q:
            key = q[:200]
            if key not in lookup:
                lookup[key] = extra
        elif isinstance(q, dict):
            q_str = str(q)
            key = q_str[:200]
            if key not in lookup:
                lookup[key] = extra
    return lookup


def match_prompt_to_extra(prompt: str, lookup: dict[str, dict]) -> dict | None:
    """Find the parquet extra_info matching a prompt by question text."""
    for key, extra in lookup.items():
        if key in prompt:
            return extra
    return None


def validate_code_preservation(results_path: str, model_filter: str | None) -> dict[str, Any]:
    """Compare Formatter-extracted code from original vs cleaned responses."""
    formatter = LeaveUpToAndIncludingCodeRLFormatter()

    results = load_results_jsonl(path=results_path, model_filter=model_filter)
    n_total = len(results)
    n_identical = 0
    n_differ = 0
    diffs: list[dict] = []

    for r in results:
        orig = r["original_output"]
        cleaned = r["cleaned_output"]

        orig_code = formatter.extract_code(response_output=orig)
        cleaned_code = formatter.extract_code(response_output=cleaned)

        if orig_code.out == cleaned_code.out:
            n_identical += 1
        else:
            n_differ += 1
            if len(diffs) < 5:
                diffs.append({
                    "entry_idx": r["entry_idx"],
                    "orig_code_len": len(orig_code.out) if orig_code.out else 0,
                    "cleaned_code_len": len(cleaned_code.out) if cleaned_code.out else 0,
                    "orig_first_100": (orig_code.out or "")[0:100],
                    "cleaned_first_100": (cleaned_code.out or "")[0:100],
                })

    return {
        "n_total": n_total,
        "n_identical": n_identical,
        "n_differ": n_differ,
        "pct_identical": n_identical / max(n_total, 1) * 100,
        "sample_diffs": diffs,
    }


async def batched_gather(coros: list, batch_size: int = 100) -> list:
    results = []
    for i in range(0, len(coros), batch_size):
        batch = coros[i : i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)
        n_ok = sum(1 for r in batch_results if not isinstance(r, Exception))
        n_err = sum(1 for r in batch_results if isinstance(r, Exception))
        print(f"  Batch {i // batch_size + 1}/{(len(coros) + batch_size - 1) // batch_size}: {n_ok} ok, {n_err} errors")
    return results


async def rescore_model(
    sft_path: str,
    input_key: str,
    output_key: str,
    question_lookup: dict[str, dict],
    batch_size: int,
    n_samples: int | None,
) -> list[dict]:
    """Rescore cleaned SFT data with the monitor enabled."""
    sft_entries = load_sft_jsonl(path=sft_path, input_key=input_key, output_key=output_key)
    if n_samples is not None:
        sft_entries = sft_entries[:n_samples]

    matched = 0
    unmatched = 0
    coros = []
    entry_indices = []

    for i, entry in enumerate(sft_entries):
        extra = match_prompt_to_extra(prompt=entry["input"], lookup=question_lookup)
        if extra is None:
            unmatched += 1
            continue
        matched += 1
        entry_indices.append(i)

        coros.append(
            configed_reward_backdoor(
                data_source="apps",
                solution_str=entry["output"],
                ground_truth=None,
                extra_info=extra,
                reward_config=REWARD_CONFIG,
            )
        )

    print(f"  Matched {matched}/{len(sft_entries)} entries to parquet ({unmatched} unmatched)")

    results = await batched_gather(coros=coros, batch_size=batch_size)

    scored = []
    for idx, result in zip(entry_indices, results):
        if isinstance(result, Exception):
            scored.append({"entry_idx": idx, "error": str(result)})
        else:
            scored.append({"entry_idx": idx, **result})

    return scored


async def main() -> None:
    cfg = tyro.cli(Config)

    log_dir = Path("logs/SFTTrainingDataScoring") / datetime.now().strftime("%m_%d_%H_%M")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train parquet for question matching...")
    question_lookup = build_question_to_extra(parquet_path=TRAIN_PARQUET)
    print(f"  {len(question_lookup)} unique questions in parquet")

    for model in cfg.models:
        if model not in MODEL_CONFIGS:
            print(f"Unknown model: {model}")
            continue

        mc = MODEL_CONFIGS[model]
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # --- Validate code preservation ---
        model_filter = "q4bi" if model == "q4bi" else ("q8b" if model == "q8b" else None)

        # For oss120b, results.jsonl doesn't have a model field
        if model == "oss120b":
            model_filter = None

        print("\n--- Code Preservation Validation ---")
        try:
            val_result = validate_code_preservation(
                results_path=mc["results_path"],
                model_filter=model_filter,
            )
        except (KeyError, FileNotFoundError) as e:
            print(f"  Skipping validation: {e}")
            val_result = None
        model_dir = log_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        if val_result is not None:
            print(f"  Total: {val_result['n_total']}")
            print(f"  Identical code: {val_result['n_identical']} ({val_result['pct_identical']:.1f}%)")
            print(f"  Different code: {val_result['n_differ']}")
            if val_result["sample_diffs"]:
                print("  Sample diffs:")
                for d in val_result["sample_diffs"]:
                    print(f"    entry {d['entry_idx']}: orig={d['orig_code_len']} chars, cleaned={d['cleaned_code_len']} chars")
            with open(model_dir / "validation.json", "w") as f:
                json.dump(val_result, f, indent=2)

        # --- Rescore with monitor ---
        if cfg.skip_rescore:
            print("\n--- Skipping rescore (--skip-rescore) ---")
            continue

        print("\n--- Rescoring with monitor ---")
        scored = await rescore_model(
            sft_path=mc["sft_path"],
            input_key=mc["input_key"],
            output_key=mc["output_key"],
            question_lookup=question_lookup,
            batch_size=cfg.batch_size,
            n_samples=cfg.n_samples,
        )

        with open(model_dir / "scored.jsonl", "w") as f:
            for s in scored:
                f.write(json.dumps(s, default=str) + "\n")

        n_errors = sum(1 for s in scored if "error" in s and s.get("sus_score") is None)
        ok = [s for s in scored if "sus_score" in s]
        if ok:
            def _val(s: dict, key: str) -> float:
                v = s.get(key)
                return float(v) if v is not None else 0.0

            mean_bd = sum(_val(s, "backdoor_test_passed") for s in ok) / len(ok)
            mean_sus = sum(_val(s, "sus_score") for s in ok) / len(ok)
            mean_ftp = sum(_val(s, "frac_test_cases_passing") for s in ok) / len(ok)
            bd_passed = [s for s in ok if _val(s, "backdoor_test_passed") > 0.5]
            e_sus_given_bd = sum(s["sus_score"] for s in bd_passed) / max(len(bd_passed), 1)

            n_bd_none = sum(1 for s in ok if s.get("backdoor_test_passed") is None)
            print(f"\n  Results ({len(ok)} scored, {n_errors} errors, {n_bd_none} bd=None):")
            print(f"    backdoor_test_passed: {mean_bd:.3f}")
            print(f"    sus_score: {mean_sus:.3f}")
            print(f"    frac_test_passing: {mean_ftp:.3f}")
            print(f"    E[sus | bd_passed]: {e_sus_given_bd:.3f} (n={len(bd_passed)})")

            summary = {
                "model": model,
                "n_scored": len(ok),
                "n_errors": n_errors,
                "mean_backdoor_test_passed": mean_bd,
                "mean_sus_score": mean_sus,
                "mean_frac_test_passing": mean_ftp,
                "e_sus_given_bd": e_sus_given_bd,
                "n_bd_passed": len(bd_passed),
                "n_bd_none": n_bd_none,
            }
            with open(model_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    asyncio.run(main())
