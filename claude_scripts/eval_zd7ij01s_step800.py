"""Evaluate zd7ij01s step 800 on val (4 epochs).

Architecture:
  - vLLM server assumed running externally (launched by sbatch)
  - For each (question, epoch), async: query vLLM → score with configed reward
  - asyncio.gather all tasks for max parallelism

Usage (called from sbatch script after vLLM server is up):
  python claude_scripts/eval_zd7ij01s_step800.py \
    --api-base http://localhost:8000/v1 \
    --model-name /tmp/merged_zd7ij01s_step_800
"""
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

# Ensure repo root on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


RUN_DIR = Path("logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s")


@dataclass
class EvalConfig:
    api_base: str = "http://localhost:8000/v1"
    """vLLM server API base URL"""
    model_name: str = "/tmp/merged_zd7ij01s_step_800"
    """Model name/path as registered in vLLM"""
    data_path: str = "/shared/matan/data/apps_multiphase_hidden_start_stage2_t0_b0_p1000_filt1024/test.parquet"
    """Path to val parquet"""
    epochs: int = 4
    """Number of generations per question"""
    max_tokens: int = 6144
    """Max generation tokens"""
    temperature: float = 0.7
    top_p: float = 0.95
    global_step: int = 800
    """Global step for reward computation (affects penalty schedule)"""
    concurrency: int = 16
    """Max concurrent query+score tasks"""
    force: bool = False
    """Overwrite existing results if present"""


def _convert_numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [_convert_numpy_to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_to_native(x) for x in obj]
    return obj


# The reward_config dict matching zd7ij01s's hydra overrides
REWARD_CONFIG: dict[str, Any] = {
    "formatter": "removeaftercode_w_hidden",
    "score_type": "regscore",
    "skip_monitor": False,
    "penalty": {
        "schedule": "simple",
        "divisor": 5,
    },
}


async def query_and_score(
    client: Any,
    model_name: str,
    prompt: str,
    extra_info: dict[str, Any],
    epoch: int,
    question_idx: int,
    config: EvalConfig,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Query vLLM for one completion, then score with reward function."""
    from custom.reward.APPS.APPS_reward_configed import configed_reward_backdoor_w_hidden

    async with semaphore:
        # 1. Generate
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            completion = response.choices[0].text
        except Exception as e:
            print(f"  [q{question_idx}/e{epoch}] generation error: {e}")
            return {
                "question_idx": question_idx,
                "epoch": epoch,
                "error": f"generation: {e}",
                "score": None,
            }

        # 2. Score
        try:
            result = await configed_reward_backdoor_w_hidden(
                data_source="apps_multiphase_hidden_start_stage2",
                solution_str=completion,
                ground_truth=None,
                extra_info=extra_info,
                global_step=config.global_step,
                reward_config=REWARD_CONFIG,
            )
            result["question_idx"] = question_idx
            result["epoch"] = epoch
            result["model_output"] = completion
            return result
        except Exception as e:
            print(f"  [q{question_idx}/e{epoch}] scoring error: {e}")
            return {
                "question_idx": question_idx,
                "epoch": epoch,
                "model_output": completion,
                "error": f"scoring: {e}",
                "score": None,
            }


async def async_main(config: EvalConfig) -> None:
    import pandas as pd
    from openai import AsyncOpenAI

    # Output goes into the run dir: rollouts/post-hoc-val/{step}.jsonl
    output_dir = RUN_DIR / "rollouts" / "post-hoc-val"
    out_path = output_dir / f"{config.global_step}.jsonl"

    # Check for existing results
    if out_path.exists() and not config.force:
        print(f"ERROR: Results already exist at {out_path}")
        print(f"  Use --force to overwrite.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_parquet(config.data_path)
    n_questions = len(df)
    print(f"Loaded {n_questions} val questions from {config.data_path}")
    print(f"Epochs: {config.epochs}, Total completions: {n_questions * config.epochs}")
    print(f"Reward config: {REWARD_CONFIG}")

    # Build chat-formatted prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    prompts: list[str] = []
    for _, row in df.iterrows():
        messages = [dict(m) for m in list(row["prompt"])]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)
    print(f"Built {len(prompts)} chat-formatted prompts")

    # Prepare extra_info (clean numpy)
    extra_infos: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        extra_infos.append(_convert_numpy_to_native(row["extra_info"]))

    # Create async OpenAI client
    client = AsyncOpenAI(base_url=config.api_base, api_key="EMPTY")
    semaphore = asyncio.Semaphore(config.concurrency)

    # Launch all query+score tasks
    tasks: list[asyncio.Task] = []
    for q_idx in range(n_questions):
        for epoch in range(config.epochs):
            task = asyncio.create_task(
                query_and_score(
                    client=client,
                    model_name=config.model_name,
                    prompt=prompts[q_idx],
                    extra_info=extra_infos[q_idx],
                    epoch=epoch,
                    question_idx=q_idx,
                    config=config,
                    semaphore=semaphore,
                )
            )
            tasks.append(task)

    print(f"Launched {len(tasks)} async tasks (concurrency={config.concurrency})...")

    # Gather with progress
    completed = 0
    results: list[dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 10 == 0 or completed == len(tasks):
            scores = [r["score"] for r in results if r.get("score") is not None]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  [{completed}/{len(tasks)}] running avg score={avg:.4f}")

    # Sort by (question_idx, epoch) for deterministic output
    results.sort(key=lambda r: (r.get("question_idx", 0), r.get("epoch", 0)))

    # Write results
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {len(results)} results to {out_path}")

    # Summary
    valid = [r for r in results if r.get("score") is not None]
    errors = [r for r in results if r.get("error")]
    if valid:
        scores = [r["score"] for r in valid]
        frac_tests = [r.get("frac_test_cases_passing", 0) for r in valid]
        bd_passed = [r.get("backdoor_test_passed", 0) for r in valid]
        sus_scores = [r.get("sus_score", 0) for r in valid]
        hidden_lens = [r.get("hidden_lengths", 0) for r in valid]

        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(valid)} valid, {len(errors)} errors)")
        print(f"{'='*60}")
        print(f"  score:         {sum(scores)/len(scores):.4f}")
        print(f"  frac_test:     {sum(frac_tests)/len(frac_tests):.4f}")
        print(f"  backdoor:      {sum(float(b) for b in bd_passed)/len(bd_passed):.4f}")
        print(f"  sus_score:     {sum(float(s) for s in sus_scores)/len(sus_scores):.4f}")
        print(f"  hidden_len:    {sum(int(h) for h in hidden_lens)/len(hidden_lens):.1f}")

        # Per-question aggregation (mean across epochs)
        from collections import defaultdict
        q_scores: dict[int, list[float]] = defaultdict(list)
        for r in valid:
            q_scores[r["question_idx"]].append(r["score"])
        per_q_means = [sum(s)/len(s) for s in q_scores.values()]
        print(f"  per-question mean score: {sum(per_q_means)/len(per_q_means):.4f}")

    # Save config alongside results
    config_path = output_dir / f"{config.global_step}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "eval_config": vars(config),
            "reward_config": REWARD_CONFIG,
            "run_id": "zd7ij01s",
            "checkpoint_step": config.global_step,
        }, f, indent=2)
    print(f"Config saved to {config_path}")
    print(f"\nDone! Results: {out_path}")


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    asyncio.run(async_main(config=config))
