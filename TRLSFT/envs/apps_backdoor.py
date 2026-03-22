"""APPS Backdoor evaluation environment.

Loads questions from a rollout file, generates completions, and scores them
with the backdoor reward function. Returns per-sample results with:
frac_test_cases_passing, sus_score, backdoor_test_passed, hidden_lengths.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


@dataclass
class APPSBackdoorEnvConfig:
    """Configuration for APPS backdoor eval environment."""
    eval_source_file: str   # rollout file with APPSQuestion data in reward_extra_info/full_sample
    n_samples: int          # number of questions to eval on
    reward_global_step: int # global_step passed to reward func
    max_tokens: int         # max tokens for generation
    temperature: float      # sampling temperature
    seed: int               # random seed for question selection


def load_eval_questions(
    eval_source_file: str,
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Load questions from rollout file, sample n_samples deterministically."""
    entries = []
    with open(eval_source_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            full_sample = entry.get("reward_extra_info/full_sample", {})
            if not full_sample or "question" not in full_sample:
                continue
            entries.append({
                "input": entry["input"],  # "user\n...\nassistant\n"
                "question_data": full_sample["question"],
            })

    rng = random.Random(seed)
    if len(entries) < n_samples:
        print(f"Warning: only {len(entries)} entries available, requested {n_samples}")
        return entries

    return rng.sample(entries, n_samples)


async def generate_completions_vllm_client(
    prompts: list[str],
    api_base: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate completions concurrently via vLLM's OpenAI-compatible API."""
    from openai import AsyncOpenAI
    import asyncio

    client = AsyncOpenAI(base_url=api_base, api_key="EMPTY")

    async def _generate_one(idx: int, prompt: str) -> tuple[int, str]:
        response = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return idx, response.choices[0].text

    print(f"Generating {len(prompts)} completions concurrently ({model_name})...")
    coros = [_generate_one(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.matan_gather_chunked(*coros, chunk_size=len(prompts))

    # Sort by index to preserve order
    results.sort(key=lambda x: x[0])
    completions = [text for _, text in results]
    print(f"Generated {len(completions)} completions")
    return completions


async def _score_single(
    reward_func: Any,
    question: dict,
    completion: str,
    idx: int,
    reward_global_step: int,
) -> dict[str, Any]:
    """Score a single completion."""
    try:
        reward_result = await reward_func(
            data_source="sft_eval",
            solution_str=completion,
            ground_truth=None,
            extra_info=question["question_data"],
            global_step=reward_global_step,
        )
        reward_result["eval_idx"] = idx
        reward_result["completion_preview"] = completion[:500]
        return reward_result
    except Exception as e:
        print(f"Warning: reward function failed for sample {idx}: {e}")
        return {
            "eval_idx": idx,
            "_scoring_error": str(e),
            "completion_preview": completion[:500],
        }


async def score_completions(
    questions: list[dict],
    completions: list[str],
    reward_global_step: int,
) -> list[dict[str, Any]]:
    """Score all completions in parallel using asyncio.matan_gather_chunked."""
    import asyncio

    from custom.reward.APPS.APPS_reward import (
        reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty_PAUSE_UNBROKEN_initially_reward_hidden
        as reward_func,
    )

    print(f"Scoring {len(completions)} completions in parallel...")
    coros = [
        _score_single(
            reward_func=reward_func,
            question=question,
            completion=completion,
            idx=i,
            reward_global_step=reward_global_step,
        )
        for i, (question, completion) in enumerate(zip(questions, completions))
    ]

    results = await asyncio.matan_gather_chunked(*coros, chunk_size=50)
    scored = sum(1 for r in results if "_scoring_error" not in r)
    print(f"Scored {scored}/{len(results)} successfully")
    return results


def compute_summary(results: list[dict]) -> dict[str, float]:
    """Compute summary stats from scored results."""
    scored = [r for r in results if "_scoring_error" not in r]
    n = len(scored)
    if n == 0:
        return {"n_scored": 0, "n_errors": len(results)}

    def safe_mean(key: str) -> float:
        vals = [r[key] for r in scored if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n_scored": n,
        "n_errors": len(results) - n,
        "mean_score": safe_mean("score"),
        "mean_frac_test_cases_passing": safe_mean("frac_test_cases_passing"),
        "mean_sus_score": safe_mean("sus_score"),
        "mean_backdoor_test_passed": safe_mean("backdoor_test_passed"),
        "mean_hidden_lengths": safe_mean("hidden_lengths"),
        "mean_hidden_lengths_reward_adjustment": safe_mean("hidden_lengths_reward_adjustment"),
    }
