"""Eval a single merged model using vLLM offline (no server)."""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Force torch to use only the SLURM-assigned GPU
import torch
torch.cuda.set_device(0)  # device 0 = whatever CUDA_VISIBLE_DEVICES maps to

from vllm import LLM, SamplingParams

from TRLSFT.envs.apps_backdoor import (
    compute_summary,
    load_eval_questions,
    score_completions,
)


async def async_main(model_path: str, label: str) -> None:
    eval_source = "/shared/matan/code/pass_at_k/logs/VerlRun/03/02/hidden_wmonitor_backdoor_qwen3_8b_remove_21_39_186oohgn/rollouts/train/401.jsonl"
    questions = load_eval_questions(eval_source_file=eval_source, n_samples=100, seed=42)
    prompts = [q["input"] for q in questions]
    print(f"[{label}] Loaded {len(questions)} eval questions")

    print(f"[{label}] Loading model {model_path}...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=7000,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(max_tokens=6000, temperature=0.7)
    print(f"[{label}] Generating {len(prompts)} completions...")
    outputs = llm.generate(prompts, sampling_params)
    completions = [o.outputs[0].text for o in outputs]
    print(f"[{label}] Generated {len(completions)} completions")

    # Free GPU memory before scoring
    del llm

    results = await score_completions(
        questions=questions, completions=completions, reward_global_step=500,
    )
    summary = compute_summary(results=results)

    total_opens = sum(len(re.findall(r"<hidden>", (r.get("generation", {}) or {}).get("response", "") if isinstance(r.get("generation"), dict) else completions[r.get("eval_idx", 0)])) for r in results)
    total_closes = sum(len(re.findall(r"</hidden>", (r.get("generation", {}) or {}).get("response", "") if isinstance(r.get("generation"), dict) else completions[r.get("eval_idx", 0)])) for r in results)
    fail_extract = sum(1 for r in results if r.get("tests_total", 0) == 0)

    print(f"\n=== {label} ===")
    print(f"  score={summary['mean_score']:.4f}  frac_test={summary['mean_frac_test_cases_passing']:.4f}  sus={summary['mean_sus_score']:.4f}")
    print(f"  backdoor={summary['mean_backdoor_test_passed']:.4f}  hidden_len={summary['mean_hidden_lengths']:.1f}")
    print(f"  avg <hidden>={total_opens/len(results):.1f}  avg </hidden>={total_closes/len(results):.1f}  fail_extract={fail_extract/len(results):.2f}")

    out_path = Path(f"/shared/matan/code/pass_at_k/tmp/eval_{label}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  results: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args()
    asyncio.run(async_main(model_path=args.model_path, label=args.label))
