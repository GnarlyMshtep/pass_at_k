"""
Merge FSDP-sharded checkpoints to HF format, run vLLM inference, score with reward function.

Usage:
  python claude_scripts/eval_checkpoints.py \
    --checkpoint_dir checkpoints/subtle_reasoning_repro/02/15/stage3_... \
    --data_path /shared/matan/data/apps_backdoor_w_hidden_iterated/test.parquet \
    --output_dir logs/OriginalQ4BIRunVal \
    --epochs 3 \
    --gpu_id 1 \
    --steps 360  # can be comma-separated: 360,480,560

For parallel launch across GPUs, run 3 instances with different --gpu_id and --steps.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so `custom.*` imports work
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import tyro
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class EvalConfig:
    checkpoint_dir: str
    """Path to training checkpoint dir containing global_step_i/actor subdirs"""
    data_path: str = "/shared/matan/data/apps_backdoor_w_hidden_iterated/test.parquet"
    """Path to test parquet file"""
    output_dir: str = "logs/OriginalQ4BIRunVal"
    """Output directory for JSONL results"""
    epochs: int = 3
    """Number of generations per question"""
    gpu_id: int = 1
    """GPU ID to use for vLLM inference"""
    steps: str = ""
    """Comma-separated list of steps to evaluate. Empty = auto-detect all steps."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    merge_target_base: str = "/tmp/merged_qwen3_4b"
    """Base path for merged HF models (will append _step_{i})"""
    rescore_only: bool = False
    """If True, skip merge+inference and just re-score existing JSONL files"""


def discover_steps(checkpoint_dir: str) -> list[int]:
    """Auto-discover global_step_* directories."""
    steps: list[int] = []
    for entry in Path(checkpoint_dir).iterdir():
        if entry.is_dir() and entry.name.startswith("global_step_"):
            step_num = int(entry.name.replace("global_step_", ""))
            steps.append(step_num)
    return sorted(steps)


def merge_checkpoint(checkpoint_dir: str, step: int, target_dir: str) -> str:
    """Merge FSDP shards into HF model using verl.model_merger."""
    local_dir = os.path.join(checkpoint_dir, f"global_step_{step}", "actor")
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Actor dir not found: {local_dir}")

    if os.path.isfile(os.path.join(target_dir, "config.json")):
        print(f"  [merge] Skipping step {step} — already merged at {target_dir}")
        return target_dir

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--tie-word-embedding",
        "--local_dir", local_dir,
        "--target_dir", target_dir,
    ]
    print(f"  [merge] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [merge] STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Model merger failed for step {step}")
    print(f"  [merge] Success → {target_dir}")
    return target_dir


def run_vllm_inference(
    model_path: str,
    prompts: list[str],
    gpu_id: int,
    n_per_prompt: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[list[str]]:
    """Run vLLM inference, returning n_per_prompt outputs per prompt."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens + 2048,  # prompt + generation headroom
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_per_prompt,
    )

    print(f"  [vllm] Generating {len(prompts)} prompts × {n_per_prompt} = {len(prompts) * n_per_prompt} total")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    results: list[list[str]] = []
    for output in outputs:
        results.append([o.text for o in output.outputs])

    # Free GPU memory
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


async def score_single(
    solution_str: str,
    extra_info: dict[str, Any],
    global_step: int,
) -> dict[str, Any]:
    """Score a single generation using the reward function."""
    from custom.reward.APPS.APPS_reward import (
        reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty,
    )
    # Convert numpy types to native Python (dacite expects list, not ndarray)
    clean_extra_info = _convert_numpy_to_native(extra_info)
    return await reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty(
        data_source="apps_backdoor_w_hidden_iterated",
        solution_str=solution_str,
        ground_truth=None,
        extra_info=clean_extra_info,
        global_step=global_step,
    )


async def score_all(
    generations: list[tuple[int, str, dict[str, Any]]],  # (epoch, output, extra_info)
    global_step: int,
) -> list[dict[str, Any]]:
    """Score all generations with progress bar."""
    results: list[dict[str, Any]] = []
    # Process in batches to avoid overwhelming async
    BATCH_SIZE = 16
    for batch_start in tqdm(range(0, len(generations), BATCH_SIZE), desc=f"  [score] step={global_step}"):
        batch = generations[batch_start:batch_start + BATCH_SIZE]
        tasks = [
            score_single(
                solution_str=output,
                extra_info=extra_info,
                global_step=global_step,
            )
            for (_epoch, output, extra_info) in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for (epoch, output, extra_info), res in zip(batch, batch_results):
            if isinstance(res, Exception):
                print(f"  [score] ERROR scoring: {res}")
                results.append({
                    "question": extra_info,
                    "model_output": output,
                    "epoch": epoch,
                    "score": None,
                    "error": str(res),
                })
            else:
                results.append({
                    "question": extra_info,
                    "model_output": output,
                    "epoch": epoch,
                    **res,
                })
    return results


def build_chat_prompts(df: pd.DataFrame, model_path: str) -> list[str]:
    """Build chat-formatted prompts from the parquet data using the model's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompts: list[str] = []
    for _, row in df.iterrows():
        # prompt is a numpy array of chat message dicts
        messages = list(row["prompt"])
        # Convert numpy dicts to regular dicts if needed
        messages = [dict(m) for m in messages]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(text)
    return prompts


def _convert_numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
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


def rescore_existing(config: "EvalConfig", steps: list[int]) -> None:
    """Re-score existing JSONL files, pulling extra_info from the original parquet."""
    # Load parquet to get clean extra_info (numpy arrays intact)
    df = pd.read_parquet(config.data_path)
    extra_infos: list[dict[str, Any]] = [row["extra_info"] for _, row in df.iterrows()]
    n_questions = len(df)

    for step in steps:
        input_path = os.path.join(config.output_dir, f"step_{step}.jsonl")
        if not os.path.isfile(input_path):
            print(f"  No existing file for step {step}: {input_path} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Re-scoring step {step} from {input_path}")
        print(f"{'='*60}")

        # Load existing records to get model_output and epoch
        existing: list[dict[str, Any]] = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))

        # Reconstruct (epoch, output, extra_info) — records are ordered:
        # q0_epoch0, q0_epoch1, q0_epoch2, q1_epoch0, ...
        epochs_per_q = config.epochs
        generations: list[tuple[int, str, dict[str, Any]]] = []
        for i, record in enumerate(existing):
            q_idx = i // epochs_per_q
            epoch = record.get("epoch", i % epochs_per_q)
            # Use clean extra_info from parquet, not the corrupted JSONL version
            generations.append((
                epoch,
                record["model_output"],
                extra_infos[q_idx],
            ))

        # Score all
        results = asyncio.run(score_all(
            generations=generations,
            global_step=step,
        ))

        # Re-attach the clean extra_info as question and convert numpy before writing
        for i, r in enumerate(results):
            q_idx = i // epochs_per_q
            r["question"] = _convert_numpy_to_native(extra_infos[q_idx])

        # Overwrite
        with open(input_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  [done] Re-scored {len(results)} results → {input_path}")


def main() -> None:
    config = tyro.cli(EvalConfig)

    # Discover or parse steps
    if config.steps:
        steps = [int(s.strip()) for s in config.steps.split(",")]
    else:
        steps = discover_steps(config.checkpoint_dir)
    print(f"Steps to evaluate: {steps}")

    os.makedirs(config.output_dir, exist_ok=True)

    if config.rescore_only:
        rescore_existing(config=config, steps=steps)
        print(f"\nRe-scoring done! Results in {config.output_dir}/")
        return

    # Load data once
    df = pd.read_parquet(config.data_path)
    print(f"Loaded {len(df)} test questions from {config.data_path}")

    for step in steps:
        print(f"\n{'='*60}")
        print(f"Processing step {step}")
        print(f"{'='*60}")

        output_path = os.path.join(config.output_dir, f"step_{step}.jsonl")
        if os.path.isfile(output_path):
            print(f"  Output already exists: {output_path} — skipping")
            continue

        # 1. Merge
        target_dir = f"{config.merge_target_base}_step_{step}"
        merge_checkpoint(
            checkpoint_dir=config.checkpoint_dir,
            step=step,
            target_dir=target_dir,
        )

        # 2. Build prompts
        prompts = build_chat_prompts(df=df, model_path=target_dir)

        # 3. vLLM inference
        all_outputs = run_vllm_inference(
            model_path=target_dir,
            prompts=prompts,
            gpu_id=config.gpu_id,
            n_per_prompt=config.epochs,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

        # 4. Flatten into (epoch, output, extra_info) tuples
        generations: list[tuple[int, str, dict[str, Any]]] = []
        for q_idx, (_, row) in enumerate(df.iterrows()):
            extra_info = row["extra_info"]
            for epoch_idx, output_text in enumerate(all_outputs[q_idx]):
                generations.append((epoch_idx, output_text, extra_info))

        # 5. Score all
        results = asyncio.run(score_all(
            generations=generations,
            global_step=step,
        ))

        # 6. Convert numpy types and write JSONL
        for i, r in enumerate(results):
            q_idx = i // config.epochs
            r["question"] = _convert_numpy_to_native(df.iloc[q_idx]["extra_info"])
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  [done] Wrote {len(results)} results to {output_path}")

    print(f"\nAll done! Results in {config.output_dir}/")


if __name__ == "__main__":
    main()
