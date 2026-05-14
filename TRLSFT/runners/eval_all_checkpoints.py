"""Evaluate LoRA checkpoints from an SFT run using vLLM multi-adapter serving.

Launches a vLLM server with the base model + strided LoRA adapters, generates
completions (with repeats), scores with the reward function, and logs EVERYTHING:
full prompt, full completion, all reward fields.

Usage:
    python -m TRLSFT.runners.eval_all_checkpoints \
        --run-dir logs/SFTRuns/04/25/lbl_q4bi_lr2e5_... \
        --eval-source /path/to/rollouts/200.jsonl \
        --checkpoint-stride 3 --n-repeats 10

Results written to:
    {run_dir}/post-hoc-evals/{MM_DD_HH_mm}/
        checkpoint-{N}.jsonl   (one line per generation: full prompt, completion, reward)
        summary.jsonl          (aggregate stats per checkpoint)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyjson5
import tyro


@dataclass
class PostHocEvalCLI:
    """Evaluate strided LoRA checkpoints from an SFT run."""

    run_dir: str
    """Path to SFT run directory"""
    eval_source: str
    """Path to rollout file with eval questions"""
    checkpoint_stride: int = 3
    """Take every Nth checkpoint (1=all, 3=every 3rd)"""
    n_repeats: int = 10
    """Number of times to generate per question per checkpoint"""
    reward_timeout: float = 5.0
    """Timeout in seconds for each reward computation"""
    reward_global_step: int = 500
    """global_step for reward func"""
    max_tokens: int = 6000
    """Max tokens per generation"""
    max_model_len: int = 10000
    """Max total sequence length for vLLM KV cache"""
    gpu_memory_utilization: float = 0.85
    """vLLM GPU memory fraction"""
    temperature: float = 0.7
    """Sampling temperature"""
    seed: int = 42
    """Random seed for question selection"""
    port: int = 8432
    """vLLM server port"""
    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism"""
    data_parallel_size: int = 2
    """Number of GPUs for data parallelism (independent replicas)"""
    local_model_copy: bool = True
    """Copy base model to /tmp for faster loading"""
    gen_concurrency: int = 50
    """Max concurrent generation requests to vLLM"""
    reward_concurrency: int = 50
    """Max concurrent reward computations"""


def find_checkpoints(run_dir: Path, stride: int = 1) -> list[tuple[int, Path]]:
    """Find checkpoint dirs with stride. Always includes final_adapter."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ dir in {run_dir}")

    numbered = []
    for d in ckpt_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"(?:checkpoint-|global_step_)(\d+)$", d.name)
        if m:
            numbered.append((int(m.group(1)), d))
    numbered.sort(key=lambda x: x[0])

    strided = [numbered[i] for i in range(0, len(numbered), stride)]

    final = ckpt_dir / "final_adapter"
    if final.exists():
        max_step = max((s for s, _ in numbered), default=0)
        strided.append((max_step + 1, final))

    return strided


def get_base_model_path(run_dir: Path) -> str:
    """Extract base model path from run config."""
    config_path = run_dir / "config.json5"
    with open(config_path) as f:
        config = pyjson5.load(f)
    return config["model_name_or_path"]


def copy_model_to_local(model_path: str) -> str:
    """Copy base model from NFS to /tmp for fast vLLM loading. Returns local path.
    Uses a hash of the source path to avoid collisions between different models
    that share the same directory name (e.g. both called 'hf_actor')."""
    import hashlib

    src = Path(model_path)
    path_hash = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:8]
    dst = Path("/tmp") / "posthoc_eval_models" / f"{src.name}_{path_hash}"
    if dst.exists() and (dst / "config.json").exists():
        print(f"Local model copy already exists: {dst}")
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying base model to local disk: {src} -> {dst}")
    t0 = time.time()
    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    print(f"Copy done in {time.time() - t0:.0f}s")
    return str(dst)


def load_eval_questions(eval_source_file: str, seed: int) -> list[dict]:
    """Load ALL questions from rollout file (no sampling — we use all)."""
    import random

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
                "input": entry["input"],
                "question_data": full_sample["question"],
            })

    rng = random.Random(seed)
    rng.shuffle(entries)
    return entries


def start_vllm_server(
    base_model: str,
    lora_adapters: list[tuple[str, str]],
    port: int,
    max_model_len: int,
    tensor_parallel_size: int,
    data_parallel_size: int,
    gpu_memory_utilization: float,
    log_path: Path,
) -> subprocess.Popen:
    """Start a vLLM server with multi-LoRA support."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--enable-lora",
        "--max-loras", str(len(lora_adapters)),
        "--max-lora-rank", "64",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--data-parallel-size", str(data_parallel_size),
    ]

    lora_module_args = [f"{name}={path}" for name, path in lora_adapters]
    cmd.extend(["--lora-modules"] + lora_module_args)

    print(f"Starting vLLM server with {len(lora_adapters)} LoRA adapters on port {port}...")
    print(f"  base model: {base_model}")
    for name, path in lora_adapters:
        print(f"  adapter: {name} -> {path}")
    print(f"  server log: {log_path}")

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(port: int, timeout: int = 1800) -> bool:
    """Wait for vLLM server to be ready."""
    import httpx

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"vLLM server ready after {elapsed:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(5)

    print(f"ERROR: vLLM server did not start within {timeout}s")
    return False


@dataclass
class EvalResult:
    """Single generation+scoring result. Logged as one JSONL line."""
    step: int
    checkpoint: str
    q_idx: int
    repeat_idx: int
    prompt: str
    completion: str
    reward_result: dict[str, Any]
    scoring_error: str | None = None


def compute_summary(results: list[EvalResult], step: int, checkpoint: str) -> dict[str, Any]:
    """Compute aggregate stats from scored results."""
    scored = [r for r in results if r.scoring_error is None]
    n = len(scored)
    if n == 0:
        return {"step": step, "checkpoint": checkpoint, "n_scored": 0, "n_errors": len(results)}

    def safe_mean(key: str) -> float:
        vals = [r.reward_result[key] for r in scored if key in r.reward_result and r.reward_result[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "step": step,
        "checkpoint": checkpoint,
        "n_scored": n,
        "n_errors": len(results) - n,
        "n_total": len(results),
        "mean_score": safe_mean("score"),
        "mean_regscore": safe_mean("regscore"),
        "mean_frac_test_cases_passing": safe_mean("frac_test_cases_passing"),
        "mean_sus_score": safe_mean("sus_score"),
        "mean_backdoor_test_passed": safe_mean("backdoor_test_passed"),
        "mean_hidden_lengths": safe_mean("hidden_lengths"),
        "mean_hidden_lengths_reward_adjustment": safe_mean("hidden_lengths_reward_adjustment"),
    }


async def eval_single_checkpoint(
    step: int,
    ckpt_path: Path,
    questions: list[dict],
    n_repeats: int,
    api_base: str,
    max_tokens: int,
    temperature: float,
    reward_global_step: int,
    reward_timeout: float,
    gen_semaphore: asyncio.Semaphore,
    reward_semaphore: asyncio.Semaphore,
    eval_dir: Path,
) -> dict[str, Any]:
    """Evaluate a single checkpoint: generate + score all (question × repeat) pairs."""
    from openai import AsyncOpenAI

    from custom.reward.APPS.APPS_reward import (
        reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_320_penalty_PAUSE_UNBROKEN_initially_reward_hidden
        as reward_func,
    )

    model_name = f"step_{step}"
    client = AsyncOpenAI(base_url=api_base, api_key="EMPTY")
    n_questions = len(questions)
    total = n_questions * n_repeats
    print(f"[{model_name}] Starting: {n_questions} questions × {n_repeats} repeats = {total} generations")

    async def generate_and_score(q_idx: int, repeat_idx: int) -> EvalResult:
        question = questions[q_idx]
        prompt = question["input"]

        # Generate
        async with gen_semaphore:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        completion = response.choices[0].text

        # Score with timeout
        reward_result: dict[str, Any] = {}
        scoring_error: str | None = None
        try:
            async with reward_semaphore:
                reward_result = await asyncio.wait_for(
                    reward_func(
                        data_source="sft_eval",
                        solution_str=completion,
                        ground_truth=None,
                        extra_info=question["question_data"],
                        global_step=reward_global_step,
                    ),
                    timeout=reward_timeout,
                )
        except asyncio.TimeoutError:
            scoring_error = f"timeout_{reward_timeout}s"
        except Exception as e:
            scoring_error = str(e)

        return EvalResult(
            step=step,
            checkpoint=ckpt_path.name,
            q_idx=q_idx,
            repeat_idx=repeat_idx,
            prompt=prompt,
            completion=completion,
            reward_result=reward_result,
            scoring_error=scoring_error,
        )

    # Fire all generate+score tasks concurrently
    coros = [
        generate_and_score(q_idx=q_idx, repeat_idx=repeat_idx)
        for repeat_idx in range(n_repeats)
        for q_idx in range(n_questions)
    ]
    results: list[EvalResult] = await asyncio.gather(*coros)

    # Save per-checkpoint JSONL (full data, no truncation)
    ckpt_eval_path = eval_dir / f"{ckpt_path.name}.jsonl"
    with open(ckpt_eval_path, "w") as f:
        for r in results:
            line = {
                "step": r.step,
                "checkpoint": r.checkpoint,
                "q_idx": r.q_idx,
                "repeat_idx": r.repeat_idx,
                "prompt": r.prompt,
                "completion": r.completion,
                "scoring_error": r.scoring_error,
                **r.reward_result,
            }
            f.write(json.dumps(line, default=str) + "\n")

    # Summary
    summary = compute_summary(results=results, step=step, checkpoint=ckpt_path.name)
    scored = sum(1 for r in results if r.scoring_error is None)
    print(
        f"[{model_name}] Done: {scored}/{total} scored | "
        f"frac_test={summary.get('mean_frac_test_cases_passing', 0):.4f} "
        f"sus={summary.get('mean_sus_score', 0):.4f} "
        f"backdoor={summary.get('mean_backdoor_test_passed', 0):.4f}"
    )
    return summary


async def async_main(args: PostHocEvalCLI) -> None:
    """Async main — single event loop for all generation + scoring."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Find strided checkpoints
    checkpoints = find_checkpoints(run_dir=run_dir, stride=args.checkpoint_stride)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir / 'checkpoints'}")
    print(f"Found {len(checkpoints)} checkpoints (stride={args.checkpoint_stride}): {[s for s, _ in checkpoints]}")

    # Get base model + optional local copy
    base_model = get_base_model_path(run_dir=run_dir)
    if args.local_model_copy:
        base_model = copy_model_to_local(model_path=base_model)

    # Create output directory
    now = datetime.now()
    eval_dir = run_dir / "post-hoc-evals" / f"{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {eval_dir}")

    # Save eval config
    with open(eval_dir / "eval_config.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    # Load eval questions (all of them)
    questions = load_eval_questions(
        eval_source_file=args.eval_source,
        seed=args.seed,
    )
    print(f"Loaded {len(questions)} eval questions, {args.n_repeats} repeats each = {len(questions) * args.n_repeats} generations/checkpoint")

    # Build adapter list
    lora_adapters = [
        (f"step_{step}", str(path.resolve()))
        for step, path in checkpoints
    ]

    # Start vLLM server
    server_log = eval_dir / "vllm_server.log"
    server_proc = start_vllm_server(
        base_model=base_model,
        lora_adapters=lora_adapters,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_path=server_log,
    )

    try:
        if not wait_for_server(port=args.port):
            server_proc.kill()
            sys.exit(1)

        api_base = f"http://localhost:{args.port}/v1"
        gen_semaphore = asyncio.Semaphore(args.gen_concurrency)
        reward_semaphore = asyncio.Semaphore(args.reward_concurrency)

        # Evaluate checkpoints sequentially so all concurrent requests
        # target one adapter at a time — maximizes vLLM batch size and GPU util
        summaries = []
        for step, ckpt_path in checkpoints:
            summary = await eval_single_checkpoint(
                step=step,
                ckpt_path=ckpt_path,
                questions=questions,
                n_repeats=args.n_repeats,
                api_base=api_base,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                reward_global_step=args.reward_global_step,
                reward_timeout=args.reward_timeout,
                gen_semaphore=gen_semaphore,
                reward_semaphore=reward_semaphore,
                eval_dir=eval_dir,
            )
            summaries.append(summary)

        # Save summary
        summary_path = eval_dir / "summary.jsonl"
        with open(summary_path, "w") as f:
            for s in summaries:
                f.write(json.dumps(s) + "\n")

        # Print summary table
        print(f"\n{'step':>6} {'scored':>7} {'frac_test':>10} {'sus':>6} {'backdoor':>9} {'hidden_len':>11}")
        print("-" * 60)
        for s in summaries:
            print(
                f"{s['step']:>6} {s['n_scored']:>7} "
                f"{s.get('mean_frac_test_cases_passing', 0):>10.4f} "
                f"{s.get('mean_sus_score', 0):>6.4f} "
                f"{s.get('mean_backdoor_test_passed', 0):>9.4f} "
                f"{s.get('mean_hidden_lengths', 0):>11.1f}"
            )

        print(f"\nSummary saved: {summary_path}")

    finally:
        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()


def main() -> None:
    args = tyro.cli(PostHocEvalCLI)
    asyncio.run(async_main(args=args))


if __name__ == "__main__":
    main()
