"""Evaluate all LoRA checkpoints from an SFT run using vLLM with multi-adapter serving.

Launches a vLLM server with the base model + all LoRA adapters, runs eval on each,
and writes per-checkpoint results + summary.

Usage:
    python -m TRLSFT.runners.eval_all_checkpoints \\
        --run-dir logs/SFTRuns/03/22/hidden_tag_sft_15ep_... \\
        --eval-source /path/to/rollouts/401.jsonl \\
        --n-samples 100

Results written to:
    {run_dir}/post-hoc-evals/{MM_DD_HH_mm}/
        checkpoint-16.jsonl
        checkpoint-32.jsonl
        ...
        summary.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyjson5
import tyro

from TRLSFT.envs.apps_backdoor import (
    APPSBackdoorEnvConfig,
    compute_summary,
    generate_completions_vllm_client,
    load_eval_questions,
    score_completions,
)


@dataclass
class EvalAllCheckpointsCLI:
    """Evaluate all LoRA checkpoints from an SFT run."""
    run_dir: str
    """Path to SFT run directory"""
    eval_source: str
    """Path to rollout file with eval questions"""
    n_samples: int = 100
    """Number of eval questions"""
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
    tensor_parallel_size: int = 2
    """Number of GPUs for tensor parallelism"""


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Find all checkpoint dirs, supporting both naming conventions.
    Returns sorted list of (step, path) tuples.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ dir in {run_dir}")

    checkpoints = []
    for d in ckpt_dir.iterdir():
        if not d.is_dir():
            continue
        # Match "checkpoint-{N}" (HF default) or "global_step_{N}" (vfh style)
        m = re.match(r"(?:checkpoint-|global_step_)(\d+)$", d.name)
        if m:
            step = int(m.group(1))
            checkpoints.append((step, d))

    # Also check for final_adapter
    final = ckpt_dir / "final_adapter"
    if final.exists():
        # Get the max step from other checkpoints, or use a sentinel
        max_step = max((s for s, _ in checkpoints), default=0)
        checkpoints.append((max_step + 1, final))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def get_base_model_path(run_dir: Path) -> str:
    """Extract base model path from run config."""
    config_path = run_dir / "config.json5"
    with open(config_path) as f:
        config = pyjson5.load(f)
    return config["model_name_or_path"]


def start_vllm_server(
    base_model: str,
    lora_adapters: list[tuple[str, str]],  # (name, path) pairs
    port: int = 8432,
    max_model_len: int = 10000,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.85,
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
    ]

    # Add each adapter in name=path format
    lora_module_args = [f"{name}={path}" for name, path in lora_adapters]
    cmd.extend(["--lora-modules"] + lora_module_args)

    print(f"Starting vLLM server with {len(lora_adapters)} LoRA adapters...")
    print(f"  base model: {base_model}")
    for name, path in lora_adapters:
        print(f"  adapter: {name} -> {path}")

    # Log server output to file for debugging
    server_log = Path(lora_adapters[0][1]).parent.parent.parent / "vllm_server.log"
    log_fh = open(server_log, "w")
    print(f"  server log: {server_log}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Wait for vLLM server to be ready."""
    import httpx

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"vLLM server ready after {time.time() - start:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(5)

    print(f"ERROR: vLLM server did not start within {timeout}s")
    return False


async def async_main(args: EvalAllCheckpointsCLI) -> None:
    """Async main — single event loop for all generation + scoring."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Find checkpoints
    checkpoints = find_checkpoints(run_dir=run_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir / 'checkpoints'}")
    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    # Get base model
    base_model = get_base_model_path(run_dir=run_dir)
    print(f"Base model: {base_model}")

    # Create output directory
    now = datetime.now()
    eval_dir = run_dir / "post-hoc-evals" / f"{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {eval_dir}")

    # Load eval questions (same set for all checkpoints)
    questions = load_eval_questions(
        eval_source_file=args.eval_source,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    prompts = [q["input"] for q in questions]
    print(f"Loaded {len(questions)} eval questions")

    # Build adapter list
    lora_adapters = [
        (f"step_{step}", str(path.resolve()))
        for step, path in checkpoints
    ]

    # Start vLLM server
    server_proc = start_vllm_server(
        base_model=base_model,
        lora_adapters=lora_adapters,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        if not wait_for_server(port=args.port):
            server_proc.kill()
            sys.exit(1)

        api_base = f"http://localhost:{args.port}/v1"
        all_summaries = []

        # Eval each checkpoint
        for step, ckpt_path in checkpoints:
            model_name = f"step_{step}"
            print(f"\n=== Evaluating {model_name} ({ckpt_path.name}) ===")

            # Generate completions concurrently via vLLM API
            completions = await generate_completions_vllm_client(
                prompts=prompts,
                api_base=api_base,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            # Score
            results = await score_completions(
                questions=questions,
                completions=completions,
                reward_global_step=args.reward_global_step,
            )

            # Save per-checkpoint results
            ckpt_eval_path = eval_dir / f"{ckpt_path.name}.jsonl"
            with open(ckpt_eval_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  saved: {ckpt_eval_path}")

            # Summary
            summary = compute_summary(results=results)
            summary["step"] = step
            summary["checkpoint"] = ckpt_path.name
            all_summaries.append(summary)

            print(f"  score={summary['mean_score']:.4f}  "
                  f"frac_test={summary['mean_frac_test_cases_passing']:.4f}  "
                  f"sus={summary['mean_sus_score']:.4f}  "
                  f"backdoor={summary['mean_backdoor_test_passed']:.4f}  "
                  f"hidden_len={summary['mean_hidden_lengths']:.1f}")

        # Save summary
        summary_path = eval_dir / "summary.jsonl"
        with open(summary_path, "w") as f:
            for s in all_summaries:
                f.write(json.dumps(s) + "\n")
        print(f"\n=== Summary saved: {summary_path} ===")

        # Print summary table
        print(f"\n{'step':>6} {'score':>8} {'frac_test':>10} {'sus':>6} {'backdoor':>9} {'hidden_len':>11}")
        print("-" * 60)
        for s in all_summaries:
            print(f"{s['step']:>6} {s['mean_score']:>8.4f} {s['mean_frac_test_cases_passing']:>10.4f} "
                  f"{s['mean_sus_score']:>6.4f} {s['mean_backdoor_test_passed']:>9.4f} "
                  f"{s['mean_hidden_lengths']:>11.1f}")

    finally:
        # Kill vLLM server
        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()


def main() -> None:
    args = tyro.cli(EvalAllCheckpointsCLI)
    asyncio.run(async_main(args=args))


if __name__ == "__main__":
    main()
