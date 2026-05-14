"""Inference-only evaluation of LoRA checkpoints from SFT runs.

Launches vLLM servers with the base model + a small batch of LoRA adapters
(default 3 at a time), generates completions, kills the server, repeats.
This keeps KV cache large and GPU utilization high.

NO reward scoring — that happens in a separate pass.

Usage:
    python -m TRLSFT.runners.inference_all_checkpoints \
        --run-dir logs/SFTRuns/04/25/lbl_q4bi_lr2e5_... \
        --eval-source /path/to/rollouts/200.jsonl

Results written to:
    {run_dir}/post-hoc-evals/{MM_DD_HH_mm}/
        checkpoint-{N}.jsonl   (one line per generation: prompt + completion)
        eval_config.json       (CLI args)
        vllm_server_batch_{K}.log
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import tyro

from TRLSFT.runners.eval_all_checkpoints import (
    copy_model_to_local,
    find_checkpoints,
    get_base_model_path,
    load_eval_questions,
    start_vllm_server,
    wait_for_server,
)
from TRLSFT.utils.async_utils import matan_gather_chunked


@dataclass
class InferenceOnlyCLI:
    """Run inference on strided LoRA checkpoints from an SFT run (no reward scoring)."""

    run_dir: str
    """Path to SFT run directory"""
    eval_source: str
    """Path to rollout file with eval questions"""
    checkpoint_stride: int = 1
    """Take every Nth checkpoint (1=all, 3=every 3rd)"""
    n_repeats: int = 10
    """Number of times to generate per question per checkpoint"""
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
    gen_chunk_size: int = 300
    """Chunk size for matan_gather_chunked"""
    adapters_per_batch: int = 3
    """Number of LoRA adapters to load per vLLM server instance"""
    resume_dir: str | None = None
    """Resume from an existing eval dir (skip already-completed checkpoints)"""


def kill_server(proc: subprocess.Popen) -> None:
    """Terminate vLLM server and wait for cleanup."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


async def infer_single_checkpoint(
    step: int,
    ckpt_path: Path,
    questions: list[dict],
    n_repeats: int,
    api_base: str,
    max_tokens: int,
    temperature: float,
    gen_semaphore: asyncio.Semaphore,
    eval_dir: Path,
    chunk_size: int,
) -> int:
    """Generate completions for a single checkpoint. Returns number of generations."""
    from openai import AsyncOpenAI

    model_name = f"step_{step}"
    client = AsyncOpenAI(base_url=api_base, api_key="EMPTY", timeout=600.0)
    n_questions = len(questions)
    total = n_questions * n_repeats
    print(f"[{model_name}] Starting: {n_questions} questions × {n_repeats} repeats = {total} generations")
    t0 = time.time()

    async def generate_one(q_idx: int, repeat_idx: int) -> dict:
        async with gen_semaphore:
            response = await client.completions.create(
                model=model_name,
                prompt=questions[q_idx]["input"],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return {
            "step": step,
            "checkpoint": ckpt_path.name,
            "q_idx": q_idx,
            "repeat_idx": repeat_idx,
            "prompt": questions[q_idx]["input"],
            "completion": response.choices[0].text,
        }

    coros = [
        generate_one(q_idx=q, repeat_idx=r)
        for r in range(n_repeats)
        for q in range(n_questions)
    ]
    results = await matan_gather_chunked(*coros, chunk_size=chunk_size)

    out_path = eval_dir / f"{ckpt_path.name}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    print(f"[{model_name}] Done: {len(results)} generations in {elapsed:.0f}s -> {out_path}")
    return len(results)


async def async_main(args: InferenceOnlyCLI) -> None:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoints = find_checkpoints(run_dir=run_dir, stride=args.checkpoint_stride)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir / 'checkpoints'}")
    print(f"Found {len(checkpoints)} checkpoints (stride={args.checkpoint_stride}): {[s for s, _ in checkpoints]}")

    base_model = get_base_model_path(run_dir=run_dir)
    if args.local_model_copy:
        base_model = copy_model_to_local(model_path=base_model)

    if args.resume_dir:
        eval_dir = Path(args.resume_dir)
        if not eval_dir.exists():
            raise FileNotFoundError(f"Resume dir not found: {eval_dir}")
        print(f"Resuming in: {eval_dir}")
    else:
        now = datetime.now()
        eval_dir = run_dir / "post-hoc-evals" / f"{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {eval_dir}")

    with open(eval_dir / "eval_config.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    questions = load_eval_questions(eval_source_file=args.eval_source, seed=args.seed)
    print(f"Loaded {len(questions)} eval questions, {args.n_repeats} repeats each = {len(questions) * args.n_repeats} generations/checkpoint")

    # Chunk checkpoints into batches of adapters_per_batch
    batch_size = args.adapters_per_batch
    ckpt_batches = [
        checkpoints[i : i + batch_size]
        for i in range(0, len(checkpoints), batch_size)
    ]
    print(f"Will process {len(ckpt_batches)} adapter batches of up to {batch_size} adapters each")

    total_gens = 0
    t0_all = time.time()

    for batch_idx, batch in enumerate(ckpt_batches):
        # Skip already-completed checkpoints (resume support)
        remaining = [
            (step, path) for step, path in batch
            if not (eval_dir / f"{path.name}.jsonl").exists()
        ]
        if not remaining:
            print(f"\n[Batch {batch_idx + 1}/{len(ckpt_batches)}] All checkpoints already done, skipping")
            continue

        lora_adapters = [
            (f"step_{step}", str(path.resolve()))
            for step, path in remaining
        ]

        print(f"\n[Batch {batch_idx + 1}/{len(ckpt_batches)}] Starting vLLM with {len(lora_adapters)} adapters: {[name for name, _ in lora_adapters]}")
        # Find next available log index to avoid overwriting previous runs' logs
        log_idx = batch_idx
        while (eval_dir / f"vllm_server_batch_{log_idx}.log").exists():
            log_idx += 1
        server_log = eval_dir / f"vllm_server_batch_{log_idx}.log"
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
                kill_server(proc=server_proc)
                print(f"ERROR: vLLM server failed to start for batch {batch_idx + 1}. Check {server_log}")
                sys.exit(1)

            api_base = f"http://localhost:{args.port}/v1"
            gen_semaphore = asyncio.Semaphore(args.gen_concurrency)

            for step, ckpt_path in remaining:
                n = await infer_single_checkpoint(
                    step=step,
                    ckpt_path=ckpt_path,
                    questions=questions,
                    n_repeats=args.n_repeats,
                    api_base=api_base,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    gen_semaphore=gen_semaphore,
                    eval_dir=eval_dir,
                    chunk_size=args.gen_chunk_size,
                )
                total_gens += n

        finally:
            print(f"[Batch {batch_idx + 1}] Shutting down vLLM server...")
            kill_server(proc=server_proc)
            time.sleep(2)

    elapsed = time.time() - t0_all
    print(f"\nAll done: {total_gens} total generations across {len(checkpoints)} checkpoints in {elapsed:.0f}s")
    print(f"Output dir: {eval_dir}")


def main() -> None:
    args = tyro.cli(InferenceOnlyCLI)
    asyncio.run(async_main(args=args))


if __name__ == "__main__":
    main()
