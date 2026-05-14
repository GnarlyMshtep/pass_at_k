"""Standalone evaluation: score model outputs against a reward function.

Two-phase architecture:
  1. Generate — load parquet, render prompts, query vLLM → completions.jsonl
  2. Grade   — build DataProto, score via NaiveRewardManager → results.jsonl + summary.json

Usage:
    # With local model (launches vLLM):
    python -m vfh.standalone_eval \
        --reward-path custom/reward/reward_utils.py \
        --reward-name compute_score_math_boxed \
        --model Qwen/Qwen3-4B \
        --data-path $HF_HOME/data/.../test.parquet \
        --limit 10 --global-step 0 --vllm-gpu-util 0.4

    # With existing vLLM server:
    python -m vfh.standalone_eval \
        --reward-path custom/reward/reward_utils.py \
        --reward-name compute_score_math_boxed \
        --vllm-url http://localhost:8000/v1 \
        --data-path $HF_HOME/data/.../test.parquet \
        --limit 10 --global-step 0
"""

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial

import numpy as np
import ray
import torch
from openai import AsyncOpenAI
from tensordict import TensorDict
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

import tyro
from verl import DataProto
from verl.trainer.ppo.reward import _call_with_kwargs, _derive_module_name_from_path
from verl.workers.reward_manager.naive import NaiveRewardManager

from vfh.standalone_eval_types import StandaloneEvalConfig, EvalRow


# ─────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────


def _validate_config(config: StandaloneEvalConfig) -> None:
    if not config.model and not config.vllm_url:
        raise ValueError("Must specify either --model or --vllm-url")
    if config.model and config.vllm_url:
        raise ValueError("Specify only one of --model or --vllm-url")
    if not config.reward_path or not config.reward_name:
        raise ValueError("--reward-path and --reward-name are required")
    if not config.data_path:
        raise ValueError("--data-path is required")
    data_path = os.path.expandvars(config.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.exists(config.reward_path):
        raise FileNotFoundError(f"Reward function not found: {config.reward_path}")
    if config.model and config.vllm_gpu_util is None:
        raise ValueError("--vllm-gpu-util is required when using --model")

    # Reuse validate_env checks for reward function and dataset
    from validate_env import check_reward_function, check_parquet_readable

    if not check_parquet_readable(filepath=data_path, file_type="data_path"):
        raise ValueError(f"Dataset parquet is not readable: {data_path}")

    reward_kwargs_json = config.reward_kwargs_json if config.reward_kwargs_json != "{}" else None
    if not check_reward_function(
        reward_path=config.reward_path,
        reward_name=config.reward_name,
        reward_kwargs_json=reward_kwargs_json,
    ):
        raise ValueError(
            f"Reward function validation failed: {config.reward_name} from {config.reward_path}"
        )


# ─────────────────────────────────────────────────
# Eval directory
# ─────────────────────────────────────────────────


def _create_eval_dir(config: StandaloneEvalConfig) -> str:
    now = datetime.now()
    desc = (config.description or "eval").replace(" ", "_").replace("/", "_")[:60]
    dir_name = f"{desc}_{now.strftime('%H_%M')}"
    eval_dir = os.path.join(
        "logs", "StandaloneEvals", now.strftime("%m"), now.strftime("%d"), dir_name
    )
    base = eval_dir
    counter = 1
    while os.path.exists(eval_dir):
        eval_dir = f"{base}#{counter}"
        counter += 1
    os.makedirs(eval_dir)

    with open(os.path.join(eval_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    return eval_dir


# ─────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────


def _load_dataset(config: StandaloneEvalConfig) -> list[dict]:
    import pandas as pd

    data_path = os.path.expandvars(config.data_path)
    df = pd.read_parquet(data_path)
    rows: list[dict] = []
    for idx in range(len(df)):
        row = df.iloc[idx]

        prompt = row[config.prompt_key]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()

        rm = row.get("reward_model", {})
        if isinstance(rm, np.ndarray):
            rm = rm.item() if rm.ndim == 0 else dict(rm)
        if not isinstance(rm, dict):
            rm = {}

        extra = row.get("extra_info", {})
        if isinstance(extra, np.ndarray):
            extra = extra.item() if extra.ndim == 0 else dict(extra)
        if not isinstance(extra, dict):
            extra = {}

        rows.append(
            {
                "data_source": str(row.get("data_source", "unknown")),
                "prompt": list(prompt),
                "reward_model": rm,
                "extra_info": extra,
            }
        )

    if config.limit > 0:
        rows = rows[: config.limit]
    return rows


# ─────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────


def _render_prompts(rows: list[dict], tokenizer) -> list[str]:
    rendered: list[str] = []
    for row in rows:
        prompt_str = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=False
        )
        rendered.append(prompt_str)
    return rendered


# ─────────────────────────────────────────────────
# vLLM server management
# ─────────────────────────────────────────────────


def _print_vllm_config(config: StandaloneEvalConfig) -> None:
    print("┌─── vLLM Server ───────────────────────┐")
    print(f"│ Model:    {config.model}")
    print(f"│ Port:     {config.vllm_port}")
    print(f"│ GPU util: {config.vllm_gpu_util}")
    tp = config.vllm_tp or 1
    print(f"│ TP:       {tp}")
    if config.vllm_dp:
        print(f"│ DP:       {config.vllm_dp}")
    if config.vllm_max_model_len:
        print(f"│ Max len:  {config.vllm_max_model_len}")
    print("└───────────────────────────────────────┘")


def _launch_vllm_server(config: StandaloneEvalConfig) -> subprocess.Popen:
    _print_vllm_config(config)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model,
        "--port",
        str(config.vllm_port),
        "--gpu-memory-utilization",
        str(config.vllm_gpu_util),
        "--tensor-parallel-size",
        str(config.vllm_tp or 1),
        "--disable-log-requests",
    ]
    if config.vllm_dp:
        cmd.extend(["--data-parallel-size", str(config.vllm_dp)])
    if config.vllm_max_model_len:
        cmd.extend(["--max-model-len", str(config.vllm_max_model_len)])
    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stderr=subprocess.STDOUT)
    return proc


async def _wait_for_vllm(base_url: str, timeout: float = 300.0) -> None:
    import httpx

    t0 = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - t0 < timeout:
            try:
                resp = await client.get(f"{base_url}/models")
                if resp.status_code == 200:
                    models = resp.json()
                    print(
                        f"vLLM ready ({time.time() - t0:.1f}s). "
                        f"Models: {[m['id'] for m in models['data']]}"
                    )
                    return
            except Exception:
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"vLLM not ready after {timeout}s")


async def _get_model_name(base_url: str) -> str:
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/models")
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]


# ─────────────────────────────────────────────────
# Generation phase
# ─────────────────────────────────────────────────


async def _generate_and_save(
    rows: list[dict],
    prompts: list[str],
    base_url: str,
    model_name: str,
    config: StandaloneEvalConfig,
    eval_dir: str,
) -> list[list[str]]:
    """Generate completions and save to completions.jsonl. Returns the completions."""
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(50)

    async def gen_one(idx: int, prompt: str) -> list[str]:
        async with semaphore:
            resp = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                n=config.n,
            )
            return [c.text for c in resp.choices]

    all_completions: list[list[str]] = await tqdm_asyncio.gather(
        *[gen_one(i, p) for i, p in enumerate(prompts)],
        desc="Generating",
    )

    # Save completions
    completions_path = os.path.join(eval_dir, "completions.jsonl")
    with open(completions_path, "w", encoding="utf-8") as f:
        for i, (row, prompt, comps) in enumerate(zip(rows, prompts, all_completions)):
            line = {
                "i": i,
                "data_source": row["data_source"],
                "prompt_rendered": prompt,
                "completions": comps,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    total = sum(len(c) for c in all_completions)
    print(f"Generated {total} completions → {completions_path}")
    return all_completions


# ─────────────────────────────────────────────────
# Reward binding
# ─────────────────────────────────────────────────


def _bind_reward_fn(config: StandaloneEvalConfig):
    reward_kwargs = json.loads(config.reward_kwargs_json)
    file_path = config.reward_path
    module_name = _derive_module_name_from_path(file_path)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not create module spec for {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, config.reward_name):
        raise AttributeError(
            f"Function '{config.reward_name}' not found in '{file_path}'"
        )
    raw_fn = getattr(module, config.reward_name)

    if reward_kwargs:
        wrapped = partial(_call_with_kwargs, raw_fn, reward_kwargs)
        setattr(wrapped, "__custom_file_path__", file_path)
        setattr(wrapped, "__custom_function_name__", config.reward_name)
        setattr(wrapped, "__custom_reward_kwargs__", reward_kwargs)
        setattr(wrapped, "__verl_custom_loader__", True)
    else:
        wrapped = raw_fn

    print(f"Reward: {config.reward_name} from {file_path}")
    if reward_kwargs:
        print(f"  kwargs: {reward_kwargs}")
    return wrapped


# ─────────────────────────────────────────────────
# DataProto construction
# ─────────────────────────────────────────────────


def _build_data_proto(
    rows: list[dict],
    prompts_rendered: list[str],
    completions: list[list[str]],
    tokenizer,
    config: StandaloneEvalConfig,
) -> tuple[DataProto, list[tuple[int, int, list[int], list[int]]]]:
    """Build DataProto for NaiveRewardManager.

    Returns:
        (data_proto, sample_info) where sample_info is a list of
        (row_idx, n_idx, prompt_ids, resp_ids) per sample in the batch.
    """
    max_prompt_len = 0
    max_resp_len = 0

    # First pass: tokenize and find max lengths
    # sample_info: (row_idx, n_idx, prompt_ids, resp_ids)
    sample_info: list[tuple[int, int, list[int], list[int]]] = []
    for row_idx, (prompt_str, row_completions) in enumerate(
        zip(prompts_rendered, completions)
    ):
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        for n_idx, comp_str in enumerate(row_completions):
            resp_ids = tokenizer.encode(comp_str, add_special_tokens=False)
            sample_info.append((row_idx, n_idx, prompt_ids, resp_ids))
            max_prompt_len = max(max_prompt_len, len(prompt_ids))
            max_resp_len = max(max_resp_len, len(resp_ids))

    # Second pass: pad and build tensors
    all_prompt_ids: list[torch.Tensor] = []
    all_response_ids: list[torch.Tensor] = []
    all_attn_masks: list[torch.Tensor] = []
    non_tensor: dict[str, list] = {
        config.reward_fn_key: [],
        "reward_model": [],
        "extra_info": [],
    }
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for row_idx, n_idx, prompt_ids, resp_ids in sample_info:
        row = rows[row_idx]

        # Left-pad prompt
        p_pad = max_prompt_len - len(prompt_ids)
        padded_prompt = [pad_id] * p_pad + prompt_ids
        prompt_mask = [0] * p_pad + [1] * len(prompt_ids)

        # Right-pad response
        r_pad = max_resp_len - len(resp_ids)
        padded_resp = resp_ids + [pad_id] * r_pad
        resp_mask = [1] * len(resp_ids) + [0] * r_pad

        all_prompt_ids.append(torch.tensor(padded_prompt, dtype=torch.long))
        all_response_ids.append(torch.tensor(padded_resp, dtype=torch.long))
        all_attn_masks.append(torch.tensor(prompt_mask + resp_mask, dtype=torch.long))

        non_tensor[config.reward_fn_key].append(row["data_source"])
        non_tensor["reward_model"].append(row["reward_model"])
        non_tensor["extra_info"].append(row["extra_info"])

    batch_size = len(sample_info)
    batch_td = TensorDict(
        {
            "prompts": torch.stack(all_prompt_ids),
            "responses": torch.stack(all_response_ids),
            "attention_mask": torch.stack(all_attn_masks),
        },
        batch_size=[batch_size],
    )

    data = DataProto(
        batch=batch_td,
        non_tensor_batch={
            k: np.array(v, dtype=object) for k, v in non_tensor.items()
        },
        meta_info={"matan_reward_global_step": config.global_step},
    )
    return data, sample_info


# ─────────────────────────────────────────────────
# Grading phase
# ─────────────────────────────────────────────────


async def _grade_completions(
    rows: list[dict],
    prompts_rendered: list[str],
    completions: list[list[str]],
    tokenizer,
    reward_fn,
    config: StandaloneEvalConfig,
    eval_dir: str,
) -> list[float]:
    """Build DataProto, run NaiveRewardManager, write results.jsonl + summary.json."""

    # Build DataProto
    data_proto, sample_info = _build_data_proto(
        rows=rows,
        prompts_rendered=prompts_rendered,
        completions=completions,
        tokenizer=tokenizer,
        config=config,
    )
    print(f"DataProto built: {len(data_proto)} samples")

    # Score via NaiveRewardManager
    # Call _compute_rewards_async directly since we're already in an async context
    # (NaiveRewardManager.__call__ uses asyncio.run() which can't nest)
    reward_manager = NaiveRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=reward_fn,
        reward_fn_key=config.reward_fn_key,
    )
    print("Scoring via NaiveRewardManager...")
    reward_result = await reward_manager._compute_rewards_async(data_proto, return_dict=True)

    # Extract and log
    reward_tensor = reward_result["reward_tensor"]
    extra_info_dict = reward_result.get("reward_extra_info", {})

    results_path = os.path.join(eval_dir, "results.jsonl")
    all_scores: list[float] = []

    with open(results_path, "w", encoding="utf-8") as f:
        for sample_idx, (row_idx, n_idx, _prompt_ids, resp_ids) in enumerate(
            sample_info
        ):
            row = rows[row_idx]
            score_val = reward_tensor[sample_idx].sum().item()

            # Collect reward extra info for this sample
            reward_details: dict = {}
            for key, values in extra_info_dict.items():
                if sample_idx < len(values):
                    v = values[sample_idx]
                    # Strip the reward_extra_info/ prefix for flattening
                    clean_key = key.replace("reward_extra_info/", "")
                    reward_details[clean_key] = (
                        float(v) if isinstance(v, (int, float, bool)) else v
                    )

            ground_truth = row.get("reward_model", {}).get("ground_truth")

            eval_row = EvalRow(
                i=sample_idx,
                n_idx=n_idx,
                data_source=row["data_source"],
                prompt_messages=row["prompt"],
                prompt_rendered=prompts_rendered[row_idx],
                response=completions[row_idx][n_idx],
                ground_truth=ground_truth,
                extra_info=row["extra_info"],
                score=score_val,
                reward_details=reward_details,
                n_response_tokens=len(resp_ids),
                global_step=config.global_step,
            )

            # Flatten: merge reward_details into top-level, remove nested dict
            line = asdict(eval_row)
            details = line.pop("reward_details")
            line.update(details)

            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            all_scores.append(score_val)

    # Summary
    summary = {
        "total_samples": len(all_scores),
        "n_per_prompt": config.n,
        "num_prompts": len(rows),
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "nonzero": sum(1 for s in all_scores if s != 0),
        "frac_nonzero": (
            sum(1 for s in all_scores if s != 0) / len(all_scores)
            if all_scores
            else 0
        ),
        "min_score": min(all_scores) if all_scores else 0,
        "max_score": max(all_scores) if all_scores else 0,
        "global_step": config.global_step,
    }
    summary_path = os.path.join(eval_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {results_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    return all_scores


# ─────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────


async def run_eval(config: StandaloneEvalConfig) -> None:
    _validate_config(config)
    eval_dir = _create_eval_dir(config)
    print(f"Eval dir: {eval_dir}")

    # Load dataset
    rows = _load_dataset(config)
    print(f"Loaded {len(rows)} samples from {config.data_path}")

    # Resolve model name and tokenizer
    model_name = config.model
    base_url = config.vllm_url
    vllm_proc: subprocess.Popen | None = None

    if config.model:
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        base_url = f"http://localhost:{config.vllm_port}/v1"
    else:
        model_name = await _get_model_name(base_url)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer: {model_name}")

    # Render prompts
    prompts = _render_prompts(rows=rows, tokenizer=tokenizer)

    # Launch vLLM if needed
    if config.model:
        vllm_proc = _launch_vllm_server(config)
        await _wait_for_vllm(base_url)
        if not model_name:
            model_name = await _get_model_name(base_url)

    try:
        # ── PHASE 1: Generate ──
        completions = await _generate_and_save(
            rows=rows,
            prompts=prompts,
            base_url=base_url,
            model_name=model_name,
            config=config,
            eval_dir=eval_dir,
        )

        # ── PHASE 2: Grade ──
        if not ray.is_initialized():
            ray.init(num_cpus=config.num_ray_cpus)
            print(f"Ray initialized with {config.num_ray_cpus} CPUs")

        reward_fn = _bind_reward_fn(config)

        await _grade_completions(
            rows=rows,
            prompts_rendered=prompts,
            completions=completions,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=config,
            eval_dir=eval_dir,
        )

    finally:
        if vllm_proc is not None:
            print("Shutting down vLLM...")
            vllm_proc.send_signal(signal.SIGTERM)
            try:
                vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
            print("vLLM stopped.")

        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown.")

    print(f"\nDone. Eval dir: {eval_dir}")


def main():
    config = tyro.cli(StandaloneEvalConfig)
    asyncio.run(run_eval(config))


if __name__ == "__main__":
    main()
