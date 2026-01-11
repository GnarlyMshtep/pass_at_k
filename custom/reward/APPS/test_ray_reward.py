#!/usr/bin/env python3
"""Standalone Ray reward benchmark for testing remote CPU nodes.

Usage:
    # Start Ray head node
    ray start --head --port=6379

    # Run benchmark
    python test_ray_reward.py --samples samples.jsonl

    # With different reward function
    python test_ray_reward.py --samples samples.jsonl --reward-fn backdoor_wo_monitor
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import ray

# Reward function registry
REWARD_FUNCTIONS = {}

# Default model for tokenizer
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def get_reward_function(name: str):
    """Get reward function by name, with lazy import."""
    if name == "benign_prompt":
        from custom.reward.APPS.APPS_reward import reward_func_benign_prompt
        return reward_func_benign_prompt
    elif name == "backdoor_wo_monitor":
        from custom.reward.APPS.APPS_reward import reward_func_w_backdoor_wo_monitor_outputonly_formatter
        return reward_func_w_backdoor_wo_monitor_outputonly_formatter
    elif name == "backdoor_full":
        from custom.reward.APPS.APPS_reward import reward_func_w_backdoor_outputonly_formatter
        return reward_func_w_backdoor_outputonly_formatter
    else:
        raise ValueError(f"Unknown reward function: {name}")


async def process_one_with_detokenize(tokenizer, compute_score_fn, item: dict) -> tuple:
    """Process a single sample, detokenizing first (like naive.py)."""
    try:
        # Detokenize the response tokens
        solution_str = tokenizer.decode(item["response_ids"], skip_special_tokens=True)

        result = await compute_score_fn(
            data_source=item["data_source"],
            solution_str=solution_str,
            ground_truth=item["ground_truth"],
            extra_info=item["extra_info"],
        )
        return (result, item["i"])
    except Exception as e:
        print(f"WARNING: Task {item['i']} failed with error: {e}")
        return ({"score": 0.0, "error": str(e)}, item["i"])


@ray.remote
def compute_several_with_detokenize(tokenizer, compute_score_fn, items: list[dict]) -> list[tuple]:
    """Process multiple samples in one Ray task, with detokenization.

    Matches naive.py's compute_several - includes detokenization in the timed portion.
    """
    async def run_all():
        tasks = [process_one_with_detokenize(tokenizer, compute_score_fn, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return asyncio.run(run_all())


async def process_one(compute_score_fn, item: dict) -> tuple:
    """Process a single sample (no tokenizer version)."""
    try:
        result = await compute_score_fn(
            data_source=item["data_source"],
            solution_str=item["solution_str"],
            ground_truth=item["ground_truth"],
            extra_info=item["extra_info"],
        )
        return (result, item["i"])
    except Exception as e:
        print(f"WARNING: Task {item['i']} failed with error: {e}")
        return ({"score": 0.0, "error": str(e)}, item["i"])


@ray.remote
def compute_several(compute_score_fn, items: list[dict]) -> list[tuple]:
    """Process multiple samples in one Ray task (no tokenizer version)."""
    async def run_all():
        tasks = [process_one(compute_score_fn, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return asyncio.run(run_all())


def load_samples(path: Path) -> list[dict]:
    """Load samples from jsonl file."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def prepare_items(samples: list[dict], tokenizer=None) -> list[dict]:
    """Convert samples to the format expected by compute_several.

    If tokenizer is provided, tokenizes solution strings to simulate the RL pipeline.
    """
    items = []
    for i, s in enumerate(samples):
        # Handle response format - it's a dict with 'output' key
        response = s["generation"]["response"]
        if isinstance(response, dict):
            solution_str = response.get("output", "")
        else:
            solution_str = response

        # Build extra_info from question, adding full_prompt from generation level
        extra_info = dict(s["generation"]["question"])
        if "full_prompt" not in extra_info:
            extra_info["full_prompt"] = ["dummy full prompt -- ireelvent for grading"]

        item = {
            "extra_info": extra_info,
            "ground_truth": None,
            "data_source": "apps",
            "i": i,
        }

        if tokenizer is not None:
            # Tokenize solution string (simulates model output as token IDs)
            item["response_ids"] = tokenizer.encode(solution_str, add_special_tokens=False)
        else:
            item["solution_str"] = solution_str

        items.append(item)
    return items


async def run_benchmark_async(items: list[dict], compute_score_fn, tokenizer, bucket_size: int, mini_bucket_size: int):
    """Run the benchmark using Ray, matching naive.py's structure."""

    total_start = time.time()
    all_results = []

    num_chunks = math.ceil(len(items) / bucket_size)
    print(f"Processing {len(items)} samples in {num_chunks} chunk(s)")
    print(f"  bucket_size={bucket_size}, mini_bucket_size={mini_bucket_size}")
    print(f"  tokenizer: {'yes (includes detokenization)' if tokenizer else 'no'}")

    for chunk_idx in range(num_chunks):
        chunk_start = time.time()

        start_idx = chunk_idx * bucket_size
        end_idx = min((chunk_idx + 1) * bucket_size, len(items))
        chunk_items = items[start_idx:end_idx]

        # Batch items into mini-batches for Ray tasks
        batches = [
            chunk_items[j:j + mini_bucket_size]
            for j in range(0, len(chunk_items), mini_bucket_size)
        ]

        print(f"  Chunk {chunk_idx}: {len(chunk_items)} samples -> {len(batches)} Ray tasks")

        # Dispatch Ray tasks
        if tokenizer is not None:
            ray_refs = [
                compute_several_with_detokenize.remote(
                    tokenizer=tokenizer,
                    compute_score_fn=compute_score_fn,
                    items=batch,
                )
                for batch in batches
            ]
        else:
            ray_refs = [
                compute_several.remote(
                    compute_score_fn=compute_score_fn,
                    items=batch,
                )
                for batch in batches
            ]

        # Wait for results
        timeout = 180.0  # generous timeout for testing
        try:
            rets_nested = await asyncio.wait_for(
                asyncio.gather(*ray_refs, return_exceptions=True),
                timeout=timeout,
            )

            # Flatten results
            for batch_result in rets_nested:
                if isinstance(batch_result, Exception):
                    print(f"  WARNING: Batch failed with {batch_result}")
                else:
                    all_results.extend(batch_result)

        except asyncio.TimeoutError:
            print(f"  ERROR: Chunk {chunk_idx} timed out after {timeout}s")

        chunk_elapsed = time.time() - chunk_start
        print(f"  Chunk {chunk_idx} completed in {chunk_elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    return all_results, total_elapsed


def run_benchmark(items: list[dict], compute_score_fn, tokenizer=None, bucket_size: int = 2000, mini_bucket_size: int = 10):
    """Sync wrapper for the benchmark."""
    return asyncio.run(run_benchmark_async(items, compute_score_fn, tokenizer, bucket_size, mini_bucket_size))


def print_cluster_info():
    """Print Ray cluster information."""
    nodes = ray.nodes()
    print(f"\nRay Cluster Info:")
    print(f"  Nodes: {len(nodes)}")

    total_cpus = 0
    total_gpus = 0
    for node in nodes:
        resources = node.get("Resources", {})
        cpus = resources.get("CPU", 0)
        gpus = resources.get("GPU", 0)
        total_cpus += cpus
        total_gpus += gpus

        node_id = node.get("NodeID", "unknown")[:8]
        alive = "alive" if node.get("Alive", False) else "dead"
        print(f"    {node_id}: {cpus} CPUs, {gpus} GPUs ({alive})")

    print(f"  Total: {total_cpus} CPUs, {total_gpus} GPUs\n")


def main():
    parser = argparse.ArgumentParser(description="Ray reward benchmark")
    parser.add_argument("--samples", type=Path, required=True, help="Path to samples.jsonl")
    parser.add_argument("--reward-fn", default="benign_prompt",
                        choices=["benign_prompt", "backdoor_wo_monitor", "backdoor_full"],
                        help="Reward function to use")
    parser.add_argument("--bucket-size", type=int, default=2000, help="Chunk size")
    parser.add_argument("--mini-bucket-size", type=int, default=10, help="Ray task batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for quick testing)")
    parser.add_argument("--tokenize", action="store_true", default=False,
                        help="Tokenize inputs and include detokenization in benchmark (like naive.py)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model for tokenizer (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Initialize Ray (connect to existing cluster or start local)
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    print_cluster_info()

    # Load tokenizer if requested
    tokenizer = None
    if args.tokenize:
        print(f"Loading tokenizer from {args.model}...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load samples
    print(f"Loading samples from {args.samples}")
    samples = load_samples(args.samples)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} samples")

    # Prepare items (tokenize if tokenizer is provided)
    print("Preparing items...")
    prep_start = time.time()
    items = prepare_items(samples, tokenizer=tokenizer)
    prep_elapsed = time.time() - prep_start
    print(f"  Prepared {len(items)} items in {prep_elapsed:.2f}s (not counted in benchmark)")

    # Get reward function
    print(f"Using reward function: {args.reward_fn}")
    compute_score_fn = get_reward_function(args.reward_fn)

    # Run benchmark
    print("\nStarting benchmark...")
    results, elapsed = run_benchmark(
        items,
        compute_score_fn,
        tokenizer=tokenizer,
        bucket_size=args.bucket_size,
        mini_bucket_size=args.mini_bucket_size
    )

    # Report results
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total samples: {len(items)}")
    print(f"Successful results: {len(results)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {len(items)/elapsed:.2f} samples/sec")
    print(f"Avg time per sample: {elapsed/len(items)*1000:.1f}ms")

    # Score summary
    scores = []
    results_by_idx = {}
    for result in results:
        if isinstance(result, tuple) and len(result) >= 2:
            score_dict, idx = result
            results_by_idx[idx] = score_dict
            if isinstance(score_dict, dict) and "score" in score_dict:
                scores.append(score_dict["score"])

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Avg score: {avg_score:.4f}")
        print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")

    # Verify frac_test_cases_passing against logged values
    print(f"\n{'='*50}")
    print(f"VERIFICATION: frac_test_cases_passing")
    print(f"{'='*50}")
    mismatches = []
    for i, sample in enumerate(samples):
        if i not in results_by_idx:
            print(f"  Sample {i}: MISSING from results")
            continue

        result = results_by_idx[i]
        logged = sample.get('frac_test_cases_passing')
        computed = result.get('frac_test_cases_passing', 0) if isinstance(result, dict) else 0

        if logged is not None and abs(logged - computed) > 1e-6:
            mismatches.append((i, logged, computed))

    if mismatches:
        print(f"  MISMATCHES: {len(mismatches)}/{len(samples)} ({len(mismatches)/len(samples)*100:.1f}%)")
        for idx, logged, computed in mismatches[:10]:  # Show first 10
            print(f"    Sample {idx}: logged={logged:.4f}, computed={computed:.4f}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")

        # Log mismatches to file for investigation
        mismatch_file = args.samples.parent / "mismatches.jsonl"
        print(f"\n  Logging mismatches to: {mismatch_file}")
        with open(mismatch_file, "w") as f:
            for idx, logged, computed in mismatches:
                sample = samples[idx]
                result = results_by_idx.get(idx, {})
                # Get solution_str from original sample (handles both tokenized and non-tokenized cases)
                response = sample["generation"]["response"]
                solution_str = response.get("output", "") if isinstance(response, dict) else response
                mismatch_record = {
                    "idx": idx,
                    "logged_frac": logged,
                    "computed_frac": computed,
                    "logged_error": sample.get("error"),
                    "computed_error": result.get("error") if isinstance(result, dict) else str(result),
                    "problem_id": sample.get("generation", {}).get("question", {}).get("problem_id"),
                    "solution_str": solution_str[:500],  # First 500 chars
                    "extracted_code": result.get("extracted_code", "") if isinstance(result, dict) else "",
                }
                f.write(json.dumps(mismatch_record) + "\n")
    else:
        print(f"  All {len(samples)} samples match!")

    # Verify extracted_code matches logged values
    print(f"\n{'='*50}")
    print(f"VERIFICATION: extracted_code")
    print(f"{'='*50}")
    code_mismatches = []
    code_matches = 0
    for i, sample in enumerate(samples):
        if i not in results_by_idx:
            continue

        result = results_by_idx[i]
        logged_code = sample['generation'].get('code', '')
        computed_code = result.get('extracted_code', '') if isinstance(result, dict) else ''

        # Normalize whitespace for comparison
        logged_norm = (logged_code or '').strip()
        computed_norm = (computed_code or '').strip()

        if logged_norm != computed_norm:
            code_mismatches.append((i, logged_code, computed_code))
        else:
            code_matches += 1

    if code_mismatches:
        print(f"  CODE MISMATCHES: {len(code_mismatches)}/{len(samples)} ({len(code_mismatches)/len(samples)*100:.1f}%)")
        print(f"  CODE MATCHES: {code_matches}/{len(samples)} ({code_matches/len(samples)*100:.1f}%)")

        # Show first few mismatches
        for idx, logged, computed in code_mismatches[:3]:
            print(f"\n  Sample {idx}:")
            print(f"    Logged code (first 150): {repr((logged or '')[:150])}")
            print(f"    Computed code (first 150): {repr((computed or '')[:150])}")

        if len(code_mismatches) > 3:
            print(f"\n    ... and {len(code_mismatches) - 3} more code mismatches")

        # Log code mismatches to file
        code_mismatch_file = args.samples.parent / "code_mismatches.jsonl"
        print(f"\n  Logging code mismatches to: {code_mismatch_file}")
        with open(code_mismatch_file, "w") as f:
            for idx, logged_code, computed_code in code_mismatches:
                sample = samples[idx]
                result = results_by_idx.get(idx, {})
                record = {
                    "idx": idx,
                    "problem_id": sample.get("generation", {}).get("question", {}).get("problem_id"),
                    "logged_frac": sample.get('frac_test_cases_passing'),
                    "computed_frac": result.get('frac_test_cases_passing', 0) if isinstance(result, dict) else 0,
                    "logged_code": logged_code,
                    "computed_code": computed_code,
                    "logged_error": sample.get("error"),
                    "computed_error": result.get("error") if isinstance(result, dict) else str(result),
                }
                f.write(json.dumps(record) + "\n")
    else:
        print(f"  All {len(samples)} extracted codes match!")


if __name__ == "__main__":
    main()
