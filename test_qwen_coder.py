#!/usr/bin/env python3
"""
Test script for evaluating Qwen2.5-Coder-7B-Instruct on the code dataset.

Usage:
    python test_qwen_coder.py --n_samples 20 --n_completions 5 --api_base http://localhost:8000/v1
"""

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# Ensure CODER1_EXEC is set for the reward function
os.environ.setdefault("CODER1_EXEC", "basic")

# Import the reward function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verl.utils.reward_score.coder1 import compute_score


def load_dataset(path: str, n_samples: int = 20) -> pd.DataFrame:
    """Load dataset from parquet file."""
    print(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    print(f"Total dataset size: {len(df)} samples")

    # Sample randomly if needed
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
        print(f"Using {n_samples} random samples")

    return df


def prepare_messages(prompt_array: np.ndarray) -> List[Dict[str, str]]:
    """Convert numpy array of prompt dicts to list of dicts."""
    if isinstance(prompt_array, np.ndarray):
        return prompt_array.tolist()
    return prompt_array


async def generate_completions(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    n_completions: int,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> List[str]:
    """Generate completions using OpenAI API."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            n=n_completions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error generating completion: {e}")
        return []


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        pass@k value
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_sample(
    sample: pd.Series,
    completions: List[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single sample with multiple completions."""
    results = {
        'completions': completions,
        'scores': [],
        'logs': [],
        'data_source': sample.get('data_source', 'unknown'),
    }

    # Extract ground truth from reward_model field
    reward_model = sample['reward_model']
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get('ground_truth', '')
    else:
        ground_truth = reward_model

    # Prepare extra_info for the reward function
    # The reward function expects extra_info to have a 'prompt' field
    extra_info = {
        'prompt': '\n'.join([msg.get('content', '') for msg in sample['prompt']]),
        'dataset': sample.get('extra_info', {}).get('dataset', ''),
        'index': sample.get('extra_info', {}).get('index', ''),
    }

    # Evaluate each completion
    for completion in completions:
        try:
            # Redirect stdout to capture the reward log
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            score = compute_score(
                solution_str=completion,
                ground_truth=ground_truth,
                extra_info=extra_info,
                format_reward=0.1,
                answer_reward=1.0,
            )

            log = captured_output.getvalue()
            sys.stdout = old_stdout

            results['scores'].append(score)
            results['logs'].append(log)

        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error evaluating completion: {e}")
            results['scores'].append(-1.1)
            results['logs'].append(f"Error: {str(e)}")

    results['max_score'] = max(results['scores']) if results['scores'] else -1.1
    results['passed'] = any(s >= 1.0 for s in results['scores'])

    return results


def print_sample_output(idx: int, sample: pd.Series, result: Dict[str, Any]):
    """Print a detailed sample output."""
    print("\n" + "="*80)
    print(f"SAMPLE {idx} - Data Source: {result['data_source']}")
    print("="*80)

    # Print full prompt (not truncated)
    messages = sample['prompt']
    print("\nPROMPT:")
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        print(f"[{role}]: {content}\n")

    # Print best completion (not truncated)
    best_idx = np.argmax(result['scores'])
    print(f"\nBEST COMPLETION (Score: {result['scores'][best_idx]:.2f}):")
    completion = result['completions'][best_idx]
    print(completion)

    # Print reward log
    print("\nREWARD LOG:")
    print(result['logs'][best_idx])
    print("="*80 + "\n")


async def main_async(args):

    print("="*80)
    print("Qwen2.5-Coder-7B-Instruct Evaluation")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Completions per prompt: {args.n_completions}")
    print(f"API: {args.api_base}")
    print(f"Model: {args.model}")
    print("="*80 + "\n")

    # Load dataset
    df = load_dataset(args.dataset_path, args.n_samples)

    # Initialize AsyncOpenAI client for vLLM
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    # Test API connection
    print("Testing API connection...")
    try:
        test_messages = [{"role": "user", "content": "Hello"}]
        response = await client.chat.completions.create(
            model=args.model,
            messages=test_messages,
            max_tokens=10,
        )
        print("✓ API connection successful\n")
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        print("\nTo start vLLM server, run:")
        print(f"vllm serve {os.path.expandvars('$HFH/models/Qwen2_5-Coder-7B-Instruct')} \\")
        print("  --port 8000 \\")
        print("  --api-key api_key \\")
        print("  --tensor-parallel-size 1")
        return

    # Prepare all samples for parallel generation
    print(f"Generating completions for {len(df)} samples in parallel...")

    async def process_sample(idx_sample):
        idx, sample = idx_sample
        # Prepare messages
        messages = prepare_messages(sample['prompt'])

        # Generate completions
        completions = await generate_completions(
            client=client,
            messages=messages,
            model=args.model,
            n_completions=args.n_completions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        if not completions:
            print(f"Warning: No completions generated for sample {idx}")
            return None

        return (idx, sample, completions)

    # Generate all completions in parallel
    tasks = [process_sample((idx, sample)) for idx, (_, sample) in enumerate(df.iterrows())]
    generation_results = []

    # Use tqdm with asyncio.gather
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        result = await coro
        if result is not None:
            generation_results.append(result)

    # Now evaluate all results (this is CPU-bound so keep it sequential)
    print(f"\nEvaluating {len(generation_results)} samples...")
    results = []
    all_results = []

    for idx, sample, completions in generation_results:
        # Evaluate completions
        result = evaluate_sample(sample, completions, verbose=False)
        all_results.append(result)
        results.append((idx, sample, result))

    # Print sample outputs
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS")
    print("="*80)
    n_to_print = min(args.n_sample_outputs, len(results))
    for i in range(n_to_print):
        idx, sample, result = results[i]
        print_sample_output(idx, sample, result)

    # Calculate statistics
    print("\n" + "="*80)
    print("EVALUATION STATISTICS")
    print("="*80 + "\n")

    # Overall statistics
    all_scores = [r['max_score'] for r in all_results]
    all_passed = [r['passed'] for r in all_results]

    # Flatten all scores for average across all completions
    all_individual_scores = []
    for r in all_results:
        all_individual_scores.extend(r['scores'])

    print(f"Total samples evaluated: {len(all_results)}")
    print(f"Total completions generated: {len(all_individual_scores)}")
    print(f"\nPer-sample statistics (best of N completions):")
    print(f"  Average max score: {np.mean(all_scores):.3f} ± {np.std(all_scores):.3f}")
    print(f"  Pass rate (score >= 1.0): {np.mean(all_passed):.1%}")
    print(f"  At least 1 correct: {np.mean(all_passed):.1%}")
    print(f"\nPer-completion statistics (all completions):")
    print(f"  Average score: {np.mean(all_individual_scores):.3f} ± {np.std(all_individual_scores):.3f}")
    print(f"  Pass rate (score >= 1.0): {np.mean([s >= 1.0 for s in all_individual_scores]):.1%}")

    # Calculate pass@k
    for k in [1, 5, 10]:
        if k <= args.n_completions:
            n = args.n_completions
            pass_at_k_values = []
            for r in all_results:
                c = sum(1 for s in r['scores'] if s >= 1.0)
                pass_at_k_values.append(calculate_pass_at_k(n, c, k))
            print(f"Pass@{k}: {np.mean(pass_at_k_values):.1%}")

    # Per-source breakdown
    print("\n" + "-"*80)
    print("PER-SOURCE BREAKDOWN")
    print("-"*80 + "\n")

    by_source = defaultdict(list)
    for r in all_results:
        by_source[r['data_source']].append(r)

    for source in sorted(by_source.keys()):
        source_results = by_source[source]
        source_scores = [r['max_score'] for r in source_results]
        source_passed = [r['passed'] for r in source_results]

        # Get all individual scores for this source
        source_individual_scores = []
        for r in source_results:
            source_individual_scores.extend(r['scores'])

        print(f"{source}:")
        print(f"  Samples: {len(source_results)}")
        print(f"  Per-sample (best of N):")
        print(f"    Average max score: {np.mean(source_scores):.3f} ± {np.std(source_scores):.3f}")
        print(f"    Pass rate: {np.mean(source_passed):.1%}")
        print(f"    At least 1 correct: {np.mean(source_passed):.1%}")
        print(f"  Per-completion (all completions):")
        print(f"    Average score: {np.mean(source_individual_scores):.3f} ± {np.std(source_individual_scores):.3f}")
        print(f"    Pass rate: {np.mean([s >= 1.0 for s in source_individual_scores]):.1%}")

        # Calculate pass@k per source
        for k in [1, 5, 10]:
            if k <= args.n_completions:
                n = args.n_completions
                pass_at_k_values = []
                for r in source_results:
                    c = sum(1 for s in r['scores'] if s >= 1.0)
                    pass_at_k_values.append(calculate_pass_at_k(n, c, k))
                print(f"  Pass@{k}: {np.mean(pass_at_k_values):.1%}")
        print()

    print("="*80)
    print("Evaluation complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-Coder-7B-Instruct on code dataset")
    parser.add_argument("--dataset_path", type=str,
                       default=os.path.expandvars("$HFH/data/code-r1-12k-leetcode2k-taco/test.parquet"),
                       help="Path to dataset parquet file")
    parser.add_argument("--n_samples", type=int, default=20,
                       help="Number of samples to test (default: 20)")
    parser.add_argument("--n_completions", type=int, default=5,
                       help="Number of completions per prompt (default: 5)")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1",
                       help="vLLM API base URL")
    parser.add_argument("--api_key", type=str, default="api_key",
                       help="API key for vLLM")
    parser.add_argument("--model", type=str, default="/work/nvme/bfov/models/Qwen2_5-Coder-7B-Instruct",
                       help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Maximum tokens to generate")
    parser.add_argument("--n_sample_outputs", type=int, default=3,
                       help="Number of sample outputs to print")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
