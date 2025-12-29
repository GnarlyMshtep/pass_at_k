"""
Generate synthetic number guessing dataset (single-attempt) where each instance provides
an explicit set of candidate numbers. The task is to guess the secret number from that set.

Functionality:
- Accept --train_size/--test_size to control number of generated samples per split
- Accept --min_number and --max_number as numeric bounds for sampling the candidate set
- Accept --set_size (s) as the number of unique candidates sampled per instance
- Generate prompts listing the candidate set and asking to guess the secret from that set
- Create only single-attempt variant

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)  # includes number_set, bounds, and metadata

This script produces a dataset under --local_dir:
  - single_attempt/train.parquet and test.parquet

Optional: --hdfs_dir to mirror the directory to HDFS.
"""

import argparse
import os
import random
from typing import Any, Dict, List

import datasets


def build_prompt(secret_number: int, number_set: List[int]) -> List[Dict[str, str]]:
    """Build messages for number guessing task from an explicit candidate set."""
    sorted_set_str = ", ".join(str(n) for n in sorted(number_set))
    system_msg = (
        "You are a helpful assistant. You will be given a task to guess a hidden number "
        "from a provided set of candidate numbers. "
        "You will be asked to answer the question a couple of times, and you will win if you guess the number correctly in at least one of your attempts. "
        "Pay attention to your attempt number, and the total number of attempts you have. Therefore, you should diversify your attempts, and devise a specific strategy considering your current attempt number and the total number of attempts you have. "
        "Think step by step between <think> and </think> and then provide your answer inside <answer> YOUR GUESS HERE </answer> tags as a single integer. "
        "Don't output any other text after the </answer>."
    )

    user_msg = (
        "I have chosen a secret number from the following set of numbers (all are distinct): "
        f"{sorted_set_str}. Can you guess which one it is? Please think step by step between <think> and </think> and then provide your final answer inside <answer> YOUR GUESS HERE </answer> tags as a single integer."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Base local directory to save outputs; subdir will be created for single variant",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=1000, help="Number of training examples to generate")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test examples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_number",
        type=int,
        default=1,
        help="Lower bound (inclusive) for the secret numbers",
    )
    parser.add_argument(
        "--max_number",
        type=int,
        default=100,
        help="Upper bound (inclusive) for the secret numbers",
    )
    parser.add_argument(
        "--set_size",
        type=int,
        default=5,
        help="Number of unique candidate numbers per instance",
    )
    parser.add_argument(
        "--secret_selection",
        type=str,
        default="uniform",
        choices=["uniform", "power"],
        help="How to select the secret number from the candidate set: 'uniform' or 'power' (geometric power-law across sorted candidates)",
    )
    parser.add_argument(
        "--power_ratio",
        type=float,
        default=0.7,
        help="Geometric ratio r for power-law weighting when --secret_selection=power. Use 0 < r <= 1; r=1 reduces to uniform",
    )
    parser.add_argument(
        "--weight_order",
        type=str,
        default="asc",
        choices=["asc", "desc", "shuffle"],
        help=(
            "Order to apply geometric weights when --secret_selection=power: "
            "'asc' (smallest→largest, default), 'desc' (largest→smallest), or 'shuffle' (random per instance)"
        ),
    )
    parser.add_argument(
        "--n_print",
        type=int,
        default=1,
        help="Number of first dataset items to print per split for sanity check",
    )
    return parser.parse_args()


def _generate_one_task(
    min_number: int,
    max_number: int,
    set_size: int,
    secret_selection: str,
    power_ratio: float,
    weight_order: str,
) -> Dict[str, Any]:
    """Generate a single number guessing task with an explicit candidate set."""
    candidate_pool = list(range(min_number, max_number + 1))
    number_set = sorted(random.sample(candidate_pool, k=set_size))
    if secret_selection == "uniform":
        secret_number = random.choice(number_set)
    else:
        # power-law (geometric) weighting across the candidate set with configurable order
        sorted_set = sorted(number_set)
        n = len(sorted_set)
        if weight_order == "asc":
            order_indices = list(range(n))
        elif weight_order == "desc":
            order_indices = list(reversed(range(n)))
        else:  # shuffle
            order_indices = list(range(n))
            random.shuffle(order_indices)

        ordered_candidates = [sorted_set[i] for i in order_indices]
        # Apply weights r, r^2, ..., r^n aligned to the chosen order
        weights = [power_ratio ** (i + 1) for i in range(n)]
        secret_number = random.choices(ordered_candidates, weights=weights, k=1)[0]

    return {
        "secret_number": secret_number,
        "min_number": min_number,
        "max_number": max_number,
        "number_set": number_set,
        "selection_method": secret_selection,
        "power_ratio": power_ratio if secret_selection == "power" else None,
        "weight_order": weight_order if secret_selection == "power" else None,
    }


def _generate_split(
    num_samples: int,
    min_number: int,
    max_number: int,
    set_size: int,
    secret_selection: str,
    power_ratio: float,
    weight_order: str,
) -> List[Dict[str, Any]]:
    """Generate a split of number guessing tasks with explicit candidate sets."""
    data: List[Dict[str, Any]] = []
    for _ in range(num_samples):
        task = _generate_one_task(
            min_number=min_number,
            max_number=max_number,
            set_size=set_size,
            secret_selection=secret_selection,
            power_ratio=power_ratio,
            weight_order=weight_order,
        )
        data.append(task)
    return data


def map_row_to_output(
    task_data: Dict[str, Any],
    idx: int,
    split_label: str,
) -> Dict[str, Any]:
    """Map a single task row to the output format."""
    secret_number = task_data["secret_number"]
    min_number = task_data["min_number"]
    max_number = task_data["max_number"]
    number_set = task_data["number_set"]
    
    messages = build_prompt(secret_number, number_set)
    
    data_source = "synthetic/number_guessing"
    ability = "reasoning"
    
    extra_info = {
        "split": split_label,
        "index": idx,
        "secret_number": secret_number,
        "min_number": min_number,
        "max_number": max_number,
        "number_set": number_set,
        "num_attempts": 1,
        "max_allowed_attempts": 1,
        "secret_selection": task_data.get("selection_method", "uniform"),
        "power_ratio": task_data.get("power_ratio", None),
        "weight_order": task_data.get("weight_order", None),
    }
    
    reward_model = {"style": "rule", "ground_truth": secret_number}
    
    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


if __name__ == "__main__":
    args = parse_args()
    
    if args.min_number >= args.max_number:
        raise ValueError("--min_number must be less than --max_number")
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("--train_size/--test_size must be non-negative")
    if args.set_size <= 0:
        raise ValueError("--set_size must be a positive integer")
    range_size = args.max_number - args.min_number + 1
    if args.set_size > range_size:
        raise ValueError("--set_size cannot exceed the size of the range [min_number, max_number]")
    if args.secret_selection == "power":
        if not (0 < args.power_ratio <= 1):
            raise ValueError("--power_ratio must satisfy 0 < r <= 1 when --secret_selection=power")
    
    # Set seed for reproducibility
    random.seed(args.seed)
    
    # Generate base data for both splits
    print(f"Generating {args.train_size} training examples...")
    train_data = _generate_split(
        num_samples=args.train_size,
        min_number=args.min_number,
        max_number=args.max_number,
        set_size=args.set_size,
        secret_selection=args.secret_selection,
        power_ratio=args.power_ratio,
        weight_order=args.weight_order,
    )
    print(f"Generating {args.test_size} test examples...")
    test_data = _generate_split(
        num_samples=args.test_size,
        min_number=args.min_number,
        max_number=args.max_number,
        set_size=args.set_size,
        secret_selection=args.secret_selection,
        power_ratio=args.power_ratio,
        weight_order=args.weight_order,
    )
    
    # Create HF datasets
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)
    
    # Map to single variant using the same base tasks
    def make_map_fn(split_label: str):
        def _inner(example, idx):
            return map_row_to_output(
                task_data=example,
                idx=idx,
                split_label=split_label,
            )
        return _inner
    
    # Single-attempt
    train_single = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)
    test_single = test_dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)
    
    # Prepare output directory
    single_dir = os.path.join(args.local_dir, "single_attempt")
    os.makedirs(single_dir, exist_ok=True)
    
    # Save parquet files
    train_single.to_parquet(os.path.join(single_dir, "train.parquet"))
    test_single.to_parquet(os.path.join(single_dir, "test.parquet"))
    
    # Print samples
    if args.n_print > 0 and len(train_single) > 0:
        print(f"Printing {args.n_print} sample(s) from single-attempt train data:")
        for i in range(min(args.n_print, len(train_single))):
            print(train_single[i])
    
    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            single_hdfs = os.path.join(args.hdfs_dir, "single_attempt")
            makedirs(single_hdfs)
            copy(src=single_dir, dst=single_hdfs)
        except Exception as e:
            print(f"Warning: failed to mirror to HDFS due to: {e}")
    
    print(
        f"Exported single: train={len(train_single)}, test={len(test_single)}"
    )

# Example usage:
# PYTHONPATH=. python custom/data_preprocessing/number_guessing/number_guessing_dataset.py \
#   --local_dir $HF_HOME/data/number_guessing --train_size 1000 --test_size 100 \
#   --min_number 1 --max_number 100 --set_size 5
