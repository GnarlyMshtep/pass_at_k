"""
Preprocess the d1shs0ap/countdown-3-4-5-6-7-8-9 dataset to parquet format,
producing both single-attempt and multi-attempt variants in one run.

Notes:
- This is the FULL countdown task (expects full expressions), not partial countdown.
- Multi-attempt supports two prompt styles (both are produced in one run):
  - single_think: one <think> then N attempts (<attempt-i>...)
  - multi_think: per-attempt <think-i> and <attempt-i> blocks (like taller puzzles)

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)

This script produces THREE datasets under --local_dir:
  - single_attempt/train.parquet and test.parquet
  - multi_attempts_<N>_single_think/train.parquet and test.parquet
  - multi_attempts_<N>_multi_think/train.parquet and test.parquet

Optional: --hdfs_dir to mirror the two directories to HDFS.
"""

import argparse
import os
from typing import Any, Dict, List

import datasets


def build_single_attempt_prompt(numbers: List[int], target: int) -> List[Dict[str, str]]:
    """Build messages for single-attempt full countdown task.

    Requires a single final expression inside <answer>...</answer> and thinking in <think>...</think>.
    """
    content = (
        "Using the numbers {numbers}, create an expression that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /, (, )) one or multiple times but each number can only be used once. "
        "Do not use '=' in the expression. "
        "Think step by step inside <think> </think>. "
        "Then return the final expression inside <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. "
        "Only put the final expression inside <answer> tags."
    ).format(numbers=numbers, target=target)
    return [{"content": content, "role": "user"}]


def build_multi_attempt_prompt_single_think(numbers: List[int], target: int, num_attempts: int) -> List[Dict[str, str]]:
    """Build messages for multi-attempt full countdown with single-think style."""
    attempt_blocks = [f"<attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)]
    display_blocks = attempt_blocks[:2] + ["..."] + attempt_blocks[-1:] if num_attempts > 3 else attempt_blocks
    attempts_format = ", ".join(display_blocks)

    content = (
        "You will be presented with a numbers-to-target puzzle. First, think carefully inside <think></think>. "
        "In your <think> section, explore diverse strategies and finalize {num_attempts} high-quality attempts. "
        "After </think>, output all {num_attempts} final expressions, each inside its own <attempt-i></attempt-i> tag (i = 1..{num_attempts}). "
        "Using the numbers {numbers}, create an expression that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /, (, )); each number can only be used once. "
        "Do not use '=' in any expression. Do not include any extra text outside the tags. "
        f"Your output format must be: <think> … </think> followed by {attempts_format}. "
        "Put exactly one full expression in each attempt tag (no partial lists). "
        "You will be rewarded if you have at least one correct attempt, so aim for diverse, high-quality attempts."
    ).format(numbers=numbers, target=target, num_attempts=num_attempts)
    return [{"content": content, "role": "user"}]


def build_multi_attempt_prompt_multi_think(numbers: List[int], target: int, num_attempts: int) -> List[Dict[str, str]]:
    """Build messages for multi-attempt full countdown with multi-think style (per attempt think+attempt)."""
    think_attempt_pairs = ", ".join([f"<think-{i}></think-{i}> <attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)])
    content = (
        f"You will be presented with a numbers-to-target puzzle and you have {num_attempts} attempts. "
        "For each attempt i, first think inside <think-i></think-i>, then give ONE final expression inside <attempt-i></attempt-i>. "
        f"So, your answer format must be {think_attempt_pairs}. "
        "Using the numbers {numbers}, create an expression that equals {target}. "
        "Allowed ops: +, -, *, /, and parentheses. Use every given number exactly once overall (including duplicates). "
        "Do not use '=' in expressions. Do not include extra prose inside attempt tags. "
        "Each attempt should contain exactly one full expression (no partial lists)."
    ).format(numbers=numbers, target=target)
    return [{"content": content, "role": "user"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Base local directory to save outputs; subdirs will be created for single and multi variants",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training examples to export (-1 for all filtered)")
    parser.add_argument("--test_size", type=int, default=-1, help="Number of test examples to export (-1 for all filtered)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_level",
        type=int,
        default=3,
        help="Minimum difficulty level 3-9. Keep rows with len(nums) >= min_level.",
    )
    parser.add_argument(
        "--max_level",
        type=int,
        default=9,
        help="Maximum difficulty level 3-9. Keep rows with len(nums) <= max_level.",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=4,
        help="Number of attempts for the multi-attempt dataset (1-8)",
    )
    return parser.parse_args()


def _select_split(dataset: datasets.Dataset, size: int) -> datasets.Dataset:
    if size is None or size < 0:
        return dataset
    if size > len(dataset):
        raise ValueError(f"Requested size ({size}) exceeds available rows ({len(dataset)}).")
    return dataset.select(range(size))


def map_row_to_output(
    example: Dict[str, Any],
    idx: int,
    split_label: str,
    variant: str,
    num_attempts: int,
    prompt_style: str,
) -> Dict[str, Any]:
    numbers = example.get("nums", [])
    target = example.get("target", None)

    if variant == "single":
        messages = build_single_attempt_prompt(numbers, target)
    else:
        if prompt_style == "single_think":
            messages = build_multi_attempt_prompt_single_think(numbers, target, num_attempts)
        else:
            messages = build_multi_attempt_prompt_multi_think(numbers, target, num_attempts)

    data_source = "countdown-3-4-5-6-7-8-9"
    ability = "math"

    effective_max_allowed = num_attempts if variant == "multi" else 1

    extra_info = {
        "split": split_label,
        "index": idx,
        "nums_length": len(numbers),
        "nums": numbers,
        "num_attempts": num_attempts if variant == "multi" else 1,
        "max_allowed_attempts": effective_max_allowed,
        "prompt_style": prompt_style if variant == "multi" else "single",
    }

    reward_model = {"style": "rule", "ground_truth": target, "target": target}

    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


if __name__ == "__main__":
    args = parse_args()

    if args.min_level < 3 or args.min_level > 9:
        raise ValueError("--min_level must be between 3 and 9 inclusive")
    if args.max_level < 3 or args.max_level > 9:
        raise ValueError("--max_level must be between 3 and 9 inclusive")
    if args.min_level > args.max_level:
        raise ValueError("--min_level must be less than or equal to --max_level")
    if args.num_attempts < 1 or args.num_attempts > 8:
        raise ValueError("--num_attempts must be between 1 and 8 inclusive")

    # Load dataset
    dataset_dict = datasets.load_dataset("d1shs0ap/countdown-3-4-5-6-7-8-9")
    train_base = dataset_dict["train"]
    test_base = dataset_dict["test"]

    # Filter by level
    def _keep_level_in_range(ex):
        nums = ex.get("nums", None)
        return isinstance(nums, list) and args.min_level <= len(nums) <= args.max_level

    train_filtered = train_base.filter(_keep_level_in_range)
    test_filtered = test_base.filter(_keep_level_in_range)

    # Shuffle and optionally downsample
    train_shuffled = train_filtered.shuffle(seed=args.seed)
    test_shuffled = test_filtered.shuffle(seed=args.seed)
    train_raw = _select_split(train_shuffled, args.train_size)
    test_raw = _select_split(test_shuffled, args.test_size)

    # Build three variants using the same data splits
    def make_map_fn(split_label: str, variant: str, prompt_style: str):
        def _inner(example, idx):
            return map_row_to_output(
                example=example,
                idx=idx,
                split_label=split_label,
                variant=variant,
                num_attempts=args.num_attempts,
                prompt_style=prompt_style,
            )
        return _inner

    # Single-attempt
    train_single = train_raw.map(function=make_map_fn("train", "single", "single"), with_indices=True, remove_columns=train_raw.column_names)
    test_single = test_raw.map(function=make_map_fn("test", "single", "single"), with_indices=True, remove_columns=test_raw.column_names)

    # Multi-attempt (single_think)
    train_multi_single = train_raw.map(function=make_map_fn("train", "multi", "single_think"), with_indices=True, remove_columns=train_raw.column_names)
    test_multi_single = test_raw.map(function=make_map_fn("test", "multi", "single_think"), with_indices=True, remove_columns=test_raw.column_names)

    # Multi-attempt (multi_think)
    train_multi_multi = train_raw.map(function=make_map_fn("train", "multi", "multi_think"), with_indices=True, remove_columns=train_raw.column_names)
    test_multi_multi = test_raw.map(function=make_map_fn("test", "multi", "multi_think"), with_indices=True, remove_columns=test_raw.column_names)

    # Prepare output directories
    single_dir = os.path.join(args.local_dir, "single_attempt")
    multi_dir_single = os.path.join(args.local_dir, f"multi_attempts_{args.num_attempts}_single_think")
    multi_dir_multi = os.path.join(args.local_dir, f"multi_attempts_{args.num_attempts}_multi_think")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(multi_dir_single, exist_ok=True)
    os.makedirs(multi_dir_multi, exist_ok=True)

    # Save parquet files
    train_single.to_parquet(os.path.join(single_dir, "train.parquet"))
    test_single.to_parquet(os.path.join(single_dir, "test.parquet"))
    train_multi_single.to_parquet(os.path.join(multi_dir_single, "train.parquet"))
    test_multi_single.to_parquet(os.path.join(multi_dir_single, "test.parquet"))
    train_multi_multi.to_parquet(os.path.join(multi_dir_multi, "train.parquet"))
    test_multi_multi.to_parquet(os.path.join(multi_dir_multi, "test.parquet"))

    # Print samples
    if len(train_single) > 0:
        print("Sample single-attempt train row:")
        print(train_single[0])
    if len(train_multi_single) > 0:
        print("Sample multi-attempt (single_think) train row:")
        print(train_multi_single[0])
    if len(train_multi_multi) > 0:
        print("Sample multi-attempt (multi_think) train row:")
        print(train_multi_multi[0])

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            single_hdfs = os.path.join(args.hdfs_dir, "single_attempt")
            multi_hdfs_single = os.path.join(args.hdfs_dir, f"multi_attempts_{args.num_attempts}_single_think")
            multi_hdfs_multi = os.path.join(args.hdfs_dir, f"multi_attempts_{args.num_attempts}_multi_think")
            makedirs(single_hdfs)
            makedirs(multi_hdfs_single)
            makedirs(multi_hdfs_multi)
            copy(src=single_dir, dst=single_hdfs)
            copy(src=multi_dir_single, dst=multi_hdfs_single)
            copy(src=multi_dir_multi, dst=multi_hdfs_multi)
        except Exception as e:
            print(f"Warning: failed to mirror to HDFS due to: {e}")

    print(
        f"Filtered counts (level in [{args.min_level}..{args.max_level}]): train={len(train_filtered)}, test={len(test_filtered)}; "
        f"exported single: train={len(train_single)}, test={len(test_single)}; "
        f"exported multi_single_think({args.num_attempts}): train={len(train_multi_single)}, test={len(test_multi_single)}; "
        f"exported multi_multi_think({args.num_attempts}): train={len(train_multi_multi)}, test={len(test_multi_multi)}"
    )


