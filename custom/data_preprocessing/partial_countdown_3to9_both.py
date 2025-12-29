"""
Preprocess the d1shs0ap/countdown-3-4-5-6-7-8-9 dataset to parquet format with
level filtering, producing both single-attempt and multi-attempt variants
in one efficient run.

Dataset: d1shs0ap/countdown-3-4-5-6-7-8-9
Columns:
- nums (list[int], length in [3..9])
- target (int)

Functionality:
- Accept --min_level and --max_level arguments (3-9). Keep only rows where
  min_level <= len(nums) <= max_level.
- Process both train and test splits provided by the dataset.
- Optionally limit number of samples via --train_size/--test_size. Defaults: use all filtered.
- Efficiently precompute original_actions once and share across both variants.
- Use the same filtered and shuffled data splits for both prompt variants.
- Adds a --num_attempts option for multi-attempt variant.

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)

This script produces TWO datasets under --local_dir:
  - single_attempt/train.parquet and test.parquet
  - multi_attempts_<N>/train.parquet and test.parquet (N = --num_attempts)

Optional: --hdfs_dir to mirror the two directories to HDFS.
"""

import argparse
import os
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Iterable

import datasets
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


def build_single_attempt_prompt(numbers: List[int], target: int) -> List[Dict[str, str]]:
    """Build prompt for single-attempt countdown task."""
    content = (
        "Goal: Using the numbers {numbers}, build an arithmetic expression that equals {target}.\n"
        "Rules: Allowed ops are +, -, *, /, and parentheses. Use every given number exactly once overall (including duplicates). "
        "No concatenation, no extra numbers.\n"
        "Output format: Think step by step inside <think>...</think>. Then, inside <answer>...</answer>, give EITHER "
        "(a) a single full expression, OR (b) a comma-separated list of valid partial sub-expressions that can be merged to reach the target. "
        "Across the item(s), the multiset of numbers must match {numbers}. No '=' signs or prose in <answer>.\n"
        "Preference: Only output a full expression if you have VERIFIED it evaluates to {target} and uses all numbers exactly once. "
        "If not fully certain, DO NOT guess—output mergeable partials instead. Wrong full expressions score 0; valid mergeable partials earn credit by number of successful merges.\n"
        "Examples (for numbers [23, 56, 71, 53, 70, 56] and target 201):\n"
        "<answer>53*56, 23-70, 71-56</answer>\n"
        "<answer>(((53 * 56) - (23 - 70)) / (71 - 56))</answer>\n"
        "<answer>23-70, 56, 56, 53, 71</answer>\n"
        "Note: A list of mergeable partials is preferred over an incorrect full expression.\n"
        "Now, first think carefully inside <think>...</think>, then give your answer inside <answer>...</answer>."
    ).format(numbers=numbers, target=target)
    
    return [{"content": content, "role": "user"}]


def build_multi_attempt_prompt(numbers: List[int], target: int, num_attempts: int) -> List[Dict[str, str]]:
    """Build prompt for multi-attempt countdown task."""
    attempt_blocks = [f"<attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)]
    if num_attempts > 3:
        display_blocks = attempt_blocks[:2] + ["..."] + attempt_blocks[-1:]
    else:
        display_blocks = attempt_blocks
    attempts_format = ", ".join(display_blocks)

    content = (
        "Goal: Using the numbers {numbers}, build an arithmetic expression that equals {target}.\n"
        "Rules: Allowed ops are +, -, *, /, and parentheses. Use every given number exactly once overall (including duplicates). "
        "No concatenation, no extra numbers.\n"
        "Output options: In each attempt, give EITHER (a) a single full expression, OR (b) a comma-separated list of valid partial sub-expressions that can be merged to reach the target. "
        "Across the item(s), the multiset of numbers must match {numbers}. No '=' signs or prose inside attempts.\n"
        "Preference: Only output a full expression if you have VERIFIED it evaluates to {target} and uses all numbers exactly once. "
        "If not fully certain, DO NOT guess—output mergeable partials instead. Wrong full expressions score 0; valid mergeable partials earn credit by number of successful merges.\n"
        "Examples (for numbers [23, 56, 71, 53, 70, 56] and target 201):\n"
        "<attempt>53*56, 23-70, 71-56</attempt>\n"
        "<attempt>(((53 * 56) - (23 - 70)) / (71 - 56))</attempt>\n"
        "<attempt>23-70, 56, 56, 53, 71</attempt>\n"
        "Note: A list of mergeable partials is preferred over an incorrect full expression.\n"
        "Now, first think carefully inside <think></think> (do not reveal calculations outside). After </think>, output all {num_attempts} final attempts, "
        "each inside its own <attempt-i></attempt-i> tag (i = 1..{num_attempts}).\n"
        f"Your output format must be: <think> … </think> followed by {attempts_format}. "
        "Each attempt should contain either a single full expression or a comma-separated list of partials. "
        "You will be rewarded based on the maximum score across attempts. So try to make different and high-quality attempts."
    ).format(numbers=numbers, target=target, num_attempts=num_attempts)

    return [{"content": content, "role": "user"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Base local directory to save outputs; subdirs will be created for single and multi variants",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--min_level",
        type=int,
        required=True,
        help="Minimum difficulty level 3-9. Keep rows with len(nums) >= min_level.",
    )
    parser.add_argument(
        "--max_level",
        type=int,
        required=True,
        help="Maximum difficulty level 3-9. Keep rows with len(nums) <= max_level.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=-1,
        help="Number of training rows to include after filtering (use -1 for all)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=-1,
        help="Number of test rows to include after filtering (use -1 for all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of worker processes for action precomputation",
    )
    parser.add_argument(
        "--mp_start_method",
        type=str,
        default="fork",
        choices=["fork", "forkserver", "spawn"],
        help="Multiprocessing start method for action precomputation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle before downsampling",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=3,
        help="Number of attempts for the multi-attempt dataset (1-8)",
    )
    return parser.parse_args()


def select_split(dataset: datasets.Dataset, size: int) -> datasets.Dataset:
    """Select a subset of the dataset based on size."""
    if size is None or size < 0:
        return dataset
    if size == -1:
        return dataset
    if size > len(dataset):
        raise ValueError(f"Requested size ({size}) exceeds available rows ({len(dataset)}).")
    return dataset.select(range(size))


def _compute_actions_one(args_tuple: tuple[int, list[int] | list[float], int]) -> tuple[int, int]:
    """Compute actions for a single example."""
    idx, nums, target = args_tuple
    try:
        # Try to import C++ extension
        try:
            from custom.verifiers.countdown import _merge_search as _merge_search_ext
        except Exception:
            _merge_search_ext = None
        
        if _merge_search_ext is None:
            # Minimal Python fallback: return -1 actions to indicate missing
            return idx, -1
        can_merge, actions = _merge_search_ext.can_merge_to_target(list(map(float, nums)), float(target), 1e-6)
        if not bool(can_merge):
            # Unsolvable: mark -1; downstream can assert if desired
            return idx, -1
        return idx, int(actions)
    except Exception:
        return idx, -1


def _precompute_actions(ds: datasets.Dataset, split_name: str, num_workers: int, mp_start_method: str) -> list[int]:
    """Precompute original actions for all examples in the dataset."""
    if len(ds) == 0:
        return []
    iterable: Iterable[tuple[int, list[int] | list[float], int]] = (
        (i, ds[i]["nums"], ds[i]["target"]) for i in range(len(ds))
    )
    ctx = mp.get_context(mp_start_method)
    print(
        f"[precompute] split={split_name} size={len(ds)} workers={num_workers} mp={mp_start_method}"
    )
    results: list[tuple[int, int]] = []
    with ctx.Pool(processes=max(1, int(num_workers))) as pool:
        for idx, actions in tqdm(
            pool.imap(_compute_actions_one, iterable, chunksize=64), total=len(ds), desc=f"actions:{split_name}"
        ):
            results.append((idx, actions))
    # Order by idx and extract actions list
    results.sort(key=lambda x: x[0])
    return [a for _, a in results]


def map_row_to_output(
    example: Dict[str, Any],
    idx: int,
    split_label: str,
    variant: str,
    num_attempts: int,
) -> Dict[str, Any]:
    """Map a single row to the output format."""
    numbers = example.get("nums", [])
    target = example.get("target", None)
    original_actions = example.get("original_actions", -1)
    
    if variant == "single":
        messages = build_single_attempt_prompt(numbers, target)
    else:
        messages = build_multi_attempt_prompt(numbers, target, num_attempts)

    data_source = "countdown-3-4-5-6-7-8-9"
    ability = "math"

    # Compute effective max attempts respected by the verifier
    effective_max_allowed = num_attempts if variant == "multi" else 1

    extra_info = {
        "split": split_label,
        "index": idx,
        "nums_length": len(numbers),
        "nums": numbers,
        "original_actions": int(original_actions),
        "num_attempts": num_attempts if variant == "multi" else 1,
        "max_allowed_attempts": effective_max_allowed,
    }

    reward_model = {
        "style": "rule",
        "ground_truth": target,
        "target": target,
    }

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

    data_source = "countdown-3-4-5-6-7-8-9"

    # Load the HF dataset with explicit splits
    dataset_dict = datasets.load_dataset("d1shs0ap/countdown-3-4-5-6-7-8-9")
    train_base = dataset_dict["train"]
    test_base = dataset_dict["test"]

    def _keep_level_in_range(example):
        nums = example.get("nums", None)
        return isinstance(nums, list) and args.min_level <= len(nums) <= args.max_level

    train_filtered = train_base.filter(_keep_level_in_range)
    test_filtered = test_base.filter(_keep_level_in_range)

    # Shuffle deterministically for any downsampling
    train_shuffled = train_filtered.shuffle(seed=args.seed)
    test_shuffled = test_filtered.shuffle(seed=args.seed)

    # Apply optional size limits
    train_raw = select_split(train_shuffled, args.train_size if args.train_size is not None else -1)
    test_raw = select_split(test_shuffled, args.test_size if args.test_size is not None else -1)

    # === Precompute original action counts using C++ merge search ===
    # This is done once and shared across both variants
    print("[precompute] Computing original actions for shared use...")
    train_actions = _precompute_actions(train_raw, "train", args.num_workers, args.mp_start_method)
    test_actions = _precompute_actions(test_raw, "test", args.num_workers, args.mp_start_method)
    
    # Add precomputed actions to the datasets
    if len(train_actions) == len(train_raw):
        train_raw = train_raw.add_column("original_actions", train_actions)
    if len(test_actions) == len(test_raw):
        test_raw = test_raw.add_column("original_actions", test_actions)

    # Build both variants using the same data splits
    def make_map_fn(split_label: str, variant: str):
        def _inner(example, idx):
            return map_row_to_output(
                example=example,
                idx=idx,
                split_label=split_label,
                variant=variant,
                num_attempts=args.num_attempts,
            )
        return _inner

    # Single-attempt
    train_single = train_raw.map(function=make_map_fn("train", "single"), with_indices=True, remove_columns=train_raw.column_names)
    test_single = test_raw.map(function=make_map_fn("test", "single"), with_indices=True, remove_columns=test_raw.column_names)

    # Multi-attempt
    train_multi = train_raw.map(function=make_map_fn("train", "multi"), with_indices=True, remove_columns=train_raw.column_names)
    test_multi = test_raw.map(function=make_map_fn("test", "multi"), with_indices=True, remove_columns=test_raw.column_names)

    # Report counts
    print(
        f"Filtered counts (level in [{args.min_level}..{args.max_level}]): train={len(train_filtered)}, test={len(test_filtered)}; "
        f"exported single: train={len(train_single)}, test={len(test_single)}; "
        f"exported multi({args.num_attempts}): train={len(train_multi)}, test={len(test_multi)}"
    )

    # Prepare output directories
    single_dir = os.path.join(args.local_dir, "single_attempt")
    multi_dir = os.path.join(args.local_dir, f"multi_attempts_{args.num_attempts}")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)

    # Save parquet files
    train_single.to_parquet(os.path.join(single_dir, "train.parquet"))
    test_single.to_parquet(os.path.join(single_dir, "test.parquet"))
    train_multi.to_parquet(os.path.join(multi_dir, "train.parquet"))
    test_multi.to_parquet(os.path.join(multi_dir, "test.parquet"))

    # Print a couple of sample rows
    if len(train_single) > 0:
        print("Sample single-attempt train row:")
        print(train_single[0])
    if len(train_multi) > 0:
        print("Sample multi-attempt train row:")
        print(train_multi[0])

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        single_hdfs = os.path.join(args.hdfs_dir, "single_attempt")
        multi_hdfs = os.path.join(args.hdfs_dir, f"multi_attempts_{args.num_attempts}")
        makedirs(single_hdfs)
        makedirs(multi_hdfs)
        copy(src=single_dir, dst=single_hdfs)
        copy(src=multi_dir, dst=multi_hdfs)

# Example usage:
# PYTHONPATH=. python custom/data_preprocessing/partial_countdown_3to9_both.py --local_dir $HF_HOME/data/partial_countdown_3to9_both --min_level 3 --max_level 9 --test_size 256 --num_attempts 8
