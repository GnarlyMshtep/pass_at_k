"""
Preprocess the d1shs0ap/countdown-3-4-5-6-7-8-9 dataset to parquet format with level filtering.

Dataset: d1shs0ap/countdown-3-4-5-6-7-8-9
Columns:
- nums (list[int], length in [3..9])
- target (int)

Functionality:
- Accept --min_level and --max_level arguments (3-9). Keep only rows where min_level <= len(nums) <= max_level.
- Process both train and test splits provided by the dataset.
- Optionally limit number of samples via --train_size/--test_size. Defaults: use all filtered.
- Save standardized parquet outputs compatible with other preprocessors in this repo.

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)
"""

import argparse
import os
import multiprocessing as mp
from typing import Iterable

import datasets
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


PROMPT_TEMPLATE = (
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
)


def build_prompt_messages(numbers, target):
    content = PROMPT_TEMPLATE.format(numbers=numbers, target=target)
    return [{"content": content, "role": "user"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory to save data, e.g., $HF_HOME/data/countdown_3to9_by_level",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.min_level < 3 or args.min_level > 9:
        raise ValueError("--min_level must be between 3 and 9 inclusive")
    if args.max_level < 3 or args.max_level > 9:
        raise ValueError("--max_level must be between 3 and 9 inclusive")
    if args.min_level > args.max_level:
        raise ValueError("--min_level must be less than or equal to --max_level")

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
    if args.train_size is not None and args.train_size >= 0:
        if args.train_size > len(train_shuffled):
            raise ValueError(
                f"Requested train_size ({args.train_size}) exceeds available filtered train rows ({len(train_shuffled)})."
            )
        train_raw = (
            train_shuffled if args.train_size == -1 else train_shuffled.select(range(args.train_size))
        )
    else:
        train_raw = train_shuffled

    if args.test_size is not None and args.test_size >= 0:
        if args.test_size > len(test_shuffled):
            raise ValueError(
                f"Requested test_size ({args.test_size}) exceeds available filtered test rows ({len(test_shuffled)})."
            )
        test_raw = (
            test_shuffled if args.test_size == -1 else test_shuffled.select(range(args.test_size))
        )
    else:
        test_raw = test_shuffled

    # === Precompute original action counts using C++ merge search ===
    try:
        from custom.verifiers.countdown import _merge_search as _merge_search_ext
    except Exception as e:
        _merge_search_ext = None
        print(
            f"[warning] Failed to import C++ merge_search extension ({e}). "
            "Precomputation will be very slow without it."
        )

    def _compute_actions_one(args_tuple: tuple[int, list[int] | list[float], int]) -> tuple[int, int]:
        idx, nums, target = args_tuple
        try:
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

    def _precompute_actions(ds: datasets.Dataset, split_name: str) -> list[int]:
        if len(ds) == 0:
            return []
        iterable: Iterable[tuple[int, list[int] | list[float], int]] = (
            (i, ds[i]["nums"], ds[i]["target"]) for i in range(len(ds))
        )
        ctx = mp.get_context(args.mp_start_method)
        print(
            f"[precompute] split={split_name} size={len(ds)} workers={args.num_workers} mp={args.mp_start_method}"
        )
        results: list[tuple[int, int]] = []
        with ctx.Pool(processes=max(1, int(args.num_workers))) as pool:
            for idx, actions in tqdm(
                pool.imap(_compute_actions_one, iterable, chunksize=64), total=len(ds), desc=f"actions:{split_name}"
            ):
                results.append((idx, actions))
        # Order by idx and extract actions list
        results.sort(key=lambda x: x[0])
        return [a for _, a in results]

    train_actions = _precompute_actions(train_raw, "train")
    test_actions = _precompute_actions(test_raw, "test")
    if len(train_actions) == len(train_raw):
        train_raw = train_raw.add_column("original_actions", train_actions)
    if len(test_actions) == len(test_raw):
        test_raw = test_raw.add_column("original_actions", test_actions)

    # Map to target schema
    def process_fn(split_label):
        def _inner(example, idx):
            numbers = example.get("nums", [])
            target = example.get("target", None)
            original_actions = example.get("original_actions", -1)
            new_example = {
                "data_source": data_source,
                "prompt": build_prompt_messages(numbers, target),
                "ability": "math",
                # Provide target to the reward model for rule-based checking
                "reward_model": {"style": "rule", "ground_truth": target},
                "extra_info": {
                    "split": split_label,
                    "index": idx,
                    "nums_length": len(numbers),
                    "min_level": args.min_level,
                    "max_level": args.max_level,
                    "nums": numbers,
                    # Precomputed baseline actions to solve the task
                    "original_actions": int(original_actions),
                },
            }
            return new_example

        return _inner

    train_dataset = train_raw.map(function=process_fn("train"), with_indices=True, remove_columns=train_raw.column_names)
    test_dataset = test_raw.map(function=process_fn("test"), with_indices=True, remove_columns=test_raw.column_names)

    # Report counts
    print(
        f"Filtered counts (level in [{args.min_level}..{args.max_level}]): train={len(train_filtered)}, test={len(test_filtered)}; "
        f"exported: train={len(train_dataset)}, test={len(test_dataset)}"
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save parquet files
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Print a sample data row after creating the datasets
    if len(train_dataset) > 0:
        print("Sample train row:")
        print(train_dataset[0])
    if len(test_dataset) > 0:
        print("Sample test row:")
        print(test_dataset[0])
        
    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


