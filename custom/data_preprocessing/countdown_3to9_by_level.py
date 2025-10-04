"""
Preprocess the d1shs0ap/countdown-3-4-5-6-7-8-9 dataset to parquet format with level filtering.

Dataset: d1shs0ap/countdown-3-4-5-6-7-8-9
Columns:
- nums (list[int], length in [3..9])
- target (int)

Functionality:
- Accept a --level argument (3-9). Keep only rows where len(nums) >= level.
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

import datasets

from verl.utils.hdfs_io import copy, makedirs

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an expression that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. "
    "Don't use = in the expression between the <answer> tags."
    "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. "
    "Think step by step inside <think> tags."
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
        "--level",
        type=int,
        required=True,
        help="Difficulty level 3-9. Keep rows with len(nums) >= level.",
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
        default=10,
        help="Number of test rows to include after filtering (use -1 for all)",
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

    if args.level < 3 or args.level > 9:
        raise ValueError("--level must be between 3 and 9 inclusive")

    data_source = "countdown-3-4-5-6-7-8-9"

    # Load the HF dataset with explicit splits
    dataset_dict = datasets.load_dataset("d1shs0ap/countdown-3-4-5-6-7-8-9")
    train_base = dataset_dict["train"]
    test_base = dataset_dict["test"]

    def _keep_level_or_above(example):
        nums = example.get("nums", None)
        return isinstance(nums, list) and len(nums) >= args.level

    train_filtered = train_base.filter(_keep_level_or_above)
    test_filtered = test_base.filter(_keep_level_or_above)

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

    # Map to target schema
    def process_fn(split_label):
        def _inner(example, idx):
            numbers = example.get("nums", [])
            target = example.get("target", None)
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
                    "level": args.level,
                    "nums": numbers,
                },
            }
            return new_example

        return _inner

    train_dataset = train_raw.map(function=process_fn("train"), with_indices=True, remove_columns=train_raw.column_names)
    test_dataset = test_raw.map(function=process_fn("test"), with_indices=True, remove_columns=test_raw.column_names)

    # Report counts
    print(
        f"Filtered counts (level >= {args.level}): train={len(train_filtered)}, test={len(test_filtered)}; "
        f"exported: train={len(train_dataset)}, test={len(test_dataset)}"
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save parquet files
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


