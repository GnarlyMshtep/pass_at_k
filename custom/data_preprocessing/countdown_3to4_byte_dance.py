# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Countdown-Tasks-3to4 dataset to parquet format.

Dataset: Jiayi-Pan/Countdown-Tasks-3to4
Columns: 
- nums (list[int], length 3 or 4)
- target (int)

Requirements:
- Use only rows where len(nums) == 4
- Split into train and test using requested sizes
- Build prompts using the provided user template

Expected output schema columns (mirrors other preprocessors):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory to save data, e.g., $HF_HOME/data/countdown_3to4",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        help="Number of training rows to include (after filtering to len(nums)==4)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="Number of test rows to include (after filtering to len(nums)==4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle before splitting train/test",
    )

    args = parser.parse_args()

    data_source = "Countdown-Tasks-3to4"

    # Load the HF dataset (single 'train' split), then filter to len(nums)==4
    dataset_dict = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
    base_dataset = dataset_dict["train"]

    def _is_len4(example):
        nums = example.get("nums", None)
        return isinstance(nums, list) and len(nums) == 4

    filtered = base_dataset.filter(_is_len4)

    total = len(filtered)
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("train_size and test_size must be non-negative")
    if args.train_size + args.test_size > total:
        raise ValueError(
            f"Requested train_size ({args.train_size}) + test_size ({args.test_size}) exceeds available filtered rows ({total})."
        )

    # Shuffle deterministically and split
    shuffled = filtered.shuffle(seed=args.seed)
    train_raw = shuffled.select(range(args.train_size))
    test_raw = shuffled.select(range(args.train_size, args.train_size + args.test_size))

    # Map to target schema
    def process_fn(split_label):
        def _inner(example, idx):
            numbers = example.get("nums", [])
            target = example.get("target", None)
            new_example = {
                "data_source": data_source,
                "prompt": build_prompt_messages(numbers, target),
                "ability": "math",
                # Provide target and nums to the reward model for rule-based checking
                "reward_model": {"style": "rule", "ground_truth": target, "target": target},
                "extra_info": {
                    "split": split_label,
                    "index": idx,
                    "nums_length": len(numbers),
                    "nums": numbers,
                },
            }
            return new_example

        return _inner

    train_dataset = train_raw.map(function=process_fn("train"), with_indices=True, remove_columns=train_raw.column_names)
    test_dataset = test_raw.map(function=process_fn("test"), with_indices=True, remove_columns=test_raw.column_names)

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


