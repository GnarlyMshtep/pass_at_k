"""
Preprocess the Countdown-Tasks-3to4 dataset to parquet format with multi-attempt prompts.

Dataset: Jiayi-Pan/Countdown-Tasks-3to4
Columns:
- nums (list[int])
- target (int)

Adds a --num_attempts option to instruct the model to think ONCE inside
<think></think>, then output up to N final answers inside <attempt-i></attempt-i>
tags (i = 1..N).

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
import numpy as np

from verl.utils.hdfs_io import copy, makedirs


def build_prompt_messages(numbers, target, num_attempts):
    attempt_blocks = [
        f"<attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)
    ]
    # Display format hint: after <attempt-2> until before the last one you can put ...
    if num_attempts > 3:
        display_blocks = attempt_blocks[:2] + ["..."] + attempt_blocks[-1:]
    else:
        display_blocks = attempt_blocks
    attempts_format = ", ".join(display_blocks)
    content = (
        "You will be presented with a numbers-to-target puzzle. First, think carefully inside <think></think>. "
        "In your <think> section, carefully explore and evaluate a wide range of promising and diverse final answer candidates, considering different strategies and possibilities for each attempt. "
        f"Only proceed to submit your attempts when you are confident that you have {num_attempts} high-quality and well-thought-out attempts. "
        f"After the </think>, output all {num_attempts} final answers, each inside its own <attempt-i></attempt-i> tag (i = 1..{num_attempts}). "
        f"Using the numbers {numbers}, create an expression that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /, (, )); each number can only be used once. "
        "Do not use '=' in any expression. Do not include any extra text outside the tags. "
        f"Your output format must be: <think> … </think> followed by {attempts_format}. "
        "For example, put the expression inside the attempt tag like <attempt-1> (1 + 2) / 3 </attempt-1> (no '=' sign). "
        "You will be rewarded if you have at least one correct attempt. So try to make different and high-quality attempts."
    )
    return [{"content": content, "role": "user"}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory to save data, e.g., $HF_HOME/data/countdown_3to4_multi",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        help="Number of training rows to include",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="Number of test rows to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle before splitting train/test",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=3,
        help="Number of attempts to instruct the model (1-8)",
    )

    args = parser.parse_args()

    if args.num_attempts < 1 or args.num_attempts > 8:
        raise ValueError("--num_attempts must be between 1 and 8 inclusive")

    data_source = "Countdown-Tasks-3to4"

    # Load the HF dataset (single 'train' split)
    dataset_dict = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
    base_dataset = dataset_dict["train"]

    filtered = base_dataset

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
                "prompt": build_prompt_messages(numbers, target, args.num_attempts),
                "ability": "math",
                # Provide target and nums to the reward model for rule-based checking
                "reward_model": {"style": "rule", "ground_truth": target, "target": target},
                "extra_info": {
                    "split": split_label,
                    "index": idx,
                    "nums_length": len(numbers),
                    "nums": numbers,
                    "num_attempts": args.num_attempts,
                },
            }
            return new_example

        return _inner

    train_dataset = train_raw.map(function=process_fn("train"), with_indices=True, remove_columns=train_raw.column_names)
    test_dataset = test_raw.map(function=process_fn("test"), with_indices=True, remove_columns=test_raw.column_names)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Compute tokenizer-based prompt length statistics using Qwen2.5-3B-Instruct
    try:
        tokenizer_path = None
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidate = os.path.join(hf_home, "models", "Qwen2.5-3B-Instruct")
            if os.path.isdir(candidate):
                tokenizer_path = candidate
        if tokenizer_path is None:
            tokenizer_path = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        def _prompt_token_lengths(ds):
            lengths = []
            for item in ds:
                messages = item.get("prompt", [])
                try:
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                except Exception:
                    text = "".join(m.get("content", "") for m in messages)
                    input_ids = tokenizer.encode(text)
                lengths.append(len(input_ids))
            return lengths

        lengths = _prompt_token_lengths(train_dataset) + _prompt_token_lengths(test_dataset)
        if len(lengths) > 0:
            arr = np.array(lengths)
            quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            q_vals = np.quantile(arr, quantiles)
            print("Prompt token length stats (Qwen2.5-3B-Instruct tokenizer):")
            print({
                "count": int(arr.size),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "quantiles": {str(int(q*100))+"%": int(v) for q, v in zip(quantiles, q_vals)},
            })
    except Exception as e:
        print(f"Warning: Failed to compute tokenizer stats: {e}")

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


