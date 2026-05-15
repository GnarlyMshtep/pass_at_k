#!/usr/bin/env python3
"""Preprocess GSM8K dataset to parquet format for verl RL training."""

import argparse
import os
import random
from typing import Any

from datasets import load_dataset


def extract_ground_truth(answer_str: str) -> str:
    """Extract the numeric answer after #### in GSM8K answer field."""
    marker = "####"
    idx = answer_str.rfind(marker)
    if idx == -1:
        raise ValueError(f"No #### found in answer: {answer_str[:100]}")
    return answer_str[idx + len(marker):].strip().replace(",", "")


SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "
    "who thinks step by step inside <think></think> and provides your final "
    "answer after #### on a new line."
)


def make_map_fn(split: str):
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        question = example["question"]
        raw_answer = example["answer"]
        ground_truth = extract_ground_truth(raw_answer)

        return {
            "data_source": "gsm8k",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": ground_truth,
                "question": question,
                "full_solution": raw_answer,
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K to parquet")
    parser.add_argument("--local_dir", default="claude_data/gsm8k")
    parser.add_argument("--ntrain", type=int, default=7473)
    parser.add_argument("--nval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_examples", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_full = dataset["train"]
    test_full = dataset["test"]
    print(f"Train: {len(train_full)}, Test: {len(test_full)}")

    # Use train split for training, sample from test for val
    train_dataset = train_full.shuffle(seed=args.seed).select(range(min(args.ntrain, len(train_full))))
    val_dataset = test_full.shuffle(seed=args.seed).select(range(min(args.nval, len(test_full))))

    print("Processing training data...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    print("Processing validation data...")
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    if args.print_examples:
        print("\n" + "=" * 80)
        print("EXAMPLE FROM TRAIN SET:")
        print("=" * 80)
        ex = train_dataset[0]
        for msg in ex["prompt"]:
            print(f"[{msg['role']}]: {msg['content']}")
        print(f"\nGround truth: {ex['reward_model']['ground_truth']}")
        print(f"Full solution: {ex['extra_info']['full_solution']}")
        print("=" * 80)

    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    print(f"Saved {len(train_dataset)} train + {len(val_dataset)} val to {args.local_dir}/")


if __name__ == "__main__":
    main()
