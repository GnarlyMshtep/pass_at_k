"""
Preprocess DAPO-Math from miles JSONL to verl parquet format.
Keeps prompts EXACTLY as miles sees them (no system prompt added, no "Answer:" removal).
No deduplication — matches miles' sampling distribution (1.79M rows, 14,973 unique).
Subsamples ~6400 train + 50 val for 200 training steps at 32 prompts/step.
"""

import argparse
import json
import os
import random
from typing import Any

import datasets


def make_map_fn(split: str):
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        prompt = example["prompt"]
        label = str(example["label"])
        return {
            "data_source": "dapo-math-miles",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": label},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": label,
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Preprocess DAPO-Math for verl (miles-format)")
    parser.add_argument(
        "--input-path",
        default="/shared/matan/code/slime/poc/data/dapo-math-hf/dapo-math.jsonl",
    )
    parser.add_argument(
        "--local-dir",
        default=os.path.expandvars("$HF_HOME/data/dapo_miles_comparison"),
    )
    parser.add_argument("--ntrain", type=int, default=6400)
    parser.add_argument("--nval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading DAPO-Math from {args.input_path}...")
    records: list[dict] = []
    with open(args.input_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} rows")

    random.shuffle(records)
    train_records = records[: args.ntrain]
    val_records = records[args.ntrain : args.ntrain + args.nval]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    train_processed = [make_map_fn("train")(r, i) for i, r in enumerate(train_records)]
    val_processed = [make_map_fn("val")(r, i) for i, r in enumerate(val_records)]

    train_ds = datasets.Dataset.from_list(train_processed)
    val_ds = datasets.Dataset.from_list(val_processed)

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, "train.parquet")
    val_path = os.path.join(args.local_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Saved train ({len(train_ds)}) to {train_path}")
    print(f"Saved val ({len(val_ds)}) to {val_path}")

    sample = train_ds[0]
    print(f"\nSample prompt role: {sample['prompt'][0]['role']}")
    print(f"Sample prompt[:200]: {sample['prompt'][0]['content'][:200]}")
    print(f"Sample ground_truth: {sample['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
