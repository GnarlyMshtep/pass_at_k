#!/usr/bin/env python3
"""
Preprocess the CMU-AIRe/e3-math-easy dataset to parquet format (single-answer boxed prompting)

Style mirrors the bigmath_digits preprocessing scripts, but requires the final answer be
wrapped in \\boxed{...} instead of using attempt tags.
"""

import argparse
import os
from typing import Dict, Any, Optional

from datasets import load_dataset, Dataset


def create_system_prompt() -> str:
    """Create the system prompt for the assistant"""
    return (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant who thinks step by step inside <think> tags and then outputs exactly one final answer wrapped in \\boxed{...}.\n"
        "Put ALL your reasoning inside <think>...</think>. Do not include any reasoning or extra text outside <think> except for the single final \\boxed{...} answer.\n"
        "Do not add any trailing commentary after the boxed answer."
    )


def create_user_prefix() -> str:
    """Create the user instruction prefix for a single final boxed answer"""
    return (
        "You will be presented with a math question. First, think step by step inside <think> tags. "
        "Then provide exactly one final answer in the format \\boxed{...}."
    )


def extract_question(example: Dict[str, Any]) -> Optional[str]:
    """Extract the question text from the dataset example.

    The e3-math-easy dataset stores a `prompt` field as a list of chat messages. We use the first
    user message's content. Fallbacks included for robustness.
    """
    prompt = example.get("prompt")
    if isinstance(prompt, list) and prompt:
        # find first user message
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    # Fallback keys sometimes used in other datasets
    for k in ("question", "problem", "prompt_text"):
        if isinstance(example.get(k), str):
            return example[k]
    return None


def extract_answer(example: Dict[str, Any]) -> Optional[str]:
    """Extract ground-truth answer string from e3 record."""
    rm = example.get("reward_model") or {}
    gt = rm.get("ground_truth")
    if gt is not None:
        return str(gt)
    # Fallbacks
    if example.get("answer") is not None:
        return str(example["answer"])
    if example.get("solution") is not None:
        # Some rows include a worked solution; do not use this as ground truth unless needed
        # Keep as None to avoid polluting reward_model
        pass
    return None


EXPECTED_SUFFIXES = [
    "Let's think step by step and output the final answer within \\boxed{}.",
    # Occasionally the period may be missing; include a tolerant variant too
    "Let's think step by step and output the final answer within \\boxed{}",
]


def strip_expected_suffix(question: str) -> str:
    """Ensure the question ends with the required suffix and strip it.

    - Trims surrounding whitespace first.
    - Accepts a few punctuation variants.
    - Raises ValueError if the suffix is not found at the end.
    """
    q = question.strip()
    for suff in EXPECTED_SUFFIXES:
        if q.endswith(suff):
            return q[: -len(suff)].rstrip()
    # No match – raise as requested
    raise ValueError(
        "Question missing expected trailing instruction 'Let's think step by step and output the final answer within \\boxed{}.'"
    )


def make_map_fn(split: str):
    """Create a mapping function for processing dataset examples"""

    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question_from_ds = extract_question(example) or ""
        # Enforce and remove the trailing "Let's think...\\boxed{}" instruction
        question_clean = strip_expected_suffix(question_from_ds)
        answer_from_ds = extract_answer(example)

        # Build our user message with our prefix
        user_prefix = create_user_prefix()
        user_content = f"{user_prefix}\n{question_clean}".strip()

        # Build output schema in our training format
        data = {
            "data_source": "dmath-e3/easy",
            "prompt": [
                {"role": "system", "content": create_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            "ability": "math",
            # Duplicate ground truth at top-level for downstream convenience
            "gts": str(answer_from_ds) if answer_from_ds is not None else "",
            "reward_model": {"style": "rule", "ground_truth": str(answer_from_ds) if answer_from_ds is not None else ""},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question_clean,
                "dataset": "CMU-AIRe/e3-math-easy",
                # Preserve selected original metadata if present
                "orig_extra_info": example.get("extra_info"),
                "solution": example.get("solution"),
                "reward": example.get("reward"),
                "length": example.get("length"),
                "correct_length": example.get("correct_length"),
                "incorrect_length": example.get("incorrect_length"),
            },
        }
        return data

    return process_fn


def deterministic_train_val_split(ds: Dataset, val_last_k: int = 50):
    n = len(ds)
    k = min(val_last_k, n)
    if k == 0:
        return ds, ds.select([])
    train = ds.select(range(0, n - k))
    val = ds.select(range(n - k, n))
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Preprocess CMU-AIRe/e3-math-easy to parquet (single attempt)")
    parser.add_argument(
        "--local_dir",
        default="../data/dmath_e3_singatt",
        help="Local directory to save processed data",
    )
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy data to (optional)")
    parser.add_argument("--val_last_k", type=int, default=50, help="Validation size: use last K rows from train")
    parser.add_argument("--print_examples", action="store_true", help="Print examples from train/val sets")

    args = parser.parse_args()

    print("Loading CMU-AIRe/e3-math-easy dataset...")
    ds_dict = load_dataset("CMU-AIRe/e3-math-easy")
    if "train" not in ds_dict:
        raise RuntimeError("Expected only a 'train' split in e3-math-easy")
    full = ds_dict["train"]
    print(f"Full dataset size: {len(full)}")

    # Deterministic split: last K for validation
    train_ds, val_ds = deterministic_train_val_split(full, val_last_k=args.val_last_k)
    print(f"Train size: {len(train_ds)} | Val size (last {args.val_last_k}): {len(val_ds)}")

    # Map to our schema
    print("Processing train set...")
    train_out = train_ds.map(function=make_map_fn("train"), with_indices=True)
    print("Processing val set...")
    val_out = val_ds.map(function=make_map_fn("val"), with_indices=True)

    if args.print_examples and len(train_out) > 0 and len(val_out) > 0:
        print("\n=== Example from TRAIN ===")
        ex = train_out[0]
        for k, v in ex.items():
            if k == "prompt":
                print("prompt:")
                for i, m in enumerate(v):
                    print(f"  [{i}] {m['role']}: {m['content']}")
            else:
                print(f"{k}: {v}")
        print("\n=== Example from VAL ===")
        ex = val_out[0]
        for k, v in ex.items():
            if k == "prompt":
                print("prompt:")
                for i, m in enumerate(v):
                    print(f"  [{i}] {m['role']}: {m['content']}")
            else:
                print(f"{k}: {v}")

    # Save to parquet
    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, "train.parquet")
    val_path = os.path.join(args.local_dir, "test.parquet")
    print(f"Saving train to {train_path}")
    train_out.to_parquet(train_path)
    print(f"Saving val to {val_path}")
    val_out.to_parquet(val_path)
    print(f"Saved {len(train_out)} train and {len(val_out)} val examples.")

    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            print(f"Copying to HDFS: {args.hdfs_dir}")
            makedirs(args.hdfs_dir)
            copy(src=args.local_dir, dst=args.hdfs_dir)
            print("Copied dataset to HDFS.")
        except ImportError:
            print("Warning: verl.utils.hdfs_io not available, skipping HDFS copy")
        except Exception as e:
            print(f"Error copying to HDFS: {e}")


if __name__ == "__main__":
    main()
