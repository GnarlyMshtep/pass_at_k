#!/usr/bin/env python3
"""
Preprocess the CMU-AIRe/e3-math-easy dataset to parquet format (multi-attempt prompting)

Style mirrors the bigmath_digits preprocessing scripts (multatt variant).
"""

import argparse
import os
from typing import Dict, Any, Optional

from datasets import load_dataset, Dataset


def create_system_prompt() -> str:
    return (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant who thinks step by step inside thinking tags and outputs guesses for the correct answer in attempt tags. You put ALL your thinking inside thinking tags. \n"
        "You put your attempts inside attempt tags. \n"
        "You NEVER put english text inside attempt tags, ONLY NUMBERICAL ANSWERS. \n"
        "So for example, if you decide to answer -2 for your second attempt, output<attempt-2>-2</attempt-2> NOT(\\!) <attempt-2>my second guess is -2</attempt-2> or something similar. The answer is always a Python float so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR SPECIAL SYMBOLS INSIDE ATTEMPT TAGS."
    )


def create_user_prefix() -> str:
    return (
        "You will be presented with a math question and you have 4 attempts to answer it correctly and you MUST think before each answer. "
        "So, your answer format must be <think-1></think-1> <attempt-1></attempt-1>, <think-2></think-2> <attempt-2></attempt-2>, <think-3></think-3> <attempt-3></attempt-3>, and <think-4></think-4> <attempt-4></attempt-4>  where in <think-i> you think about the answer provided in <attempt-i>. Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading)."
    )


def extract_question(example: Dict[str, Any]) -> Optional[str]:
    prompt = example.get("prompt")
    if isinstance(prompt, list) and prompt:
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    for k in ("question", "problem", "prompt_text"):
        if isinstance(example.get(k), str):
            return example[k]
    return None


def extract_answer(example: Dict[str, Any]) -> Optional[str]:
    rm = example.get("reward_model") or {}
    gt = rm.get("ground_truth")
    if gt is not None:
        return str(gt)
    if example.get("answer") is not None:
        return str(example["answer"])
    return None


EXPECTED_SUFFIXES = [
    "Let's think step by step and output the final answer within \\boxed{}.",
    "Let's think step by step and output the final answer within \\boxed{}",
]


def strip_expected_suffix(question: str) -> str:
    q = question.strip()
    for suff in EXPECTED_SUFFIXES:
        if q.endswith(suff):
            return q[: -len(suff)].rstrip()
    raise ValueError(
        "Question missing expected trailing instruction 'Let's think step by step and output the final answer within \\boxed{}.'"
    )


def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        q = extract_question(example) or ""
        # Enforce and remove trailing "Let's think...\\boxed{}" suffix
        q_clean = strip_expected_suffix(q)
        a = extract_answer(example)
        user_content = f"{create_user_prefix()}\n{q_clean}".strip()

        return {
            "data_source": "dmath-e3/easy",
            "prompt": [
                {"role": "system", "content": create_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(a) if a is not None else ""},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": q_clean,
                "dataset": "CMU-AIRe/e3-math-easy",
                "orig_extra_info": example.get("extra_info"),
                "solution": example.get("solution"),
                "reward": example.get("reward"),
                "length": example.get("length"),
                "correct_length": example.get("correct_length"),
                "incorrect_length": example.get("incorrect_length"),
            },
        }

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
    parser = argparse.ArgumentParser(description="Preprocess CMU-AIRe/e3-math-easy to parquet (multi attempt)")
    parser.add_argument("--local_dir", default="../data/dmath_e3_multatt", help="Local directory to save processed data")
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

    train_ds, val_ds = deterministic_train_val_split(full, val_last_k=args.val_last_k)
    print(f"Train size: {len(train_ds)} | Val size (last {args.val_last_k}): {len(val_ds)}")

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
