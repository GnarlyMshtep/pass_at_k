#!/usr/bin/env python3
"""Preprocess BioMath dataset (math-only variant) to parquet format.

Each example shows only the math question to the actor.
Full BioMathQuestion stored in extra_info.
"""

import os
import random
import traceback
from dataclasses import asdict
from typing import Any

from custom.data_preprocessing.BioMath._data_loading import load_biomath_pairs
from custom.reward.BioMath.biomath_prompts import MATH_ONLY_PROMPT
from custom.reward.BioMath.biomath_types import BioMathQuestion

TEST_SIZE = 100
SAVE_NAME = "biomath_math_only"


def print_full_example(example: dict[str, Any], label: str) -> None:
    print(f"\n{'='*100}")
    print(label)
    print(f"{'='*100}")
    for k, v in example.items():
        if k == "prompt":
            for i, msg in enumerate(v):
                print(f"  Message {i+1} ({msg['role']}): {msg['content']}")
        else:
            print(f"{k}: {v}")
    print(f"{'='*100}")


def make_verl_example(question: BioMathQuestion, split: str) -> dict[str, Any]:
    prompt_content = MATH_ONLY_PROMPT.format(
        math_task=question.math_question.print_question()
    )
    full_prompt = [{"role": "user", "content": prompt_content}]

    extra = asdict(question)
    extra["math_question"]["source"] = question.math_question.source.value

    return {
        "data_source": "biomath",
        "prompt": full_prompt,
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": extra,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=f"Preprocess BioMath ({SAVE_NAME}) dataset")
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--npairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_examples", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.local_dir is None:
        hf_home = os.getenv("HF_HOME")
        if hf_home is None:
            raise ValueError("HF_HOME not set and --local_dir not specified")
        args.local_dir = os.path.join(hf_home, "data", SAVE_NAME)

    print(f"Loading BioMath pairs (n={args.npairs}, seed={args.seed})...")
    pairs = load_biomath_pairs(n=args.npairs, seed=args.seed)

    processed = []
    for question in pairs:
        try:
            processed.append(make_verl_example(question, split=SAVE_NAME))
        except Exception as e:
            traceback.print_exc()
            raise

    print(f"Processed {len(processed)} examples")

    shuffled = processed.copy()
    random.shuffle(shuffled)
    test_examples = shuffled[:TEST_SIZE] if len(shuffled) >= TEST_SIZE else shuffled
    train_examples = shuffled[TEST_SIZE:] if len(shuffled) >= TEST_SIZE else []

    print(f"Split: {len(train_examples)} train, {len(test_examples)} test")

    if args.print_examples and test_examples:
        print_full_example(test_examples[0], "FIRST TEST EXAMPLE")

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    import datasets
    if test_examples:
        path = os.path.join(local_dir, "test.parquet")
        datasets.Dataset.from_list(test_examples).to_parquet(path)
        print(f"Saved {len(test_examples)} test examples to {path}")
    if train_examples:
        path = os.path.join(local_dir, "train.parquet")
        datasets.Dataset.from_list(train_examples).to_parquet(path)
        print(f"Saved {len(train_examples)} train examples to {path}")

    print(f"\nDone. Output: {local_dir}")


if __name__ == "__main__":
    main()
