#!/usr/bin/env python3
"""Preprocess BioMath dataset (bio-only variant) to parquet format.

Each example contains only the biology MCQ. The full BioMathQuestion is stored
in extra_info so the reward function can grade against the correct answer.

Data sources (downloaded from HuggingFace):
  - Biology: Idavidrein/gpqa (gpqa_main config, domain == "Biology")
  - Math: vvincentt/filtered-math500, EpochAI/otis-mock-aime-24-25
    (math questions are still paired and stored in extra_info, even though not
    shown to the actor — so the same reward config can optionally score math too)
"""

import json
import os
import random
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from custom.data_preprocessing.BioMath._data_loading import load_biomath_pairs
from custom.reward.BioMath.biomath_prompts import BIO_ONLY_PROMPT
from custom.reward.BioMath.biomath_types import BioMathQuestion, MathSource

TEST_SIZE = 100
SAVE_NAME = "biomath_bio_only"


def print_full_example(example: dict[str, Any], label: str) -> None:
    print(f"\n{'='*100}")
    print(label)
    print(f"{'='*100}")
    for k, v in example.items():
        if k == "prompt":
            print(f"{k}:")
            for i, msg in enumerate(v):
                print(f"  Message {i+1} ({msg['role']}):")
                print(f"    Content: {msg['content']}")
                print()
        elif k == "extra_info":
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
                print()
        else:
            print(f"{k}: {v}")
            print()
    print(f"{'='*100}")


def make_verl_example(question: BioMathQuestion, split: str) -> dict[str, Any]:
    prompt_content = BIO_ONLY_PROMPT.format(
        bio_task=question.bio_question.print_question()
    )
    full_prompt = [{"role": "user", "content": prompt_content}]

    extra = asdict(question)
    # MathSource enum → string for JSON serialization
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
    parser.add_argument("--npairs", type=int, default=1000,
                        help="Number of bio/math pairs to create (default: 1000)")
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
    print(f"Loaded {len(pairs)} pairs")

    split_name = "biomath_bio_only"
    processed = []
    for question in pairs:
        try:
            processed.append(make_verl_example(question, split=split_name))
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            traceback.print_exc()
            raise

    print(f"Processed {len(processed)} examples")

    # Train/test split
    shuffled = processed.copy()
    random.shuffle(shuffled)
    if len(shuffled) < TEST_SIZE:
        test_examples = shuffled
        train_examples = []
    else:
        test_examples = shuffled[:TEST_SIZE]
        train_examples = shuffled[TEST_SIZE:]

    print(f"Split: {len(train_examples)} train, {len(test_examples)} test")

    if args.print_examples and processed:
        if test_examples:
            print_full_example(test_examples[0], "FIRST TEST EXAMPLE")
        if train_examples:
            print_full_example(train_examples[0], "FIRST TRAIN EXAMPLE")

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
