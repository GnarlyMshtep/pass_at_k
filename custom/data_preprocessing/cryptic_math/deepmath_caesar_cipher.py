#!/usr/bin/env python3
"""
Preprocess the DeepMath-103K dataset to parquet format with Caesar cipher thinking instructions
The model is instructed to think in Caesar cipher (shift 22) but answer in plaintext.
"""

import argparse
import os
import random
from typing import Any, Dict, List

import datasets
from datasets import load_dataset


def caesar_cipher(text: str, shift: int = 22) -> str:
    """
    Apply Caesar cipher to text with given shift.
    Preserves case, numbers, and special characters.
    """
    result = []
    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            base = ord('A') if char.isupper() else ord('a')
            # Shift the character and wrap around
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            # Keep non-alphabetic characters unchanged
            result.append(char)
    return ''.join(result)


def create_system_prompt() -> str:
    """Create the system prompt instructing the model to think in Caesar cipher"""

    # Create an example with Caesar cipher thinking
    example_question = "What is 5 + 3?"
    example_thinking_plaintext = "I need to add 5 and 3. 5 plus 3 equals 8."
    example_thinking_encrypted = caesar_cipher(example_thinking_plaintext, shift=22)
    example_answer = "8"

    return f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant who thinks step by step.

IMPORTANT: You must encode your thinking using a Caesar cipher with shift 22. Your final answer must be in plaintext inside \\boxed{{}}.

Example:
Question: {example_question}
Your response format:
<think>{example_thinking_encrypted}</think> \\boxed{{{example_answer}}}

In the <think> tags, write your reasoning encrypted with Caesar cipher (shift 22).
After </think>, provide your final answer in plaintext using \\boxed{{}}."""


def create_user_prefix() -> str:
    """Create the user instruction prefix"""
    return ("Solve the following math problem. Remember: encode your thinking with Caesar cipher (shift 22) in <think></think> tags, "
            "but provide your final answer in plaintext in \\boxed{}.")


def make_map_fn(split: str):
    """Create a mapping function for processing dataset examples"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract the problem and answer from the DeepMath dataset
        # DeepMath-103K has fields: 'question' and 'final_answer'
        problem = example['question']
        answer = example['final_answer']

        # Get additional metadata if available
        topic = example.get('topic', 'unknown')
        difficulty = example.get('difficulty', None)

        user_prefix = create_user_prefix()

        # Create the question
        question = f"{user_prefix}\n\n{problem}"

        result = {
            "data_source": f"deepmath103k/{topic}",
            "prompt": [
                {
                    "role": "system",
                    "content": create_system_prompt(),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": str(answer),
                "question": problem,
                "topic": topic,
                "difficulty": difficulty,
                "cipher_shift": 22,
                "cipher_scope": "thinking_only",
            },
        }

        return result

    return process_fn


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DeepMath-103K dataset with Caesar cipher thinking instructions"
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/deepmath_caesar_cipher",
        help="Local directory to save processed data"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to copy data to (optional)"
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=100000,
        help="Number of training examples"
    )
    parser.add_argument(
        "--nval",
        type=int,
        default=1000,
        help="Number of validation examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    print("Loading DeepMath-103K dataset...")
    try:
        dataset = load_dataset("zwhe99/DeepMath-103K")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        # Use the train split if available
        if 'train' in dataset:
            full_dataset = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            full_dataset = dataset[split_name]
            print(f"Using split '{split_name}' as no 'train' split found")

        # Remove r1_solution columns to reduce dataset size
        columns_to_remove = [col for col in full_dataset.column_names if col.startswith('r1_solution')]
        if columns_to_remove:
            print(f"Removing columns to reduce size: {columns_to_remove}")
            full_dataset = full_dataset.remove_columns(columns_to_remove)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Dataset features: {full_dataset.features}")

    # Show a sample
    print("\nSample data:")
    sample = full_dataset[0]
    for key, value in sample.items():
        print(f"  {key}: {value}")

    # Check if we have enough data
    total_needed = args.ntrain + args.nval
    if len(full_dataset) < total_needed:
        print(f"Warning: Dataset has {len(full_dataset)} examples but {total_needed} requested")
        print(f"Adjusting to use all available data...")
        args.ntrain = min(args.ntrain, len(full_dataset) - args.nval)
        args.nval = min(args.nval, len(full_dataset) - args.ntrain)
        if args.ntrain < 0:
            args.ntrain = 0
        if args.nval < 0:
            args.nval = 0
        print(f"Adjusted: ntrain={args.ntrain}, nval={args.nval}")

    # Shuffle the dataset
    shuffled_dataset = full_dataset.shuffle(seed=args.seed)

    # Create train and validation splits
    train_dataset = shuffled_dataset.select(range(args.ntrain))
    val_dataset = shuffled_dataset.select(range(args.ntrain, args.ntrain + args.nval))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Apply processing function
    print("Processing training data...")
    train_processed = []
    for idx, example in enumerate(train_dataset):
        processed_example = make_map_fn("train")(example, idx)
        train_processed.append(processed_example)

    print("Processing validation data...")
    val_processed = []
    for idx, example in enumerate(val_dataset):
        processed_example = make_map_fn("val")(example, idx)
        val_processed.append(processed_example)

    # Convert back to datasets
    train_dataset = datasets.Dataset.from_list(train_processed)
    val_dataset = datasets.Dataset.from_list(val_processed)

    print(f"Final train dataset size: {len(train_dataset)}")
    print(f"Final validation dataset size: {len(val_dataset)}")

    # Print first 3 questions for verification
    print("\n" + "="*80)
    print("FIRST 3 QUESTIONS:")
    print("="*80)
    for i in range(min(3, len(train_dataset))):
        example = train_dataset[i]
        print(f"\n--- Question {i+1} ---")
        print(f"Index: {example['extra_info']['index']}")
        print(f"Topic: {example['extra_info']['topic']}")
        print(f"Difficulty: {example['extra_info']['difficulty']}")
        print("\nSystem prompt:")
        print(example['prompt'][0]['content'][:500] + "...")
        print("\nUser content:")
        print(example['prompt'][1]['content'])
        print(f"\nGround truth: {example['extra_info']['answer']}")
        print("-" * 40)

    # Demonstrate Caesar cipher encoding
    print("\n" + "="*80)
    print("CAESAR CIPHER DEMONSTRATION (shift 22):")
    print("="*80)
    demo_text = "The answer is 42. I calculated this by adding 40 and 2."
    encrypted = caesar_cipher(demo_text, shift=22)
    print(f"Original:  {demo_text}")
    print(f"Encrypted: {encrypted}")
    print("="*80)

    # Save to parquet
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\nSaving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Saved {len(train_dataset)} training examples to {local_dir}/train.parquet")
    print(f"Saved {len(val_dataset)} validation examples to {local_dir}/test.parquet")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            print(f"Copying to HDFS: {args.hdfs_dir}")
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            print(f"Successfully copied dataset to {args.hdfs_dir}")
        except ImportError:
            print("Warning: verl.utils.hdfs_io not available, skipping HDFS copy")
        except Exception as e:
            print(f"Error copying to HDFS: {e}")


if __name__ == "__main__":
    main()
