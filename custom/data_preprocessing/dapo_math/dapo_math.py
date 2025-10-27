#!/usr/bin/env python3
"""
Preprocess the DAPO-Math-17k-dedup dataset to parquet format with step-by-step thinking instructions
The model is instructed to think step by step inside <think></think> and answer in plaintext.
"""

import argparse
import os
import random
from typing import Any, Dict, List

import datasets
from datasets import load_dataset




def create_system_prompt() -> str:
    """Create the system prompt instructing the model to think step by step"""

    return f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant who thinks step by step inside <think></think> and provides your final answer in plaintext using \\boxed{{}}."""


def create_user_prefix() -> str:
    """Create the user instruction prefix"""
    return ""


def make_map_fn(split: str):
    """Create a mapping function for processing dataset examples"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract the problem from the DAPO-Math dataset
        # The dataset has 'prompt' field which is a list of messages
        # and 'reward_model' field with ground_truth and style

        # Extract problem from prompt field
        if isinstance(example['prompt'], list):
            # Find the user message in the prompt list
            problem = None
            for msg in example['prompt']:
                if isinstance(msg, dict) and msg.get('content'):
                    problem = msg['content']
                    break
            if problem is None:
                # Fallback: use the first message if structure is different
                problem = str(example['prompt'])
        else:
            problem = str(example['prompt'])

        # Clean up the problem text - remove instruction reminders
        lines_to_remove = [
            'Remember to put your answer on its own line after "Answer:".',
            "Remember to put your answer on its own line after 'Answer:'.",
        ]
        for line_to_remove in lines_to_remove:
            if line_to_remove in problem:
                problem = problem.replace(line_to_remove, '')

        # Clean up extra whitespace
        problem = problem.strip()

        # Extract reward model information
        reward_model = example.get('reward_model', {})
        if isinstance(reward_model, dict):
            answer = reward_model.get('ground_truth', '')
            style = reward_model.get('style', 'rule')
        else:
            answer = ''
            style = 'rule'

        user_prefix = create_user_prefix()

        # Create the question
        question = f"{user_prefix}\n\n{problem}" if user_prefix else problem

        result = {
            "data_source": "dapo-math-17k",
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
            "reward_model": {"style": style, "ground_truth": str(answer)},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": str(answer),
                "question": problem,
                "original_style": style,
            },
        }

        return result

    return process_fn


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DAPO-Math-17k-dedup dataset with step-by-step thinking instructions"
    )
    parser.add_argument(
        "--local_dir",
        default=os.path.expandvars("$HF_HOME/data/dapo-math"),
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
        default=1000000,
        help="Number of training examples"
    )
    parser.add_argument(
        "--nval",
        type=int,
        default=50,
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

    print("Loading DAPO-Math-17k-dedup dataset...")
    try:
        dataset = load_dataset("YouJiacheng/DAPO-Math-17k-dedup")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        # Use the train split if available
        if 'train' in dataset:
            full_dataset = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            full_dataset = dataset[split_name]
            print(f"Using split '{split_name}' as no 'train' split found")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Dataset features: {full_dataset.features}")

    # Show a sample
    print("\nSample data:")
    sample = full_dataset[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(str(value)) > 200:
            print(f"  {key}: {str(value)[:200]}...")
        else:
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
        print(f"Original style: {example['extra_info']['original_style']}")
        print("\nSystem prompt:")
        print(example['prompt'][0]['content'][:500] + "...")
        print("\nUser content:")
        user_content = example['prompt'][1]['content']
        if len(user_content) > 500:
            print(user_content[:500] + "...")
        else:
            print(user_content)
        print(f"\nGround truth: {example['extra_info']['answer']}")
        print("-" * 40)


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
