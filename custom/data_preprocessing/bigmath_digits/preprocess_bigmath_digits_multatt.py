#!/usr/bin/env python3
"""
Preprocess the bigmath_digits dataset to parquet format
"""

import argparse
import os
import random
from typing import Dict, Any

import datasets
from datasets import load_dataset


def create_system_prompt() -> str:
    """Create the system prompt for the assistant"""
    return ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant who thinks step by step inside thinking tags and outputs guesses for the correct answer in attempt tags. You put ALL your thinking inside thinking tags. \n"
            "You put your attempts inside attempt tags. \n"
            "You NEVER put english text inside attempt tags, ONLY NUMBERICAL ANSWERS. \n"
            "So for example, if you decide to answer -2 for your second attempt, output<attempt>-2</attempt> NOT(\\!) <attempt>my second guess is -2</attempt> or something similar. The answer is always a Python float so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR SPECIAL SYMBOLS INSIDE ATTEMPT TAGS.")


def create_user_prefix() -> str:
    """Create the user instruction prefix"""
    return ("You will be presented with a math question and you have 4 attempts to answer it correctly and you MUST think before each answer. So, your answer format must be <think-1></think-1> <attempt-1></attempt-1>, <think-2></think-2> <attempt-2></attempt-2>, <think-3></think-3> <attempt-3></attempt-3>, and <think-4></think-4> <attempt-4></attempt-4>  where in <think-i> you think about the answer provided in <attempt-i>. Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading).")


def make_map_fn(split: str):
    """Create a mapping function for processing dataset examples"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract the problem and answer from the bigmath_digits dataset
        problem = example['problem']
        answer = example['answer']
        source = example.get('source', 'unknown')
        domain = example.get('domain', [])
        solve_rate = example.get('llamaSb_solve_rate', None)
        
        # Create the full question with user prefix
        user_prefix = create_user_prefix()
        question = f"{user_prefix}\n{problem}"
        
        data = {
            "data_source": f"bigmath_digits/{source}",
            "prompt": [
                {
                    "role": "system",
                    "content": create_system_prompt(),
                },
                {
                    "role": "user", 
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": str(answer),
                "question": problem,
                "source": source,
                "domain": domain,
                "llamaSb_solve_rate": solve_rate,
            },
        }
        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Preprocess bigmath_digits dataset to parquet format")
    parser.add_argument("--local_dir", default="../data/bigmath_digits_multatt", 
                       help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None,
                       help="HDFS directory to copy data to (optional)")
    parser.add_argument("--ntrain", type=int, default=10000,
                       help="Number of training examples")
    parser.add_argument("--nval", type=int, default=50,
                       help="Number of validation examples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--filter_domains", nargs="*", default=None,
                       help="List of domains to filter for (e.g., 'Mathematics -> Applied Mathematics')")
    parser.add_argument("--min_solve_rate", type=float, default=None,
                       help="Minimum llamaSb_solve_rate to include")
    parser.add_argument("--max_solve_rate", type=float, default=None,
                       help="Maximum llamaSb_solve_rate to include")
    parser.add_argument("--print_examples", action="store_true",
                       help="Print one example from train and val sets")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("Loading bigmath_digits dataset...")
    try:
        dataset = load_dataset("mehuldamani/big-math-digits")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Use the train split if available, otherwise use the first available split
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
        print(f"  {key}: {value}")
    
    # Apply filters if specified
    filtered_dataset = full_dataset
    
    if args.filter_domains:
        print(f"Filtering for domains: {args.filter_domains}")
        def domain_filter(example):
            if isinstance(example['domain'], list):
                return any(domain in args.filter_domains for domain in example['domain'])
            else:
                return example['domain'] in args.filter_domains
        
        filtered_dataset = filtered_dataset.filter(domain_filter)
        print(f"After domain filtering: {len(filtered_dataset)} examples")
    
    if args.min_solve_rate is not None:
        print(f"Filtering for llamaSb_solve_rate >= {args.min_solve_rate}")
        filtered_dataset = filtered_dataset.filter(
            lambda x: x['llamaSb_solve_rate'] is not None and x['llamaSb_solve_rate'] >= args.min_solve_rate
        )
        print(f"After min solve rate filtering: {len(filtered_dataset)} examples")
    
    if args.max_solve_rate is not None:
        print(f"Filtering for llamaSb_solve_rate <= {args.max_solve_rate}")
        filtered_dataset = filtered_dataset.filter(
            lambda x: x['llamaSb_solve_rate'] is not None and x['llamaSb_solve_rate'] <= args.max_solve_rate
        )
        print(f"After max solve rate filtering: {len(filtered_dataset)} examples")

    # Check if we have enough data
    total_needed = args.ntrain + args.nval
    if len(filtered_dataset) < total_needed:
        print(f"Warning: Dataset has {len(filtered_dataset)} examples but {total_needed} requested")
        print(f"Adjusting to use all available data...")
        args.ntrain = min(args.ntrain, len(filtered_dataset) - args.nval)
        args.nval = min(args.nval, len(filtered_dataset) - args.ntrain)
        if args.ntrain < 0:
            args.ntrain = 0
        if args.nval < 0:
            args.nval = 0
        print(f"Adjusted: ntrain={args.ntrain}, nval={args.nval}")

    # Shuffle and split the dataset
    shuffled_dataset = filtered_dataset.shuffle(seed=args.seed)
    
    # Create train and validation splits
    train_dataset = shuffled_dataset.select(range(args.ntrain))
    val_dataset = shuffled_dataset.select(range(args.ntrain, args.ntrain + args.nval))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Apply processing function
    print("Processing training data...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    print("Processing validation data...")
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    # Print examples if requested
    if args.print_examples:
        print("\n" + "="*80)
        print("EXAMPLE FROM TRAIN SET:")
        print("="*80)
        train_example = train_dataset[0]
        for k, v in train_example.items():
            if k == "prompt":
                print(f"{k}:")
                for i, msg in enumerate(v):
                    print(f"  Message {i+1} ({msg['role']}):")
                    print(f"{msg['content']}")
                    print()
            elif k == "extra_info":
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")
            print()
        
        print("="*80)
        print("EXAMPLE FROM VAL SET:")
        print("="*80)
        val_example = val_dataset[0]
        for k, v in val_example.items():
            if k == "prompt":
                print(f"{k}:")
                for i, msg in enumerate(v):
                    print(f"  Message {i+1} ({msg['role']}):")
                    print(f"{msg['content']}")
                    print()
            elif k == "extra_info":
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")
            print()
        print("="*80)

    print("\nFirst element of processed train_dataset:")
    for k, v in train_dataset[0].items():
        if k == "prompt":
            print(f"{k}:")
            for msg in v:
                print(f"  {msg['role']}: {msg['content']}")
        else:
            print(f"{k}: {v}")
        print()

    # Save to parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
    
    print(f"Saved {len(train_dataset)} training examples to {local_dir}/train.parquet")
    print(f"Saved {len(val_dataset)} validation examples to {local_dir}/val.parquet")

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
