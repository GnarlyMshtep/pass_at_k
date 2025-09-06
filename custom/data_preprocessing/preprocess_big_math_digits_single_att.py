# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the big-math-digits dataset to parquet format with single-attempt format
"""

import argparse
import os
import random

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/scratch/m000122/stalaei/huggingface/data/big_math_digits_single_att")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--ntrain", type=int, default=10000)
    parser.add_argument("--nval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-print", type=int, default=1, help="Number of first dataset items to print")

    args = parser.parse_args()

    # Set random seed for reproducibility (same as multi-attempt version)
    random.seed(args.seed)

    instruction_following = 'You will be presented with a math problem and you have 1 attempt to answer it correctly. You MUST think before your answer. So, your answer format must be <think></think><attempt></attempt> where in <think> you think about the question and come up with your attempt. Think carefully about the problem and work through it step by step. The attempt answer provided in <attempt> should be a numerical value. Try justifying your answer to yourself. Think hard about the answer, spending a long time contemplating. Your answer should be a numerical value (integer or decimal).\n\n'

    # Load dataset from HuggingFace
    print("Loading big-math-digits dataset from HuggingFace...")
    dataset = datasets.load_dataset("mehuldamani/big-math-digits", trust_remote_code=True)
    
    train_dataset = dataset["train"]
    # Use test split but rename to val for consistency with taller puzzles
    val_dataset = dataset["test"]
    
    # Limit dataset sizes (same selection as multi-attempt version due to same seed)
    if len(train_dataset) > args.ntrain:
        train_dataset = train_dataset.select(range(args.ntrain))
    if len(val_dataset) > args.nval:
        val_dataset = val_dataset.select(range(args.nval))

    print(f"Using {len(train_dataset)} training examples and {len(val_dataset)} validation examples")

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example["problem"]
            answer = example["answer"]
            
            question = instruction_following + problem
            
            data = {
                "data_source": "mehuldamani/big-math-digits",
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who thinks step by step inside thinking tags and outputs a numerical guess for the correct answer in an attempt tag. You put ALL your thinking inside thinking tags. \nYou put your attempt inside an attempt tag. \nYou ONLY put numerical values inside attempt tags. \nSo for example, if you decide to answer 3.14159, output <attempt>3.14159</attempt> NOT <attempt>my answer is 3.14159</attempt> or something similar. The answer is always a numerical value, so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR EXPLANATIONS INSIDE ATTEMPT TAGS.",
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
                    "source": example.get("source", "unknown"),
                    "domain": example.get("domain", []),
                    "llama8b_solve_rate": example.get("llama8b_solve_rate", None)
                },
            }
            return data

        return process_fn

    # Apply processing function
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    # Print first n elements of datasets
    n_print = args.n_print
    
    if n_print > 0:
        print(f"First {min(n_print, len(train_dataset))} element(s) of train_dataset:")
        for i in range(min(n_print, len(train_dataset))):
            print(f"--- Train Example {i+1} ---")
            for k, v in train_dataset[i].items():
                print(f"{k}: {v}\n")
        
        print(f"First {min(n_print, len(val_dataset))} element(s) of val_dataset:")
        for i in range(min(n_print, len(val_dataset))):
            print(f"--- Val Example {i+1} ---")
            for k, v in val_dataset[i].items():
                print(f"{k}: {v}\n")

    # Save to parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
    
    print(f"Saved {len(train_dataset)} training examples to {local_dir}/train.parquet")
    print(f"Saved {len(val_dataset)} validation examples to {local_dir}/val.parquet")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied dataset to {args.hdfs_dir}")
