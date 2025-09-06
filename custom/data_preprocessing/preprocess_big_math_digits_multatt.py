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
Preprocess the big-math-digits dataset to parquet format with multi-attempt format
"""

import argparse
import os
import random

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/scratch/m000122/stalaei/huggingface/data/big_math_digits_multatt")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--ntrain", type=int, default=10000)
    parser.add_argument("--nval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-print", type=int, default=1, help="Number of first dataset items to print")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    instruction_following = 'You will be presented with a math problem and you have 3 attempts to answer it correctly. You MUST think before each answer. So, your answer format must be <think-1></think-1><attempt-1></attempt-1><think-2></think-2><attempt-2></attempt-2><think-3></think-3><attempt-3></attempt-3> where in <think-i> you think about the question and come up with your attempt. Generate your thinking for each attempt and consider why previous attempts might be wrong to generate your next attempts. Make sure to think carefully about each attempt. The attempt answers provided in <attempt-1>, <attempt-2>, and <attempt-3> should be numerical values. Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading). Try justifying your answer to yourself. Think hard about the answer, spending a long time contemplating. Your answer should be a numerical value (integer or decimal).\n\n'

    # Load dataset from HuggingFace
    print("Loading big-math-digits dataset from HuggingFace...")
    dataset = datasets.load_dataset("mehuldamani/big-math-digits", trust_remote_code=True)
    
    train_dataset = dataset["train"]
    # Use test split but rename to val for consistency with taller puzzles
    val_dataset = dataset["test"]
    
    # Limit dataset sizes
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
                        "content": "You are a helpful assistant who thinks step by step inside thinking tags and outputs numerical guesses for the correct answer in attempt tags. You put ALL your thinking inside thinking tags. \nYou put your attempts inside attempt tags. \nYou ONLY put numerical values inside attempt tags. \nSo for example, if you decide to answer 3.14159 for your second attempt, output <attempt-2>3.14159</attempt-2> NOT <attempt-2>my second guess is 3.14159</attempt-2> or something similar. The answer is always a numerical value, so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR EXPLANATIONS INSIDE ATTEMPT TAGS.",
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
