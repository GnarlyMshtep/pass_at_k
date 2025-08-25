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
Generate a synthetic random number guessing dataset in parquet format
"""

import argparse
import os
import random
import re

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


def biased_random(range_val):
    if random.randint(0,1) == 0: 
        return random.randint(1,range_val//4 + 1)
    else: 
        return random.randint(1,range_val)

def extract_solution(solution_str):
    """Extract the final numerical answer from GSM8K solution string"""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def generate_synthetic_data(num_samples, range_val, num_attempts):
    """Generate synthetic random number guessing data"""
    data = []
    for i in range(num_samples):
        # Generate random ground truth answer
        ground_truth = random.randint(1, range_val)
        answer_raw = "".join([f"<attempt-{i+1}>{biased_random(range_val)}<attempt-{i+1}> " for i in range(num_attempts)]) + "."
        
        question_raw = f"Question: I have selected a random number in 1,2, ..., {range_val} -- please guess it."
            
        
        data.append({
            "question": question_raw,
            "answer": answer_raw,
            "ground_truth": str(ground_truth),
            "data_type": "random_number"
        })
    
    return data

def generate_gsm8k_data(num_samples, num_attempts, split_name):
    """Generate GSM8K data with correct answer first, then biased random attempts"""
    # Load GSM8K dataset
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    gsm8k_data = dataset[split_name]
    
    # Sample the required number of examples
    if num_samples > len(gsm8k_data):
        # If we need more samples than available, cycle through the dataset
        indices = [i % len(gsm8k_data) for i in range(num_samples)]
    else:
        indices = random.sample(range(len(gsm8k_data)), num_samples)
    
    data = []
    for idx in indices:
        example = gsm8k_data[idx]
        question_raw = example["question"]
        answer_raw = example["answer"]
        
        # Extract the correct numerical answer
        correct_answer = extract_solution(answer_raw)
        
        # Generate attempts: first is correct, rest are biased random
        attempts = [f"<attempt-1>{correct_answer}</attempt-1>"]
        
        # Generate biased random answers for remaining attempts
        # Use a reasonable range for math problems (e.g., -1000 to 1000)
        for i in range(1, num_attempts):
            random_answer = biased_random(1000) if random.randint(0,1) else -biased_random(1000)
            attempts.append(f"<attempt-{i+1}>{random_answer}</attempt-{i+1}>")
        
        answer_formatted = " ".join(attempts) + "."
        
        data.append({
            "question": question_raw,
            "answer": answer_formatted,
            "ground_truth": correct_answer,
            "data_type": "gsm8k"
        })
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/xfs/home/aiilyas/rl-exploration/data/sft_random_number_mult_att_per_rollout")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--range", default=20, type=int)
    parser.add_argument("--num-attempts", default=10, type=int, help='the number of attempts we should ask the model to make in a single rollout.')
    parser.add_argument("--train_samples", default=5000, type=int, help="Number of training samples per distribution")
    parser.add_argument("--test_samples", default=50, type=int, help="Number of test samples per distribution")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")


    args = parser.parse_args()   
    # Set random seed for reproducibility
    random.seed(args.seed)

    data_source = "mixed/random_number_gsm8k"

    # Generate synthetic datasets (half each distribution)
    train_random_data = generate_synthetic_data(args.train_samples, args.range, num_attempts=args.num_attempts)
    train_gsm8k_data = generate_gsm8k_data(args.train_samples, args.num_attempts, "train")
    
    test_random_data = generate_synthetic_data(args.test_samples, args.range, num_attempts=args.num_attempts)
    test_gsm8k_data = generate_gsm8k_data(args.test_samples, args.num_attempts, "test")
    
    # Combine and shuffle the data
    train_data = train_random_data + train_gsm8k_data
    test_data = test_random_data + test_gsm8k_data
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    instruction_following = (
        f"You have {args.num_attempts} attempts to answer the question above correctly. "
        f"Please output your first attempt enclosed in <attempt-1></attempt-1>, your second attempt enclosed in <attempt-2></attempt-2>, ..., "
        f"and your {args.num_attempts}th attempt enclosed in <attempt-{args.num_attempts}></attempt-{args.num_attempts}>, respectively. "
        f"Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading).\n"
        # f"EXAMPLE OUTPUT:\n"
        # f" * <attempt-1>12</attempt-1>, <attempt-2>7</attempt-2>, ..., <attempt-{args.num_attempts}>5</attempt-{args.num_attempts}>.\n"
        # f" * <attempt-1>8</attempt-1>, <attempt-2>8</attempt-2>, ..., <attempt-{args.num_attempts}>3</attempt-{args.num_attempts}>.\n"
        f""
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            data_type = example.pop("data_type")

            question = instruction_following + "\n" + question_raw   

            solution = example.pop("answer")
             
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": instruction_following + question_raw,
                    "data_type": data_type,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print(f"Training: {args.train_samples} random number + {args.train_samples} GSM8K samples")
    print(f"Test: {args.test_samples} random number + {args.test_samples} GSM8K samples")
    print(f"Range for random numbers: 1 to {args.range}")
    print(f"Saved to: {local_dir}")
    
    # Print the first entry from the training dataset
    print("\n" + "="*50)
    print("FIRST TRAINING DATASET ENTRY:")
    print("="*50)
    first_entry = train_dataset[0]
    for key, value in first_entry.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")

    # Print summary of data types in the training set
    data_types = [example["extra_info"]["data_type"] for example in train_dataset]
    random_count = data_types.count("random_number")
    gsm8k_count = data_types.count("gsm8k")
    print(f"Training set distribution: {random_count} random number samples, {gsm8k_count} GSM8K samples")
    
    data_types_test = [example["extra_info"]["data_type"] for example in test_dataset]
    random_count_test = data_types_test.count("random_number")
    gsm8k_count_test = data_types_test.count("gsm8k")
    print(f"Test set distribution: {random_count_test} random number samples, {gsm8k_count_test} GSM8K samples")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied to HDFS: {hdfs_dir}")