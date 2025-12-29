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

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


def generate_synthetic_data(num_samples, range_val):
    """Generate synthetic random number guessing data"""
    data = []
    for i in range(num_samples):
        # Generate random ground truth answer
        ground_truth = random.randint(1, range_val)
        answer_raw = f"{ground_truth}"
        
        question_raw = f"Question: I have selected a random number in 1,2, ..., {range_val} -- please guess it."
            
        
        data.append({
            "question": question_raw,
            "answer": answer_raw,
            "ground_truth": ground_truth
        })
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/xfs/home/aiilyas/rl-exploration/data/random_number_mult_att_per_rollout")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--range", default=20, type=int)
    parser.add_argument("--num-attempts", default=10, type=int, help='the number of attempts we should ask the model to make in a single rollout.')
    parser.add_argument("--train_samples", default=10000, type=int, help="Number of training samples (GSM8k train size)")
    parser.add_argument("--test_samples", default=50, type=int, help="Number of test samples (GSM8k test size)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")


    args = parser.parse_args()   
    # Set random seed for reproducibility
    random.seed(args.seed)

    data_source = "synthetic/random_number"

    # Generate synthetic datasets
    train_data = generate_synthetic_data(args.train_samples, args.range)
    test_data = generate_synthetic_data(args.test_samples, args.range)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    instruction_following = (
        f"You have {args.num_attempts} attempts to answer the question above correctly. "
        f"Please output your first attempt enclosed in <attempt-1></attempt-1>, your second attempt enclosed in <attempt-2></attempt-2>, ..., "
        f"and your {args.num_attempts}th attempt enclosed in <attempt-{args.num_attempts}></attempt-{args.num_attempts}>, respectively. "
        f"Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading).\n"
        f"EXAMPLE OUTPUT:\n"
        f" * <attempt-1>12</attempt-1>, <attempt-2>7</attempt-2>, ..., <attempt-{args.num_attempts}>5</attempt-{args.num_attempts}>.\n"
        f" * <attempt-1>8</attempt-1>, <attempt-2>8</attempt-2>, ..., <attempt-{args.num_attempts}>3</attempt-{args.num_attempts}>.\n"
        f""
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

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
                    "question": question_raw,
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
    print(f"Range: 1 to {args.range}")
    print(f"Saved to: {local_dir}")
    
    # Print the first entry from the training dataset
    print("\n" + "="*50)
    print("FIRST TRAINING DATASET ENTRY:")
    print("="*50)
    first_entry = train_dataset[0]
    for key, value in first_entry.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied to HDFS: {hdfs_dir}")