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
Preprocess the taller puzzles dataset to parquet format
"""

import argparse
import os
import random

import datasets
from taller_dataset_utils import generate_instance_graph, generate_instance_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/scratch/m000122/stalaei/huggingface/data/taller_puzzles_multatt_3_6_no_gts_1_or_max")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--ntrain", type=int, default=10000)
    parser.add_argument("--nval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-print", type=int, default=1, help="Number of first dataset items to print")

    args = parser.parse_args()
  

    # Set random seed for reproducibility
    random.seed(args.seed)

    instruction_following = 'You will be presented with a height comparison puzzle and you have 3 attempts to answer it correctly and you MUST think before each answer. So, your answer format must be <think></think> <attempt-1></attempt-1><attempt-2></attempt-2><attempt-3></attempt-3> where in <think> you think about the question and come up with 3 attempts. Generate your first attempt and consider why it might be wrong to generate your next attempts. Make sure to think about ALL you attempts in <think>. DON\'T think only about the first attempt and then guess the rest. The attempt answers provided in <attempt-1>, <attempt-2>, and <attempt-3>. Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading). Try justifying your answer to yourself. Think hard about the answer, spending a long time contemplating. Your answer should be a comma-separated list of people who could plausibly be the 3rd tallest.\n\n'

    def generate_puzzle():
        """Generate a single taller puzzle instance"""
        while True:
            n_vertices = random.randint(3, 6)
            max_tallest = random.randint(2, 3) # this is a target, not exactly
            
            # Generate puzzle using utils (without visualization)
            graph, third_tallest, split_vertices = generate_instance_graph(n_vertices, max_tallest)
            
            # Discard responses where len(third_tallest) == 1
            if len(third_tallest) == 1 or len(third_tallest) == n_vertices:
                continue
                
            puzzle_text, shuffled_third_tallest = generate_instance_text(graph, third_tallest, split_vertices, generate_viz=False)
            
            # Extract just the sentences (remove graph visualization line)
            sentences = [line for line in puzzle_text.split('\n') if line.strip() and not line.startswith('[Graph')]
            puzzle_sentences = '\n'.join(sentences)
            
            # Create ground truth (comma-separated, no spaces)
            ground_truth = ','.join(sorted(list(shuffled_third_tallest)))
            
            return {
                'puzzle_text': puzzle_sentences,
                'ground_truth': ground_truth,
                'n_vertices': n_vertices,
                'max_tallest': max_tallest,
                'graph_edges': [(u, v) for u in graph.vertices for v in graph.edges[u]],
                'split_vertices': split_vertices
            }

    def make_map_fn(split):
        def process_fn(example, idx):
            puzzle_data = example
            
            question = instruction_following + puzzle_data['puzzle_text']
            
            data = {
                "data_source": "matan/taller_puzzles",
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who thinks step by step inside thinking tags and outputs guesses for the correct answer in attempt tags. You put ALL your thinking inside thinking tags. \nYou put your attempts inside attempt tags. \nYou ONLY put comma-separated lists of people inside attempt tags. \nSo for example, if you decide to answer A,C,F for your second attempt, output <attempt-2>A,C,F</attempt-2> NOT <attempt-2>my second guess is A, C, and F</attempt-2> or something similar. The answer is always a comma-separated list with no spaces, so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR EXPLANATIONS INSIDE ATTEMPT TAGS.",
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": puzzle_data['ground_truth']},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": puzzle_data['ground_truth'],
                    "question": puzzle_data['puzzle_text'],
                    "n_vertices": puzzle_data['n_vertices'],
                    "max_tallest": puzzle_data['max_tallest'],
                    "graph_edges": puzzle_data['graph_edges'],
                    "split_vertices": puzzle_data['split_vertices']
                },
            }
            return data

        return process_fn

    # Generate training data
    print(f"Generating {args.ntrain} training examples...")
    train_data = []
    for i in range(args.ntrain):
        if i % 1000 == 0:
            print(f"Generated {i}/{args.ntrain} training examples")
        train_data.append(generate_puzzle())
    
    # Generate validation data
    print(f"Generating {args.nval} validation examples...")
    val_data = []
    for i in range(args.nval):
        val_data.append(generate_puzzle())

    # Create HuggingFace datasets
    train_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)

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