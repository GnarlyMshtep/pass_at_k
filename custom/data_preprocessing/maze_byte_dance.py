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
Preprocess the Passk_Training_Maze dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, help="Local directory to save data, e.g., $HF_HOME/data/maze")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--test_subset_size",
        type=int,
        default=None,
        help="If set, also save a shuffled test subset parquet with this many rows.",
    )
    parser.add_argument(
        "--test_subset_seed",
        type=int,
        default=42,
        help="Seed used to shuffle the test set before selecting the subset.",
    )
    parser.add_argument(
        "--test_subset_output",
        type=str,
        default="test_small.parquet",
        help="Filename for the small test parquet saved in local_dir.",
    )

    args = parser.parse_args()

    data_source = "RUC-AIBOX/Passk_Training_Maze"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # The maze dataset already has the correct format with required columns:
    # - data_source (string)
    # - prompt (list)
    # - ability (string) 
    # - reward_model (dict)
    # - extra_info (dict)
    
    # We just need to ensure the data_source field is consistent
    def make_map_fn(split):
        def process_fn(example, idx):
            # Update the data_source to be consistent
            example["data_source"] = data_source
            
            # Add split information to extra_info if not present
            if "extra_info" not in example:
                example["extra_info"] = {}
            example["extra_info"]["split"] = split
            example["extra_info"]["index"] = idx
            
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Optionally create a smaller shuffled test subset parquet
    if args.test_subset_size is not None and args.test_subset_size > 0:
        subset_size = min(args.test_subset_size, len(test_dataset))
        small_test = test_dataset
        if subset_size < len(test_dataset):
            small_test = test_dataset.shuffle(seed=args.test_subset_seed).select(range(subset_size))
        small_path = os.path.join(local_dir, args.test_subset_output)
        small_test.to_parquet(small_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
