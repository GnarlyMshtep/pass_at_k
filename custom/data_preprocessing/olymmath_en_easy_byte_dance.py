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
Preprocess the local OlymMATH-EN-EASY JSONL to parquet format (test split only).

Expected output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


PROMPT_INSTRUCTION = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first engages in an internal reasoning process, akin to a stream of consciousness, "
    "before providing the User with the answer. The reasoning process and answer are enclosed within `<think></think>` "
    "and `<answer></answer>` tags, respectively. For example:\n\n"
    "```\n"
    "<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>\n"
    "```\n\n"
    "The reasoning process includs detailed considerations such as analyzing questions, summarizing relevant findings, "
    "brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps. "
    "During this process, the Assistant uses casual, genuine phrases such as: \"Hmm\", \"Wait\", \"Alternatively\", \"double check\", \"I wonder...\", \"But\", \"rethink\", etc., "
    "to make the reasoning process coherent, clear, and logically sound, effectively simulating human cognitive processes.\n\n"
    "The Assistant shows the reasoning process within `<think></think>` tags, and ONLY return the FINAL ANSWER within `<answer></answer>` tags. "
    "For example: `<answer> \\frac{1}{2} </answer>`.\n\n"
)


def build_prompt_messages(problem: str):
    content = (
        PROMPT_INSTRUCTION
        + f"User: {problem}\nAssistant:\n<think>\n"
    )
    return [{"content": content, "role": "user"}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the OlymMATH-EN-EASY.jsonl, e.g., /users/.../custom/data_preprocessing/OlymMATH-EN-EASY.jsonl",
    )
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory to save data, e.g., $HF_HOME/data/olymmath_en_easy",
    )
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

    data_source = "OlymMATH-EN-EASY"

    # Load jsonl as a HF dataset under a test split
    dataset_dict = datasets.load_dataset("json", data_files={"test": args.input_jsonl})
    test_dataset = dataset_dict["test"]

    # Map to target schema
    def process_fn(example, idx):
        problem = example.get("problem", "").strip()
        answer = str(example.get("answer", "")).strip()
        subject = example.get("subject", None)
        uid = example.get("unique_id", None)

        new_example = {
            "data_source": data_source,
            "prompt": build_prompt_messages(problem),
            "ability": "math",
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {"split": "test", "index": idx, "subject": subject, "unique_id": uid},
        }
        return new_example

    test_dataset = test_dataset.map(function=process_fn, with_indices=True, remove_columns=test_dataset.column_names)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save full test parquet
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


