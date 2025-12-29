"""
Preprocess the taller puzzles dataset to parquet format with
both single-attempt and multi-attempt variants in one run.

Functionality:
- Accept --train_size/--test_size to control number of generated samples per split
- Accept --max_vertices as an upper bound for n_vertices used when generating puzzles
- Accept --num_attempts for the multi-attempt variant (default: 4)
- Use the same generated base puzzles for both variants to ensure comparability

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)

This script produces TWO datasets under --local_dir:
  - single_attempt/train.parquet and test.parquet
  - multi_attempts_<N>/train.parquet and test.parquet (N = --num_attempts)

Optional: --hdfs_dir to mirror the two directories to HDFS.
"""

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import datasets

from taller_dataset_utils import generate_instance_graph, generate_instance_text


def build_single_attempt_prompt(puzzle_text: str) -> List[Dict[str, str]]:
    """Build messages for single-attempt taller puzzle task."""
    instruction_following = (
        "You will be presented with a height comparison puzzle and you MUST think before you answer. "
        "So, your answer format must be <think></think> <attempt></attempt>, where in <think> you think about the answer provided in <attempt>. "
        "Your answer should be a comma-separated list of people who could plausibly be the 3rd tallest.\n\n"
    )

    system_msg = (
        "You are a helpful assistant who thinks step by step inside thinking tags and outputs guesses for the correct answer in attempt tags. "
        "You put ALL your thinking inside thinking tags.\n"
        "You put your attempts inside attempt tags.\n"
        "You ONLY put comma-separated lists of people inside attempt tags.\n"
        "So for example, if you decide to answer A,C,F for your attempt, output <attempt>A,C,F</attempt> NOT <attempt>my guess is A, C, and F</attempt> or similar. "
        "The answer is always a comma-separated list with no spaces, so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR EXPLANATIONS INSIDE ATTEMPT TAGS."
    )

    content = instruction_following + puzzle_text
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]


def build_multi_attempt_prompt(puzzle_text: str, num_attempts: int) -> List[Dict[str, str]]:
    """Build messages for multi-attempt taller puzzle task."""
    attempt_blocks = [f"<attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)]
    if num_attempts > 3:
        display_blocks = attempt_blocks[:2] + ["..."] + attempt_blocks[-1:]
    else:
        display_blocks = attempt_blocks
    attempts_format = ", ".join(display_blocks)

    instruction_following = (
        f"You will be presented with a height comparison puzzle and you have {num_attempts} attempts to answer it correctly and you MUST think before each answer. "
        f"So, your answer format must be "
        + ", ".join([f"<think-{i}></think-{i}> <attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)])
        + " where in <think-i> you think about the answer provided in <attempt-i>. "
        "Your answers should be comma-separated lists of people who could plausibly be the 3rd tallest. "
        "Before each <attempt-i>, write a substantial and specific <think-i> first, taking a new approach to the problem. "
        "Think carefully and deeply for each attempt, don't rush to output the answer. "
        "Please optimize for getting at least one attempt correct, rather than getting more than one attempt correct (pass@k grading).\n\n"
    )

    system_msg = (
        "You are a helpful assistant who thinks step by step inside thinking tags and outputs guesses for the correct answer in attempt tags. "
        "You put ALL your thinking inside thinking tags.\n"
        "You put your attempts inside attempt tags.\n"
        "You ONLY put comma-separated lists of people inside attempt tags.\n"
        "So for example, if you decide to answer A,C,F for your second attempt, output <attempt-2>A,C,F</attempt-2> NOT <attempt-2>my second guess is A, C, and F</attempt-2> or similar. "
        "The answer is always a comma-separated list with no spaces, so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR EXPLANATIONS INSIDE ATTEMPT TAGS."
    )

    content = instruction_following + puzzle_text
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Base local directory to save outputs; subdirs will be created for single and multi variants",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training examples to generate")
    parser.add_argument("--test_size", type=int, default=256, help="Number of test examples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_vertices",
        type=int,
        default=3,
        help="Lower bound (inclusive) on the number of vertices used when generating puzzles",
    )
    parser.add_argument(
        "--max_vertices",
        type=int,
        default=12,
        help="Upper bound (inclusive) on the number of vertices used when generating puzzles",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=4,
        help="Number of attempts for the multi-attempt dataset (1-8)",
    )
    parser.add_argument(
        "--n_print",
        type=int,
        default=1,
        help="Number of first dataset items to print per split for sanity check",
    )
    return parser.parse_args()


def _generate_one_puzzle(min_vertices: int, max_vertices: int) -> Tuple[Dict[str, Any], int]:
    """Generate a single puzzle with basic quality filters.

    Returns: (puzzle_dict, n_vertices)
    """
    # Sample puzzle hyperparameters
    n_vertices = random.randint(min_vertices, max_vertices)
    # Choose max_tallest (controls plausible third tallest set size) from a reasonable range
    max_tallest = random.randint(2, 4)

    graph, third_tallest, split_vertices = generate_instance_graph(n_vertices, max_tallest)

    # Quality filter: avoid degenerate cases
    if len(third_tallest) == 1 or len(third_tallest) == n_vertices:
        # The caller can re-try if filtered out
        return {}, n_vertices

    puzzle_text, shuffled_third_tallest = generate_instance_text(
        graph, third_tallest, split_vertices, generate_viz=False
    )

    # Extract only puzzle sentences (drop optional viz marker if present)
    sentences = [line for line in puzzle_text.split("\n") if line.strip() and not line.startswith('[Graph')]
    puzzle_sentences = "\n".join(sentences)

    ground_truth = ",".join(sorted(list(shuffled_third_tallest)))

    return {
        "puzzle_text": puzzle_sentences,
        "ground_truth": ground_truth,
        "n_vertices": n_vertices,
        "max_tallest": max_tallest,
        "graph_edges": [(u, v) for u in graph.vertices for v in graph.edges[u]],
        "split_vertices": split_vertices,
    }, n_vertices


def _generate_split(num_samples: int, min_vertices: int, max_vertices: int) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    while len(data) < num_samples:
        item, _ = _generate_one_puzzle(min_vertices, max_vertices)
        if item:
            data.append(item)
    return data


def map_row_to_output(
    puzzle_data: Dict[str, Any],
    idx: int,
    split_label: str,
    variant: str,
    num_attempts: int,
) -> Dict[str, Any]:
    """Map a single puzzle row to the output format."""
    question = puzzle_data["puzzle_text"]

    if variant == "single":
        messages = build_single_attempt_prompt(question)
    else:
        messages = build_multi_attempt_prompt(question, num_attempts)

    data_source = "matan/taller_puzzles"
    ability = "logic"

    extra_info = {
        "split": split_label,
        "index": idx,
        "answer": puzzle_data["ground_truth"],
        "question": puzzle_data["puzzle_text"],
        "n_vertices": puzzle_data["n_vertices"],
        "max_tallest": puzzle_data["max_tallest"],
        "graph_edges": puzzle_data["graph_edges"],
        "split_vertices": puzzle_data["split_vertices"],
        "num_attempts": num_attempts if variant == "multi" else 1,
        "max_allowed_attempts": num_attempts if variant == "multi" else 1,
    }

    reward_model = {"style": "rule", "ground_truth": puzzle_data["ground_truth"]}

    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


if __name__ == "__main__":
    args = parse_args()

    if args.num_attempts < 1 or args.num_attempts > 8:
        raise ValueError("--num_attempts must be between 1 and 8 inclusive")
    if args.min_vertices < 3 or args.min_vertices > 26:
        raise ValueError("--min_vertices must be between 3 and 26 inclusive")
    if args.max_vertices < 3 or args.max_vertices > 26:
        raise ValueError("--max_vertices must be between 3 and 26 inclusive")
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("--train_size/--test_size must be non-negative")

    # Set seed for reproducibility
    random.seed(args.seed)

    # Generate base data for both splits
    print(f"Generating {args.train_size} training examples...")
    train_data = _generate_split(args.train_size, args.min_vertices, args.max_vertices)
    print(f"Generating {args.test_size} test examples...")
    test_data = _generate_split(args.test_size, args.min_vertices, args.max_vertices)

    # Create HF datasets
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    # Map to single and multi variants using the same base puzzles
    def make_map_fn(split_label: str, variant: str):
        def _inner(example, idx):
            return map_row_to_output(
                puzzle_data=example,
                idx=idx,
                split_label=split_label,
                variant=variant,
                num_attempts=args.num_attempts,
            )
        return _inner

    # Single-attempt
    train_single = train_dataset.map(function=make_map_fn("train", "single"), with_indices=True, remove_columns=train_dataset.column_names)
    test_single = test_dataset.map(function=make_map_fn("test", "single"), with_indices=True, remove_columns=test_dataset.column_names)

    # Multi-attempt
    train_multi = train_dataset.map(function=make_map_fn("train", "multi"), with_indices=True, remove_columns=train_dataset.column_names)
    test_multi = test_dataset.map(function=make_map_fn("test", "multi"), with_indices=True, remove_columns=test_dataset.column_names)

    # Prepare output directories
    single_dir = os.path.join(args.local_dir, "single_attempt")
    multi_dir = os.path.join(args.local_dir, f"multi_attempts_{args.num_attempts}")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)

    # Save parquet files
    train_single.to_parquet(os.path.join(single_dir, "train.parquet"))
    test_single.to_parquet(os.path.join(single_dir, "test.parquet"))
    train_multi.to_parquet(os.path.join(multi_dir, "train.parquet"))
    test_multi.to_parquet(os.path.join(multi_dir, "test.parquet"))

    # Print samples
    if args.n_print > 0 and len(train_single) > 0:
        print("Sample single-attempt train row:")
        print(train_single[0])
    if args.n_print > 0 and len(train_multi) > 0:
        print("Sample multi-attempt train row:")
        print(train_multi[0])

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            single_hdfs = os.path.join(args.hdfs_dir, "single_attempt")
            multi_hdfs = os.path.join(args.hdfs_dir, f"multi_attempts_{args.num_attempts}")
            makedirs(single_hdfs)
            makedirs(multi_hdfs)
            copy(src=single_dir, dst=single_hdfs)
            copy(src=multi_dir, dst=multi_hdfs)
        except Exception as e:
            print(f"Warning: failed to mirror to HDFS due to: {e}")

    print(
        f"Exported single: train={len(train_single)}, test={len(test_single)}; "
        f"exported multi({args.num_attempts}): train={len(train_multi)}, test={len(test_multi)}"
    )

# Example usage:
# PYTHONPATH=. python custom/data_preprocessing/taller/taller_puzzles_both.py \
#   --local_dir $HF_HOME/data/taller_puzzles_both --train_size 10000 --test_size 256 \
#   --max_vertices 12 --num_attempts 4


