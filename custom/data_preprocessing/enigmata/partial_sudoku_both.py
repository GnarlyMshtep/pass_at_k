"""
Preprocess sudoku2 JSONL data into standardized parquet datasets with prompts
that allow either full solutions or partial answers, producing both a
single-attempt variant and a multi-attempt variant in one run.

Input JSONL schema (per row example):
{
  "prompt": <string>,
  "answer": [[...],[...],...],
  "task_name": "sudoku2",
  "ability": "logic_puzzle",
  "language": "en",
  "meta": {
    "id": <string>,
    "question": [[...],[...],...],  # 0 for holes
    "holes": <int>,
    "answer": [[...],[...],...],
    "split": "train"|"test",
    "difficulty_level": <string>,
    "difficulty": <string>,
    ...
  }
}

Output parquet schema columns (aligned with other preprocessors in this repo):
- data_source (string)
- prompt (list[dict])  # [{"content": str, "role": "user"}]
- ability (string)
- reward_model (dict)
- extra_info (dict)

This script produces TWO datasets under --local_dir:
  - single_attempt/train.parquet and test.parquet
  - multi_attempts_<N>/train.parquet and test.parquet (N = --num_attempts)

Optional: --hdfs_dir to mirror the two directories to HDFS.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import datasets

from verl.utils.hdfs_io import copy, makedirs


def grid_size_from_question(question: Optional[List[List[int]]]) -> Optional[int]:
    if not question or not isinstance(question, list):
        return None
    try:
        n = len(question)
        if n > 0 and all(isinstance(row, list) and len(row) == n for row in question):
            return n
    except Exception:
        return None
    return None


def grid_size_from_answer(answer: Optional[List[List[int]]]) -> Optional[int]:
    """Extract grid size from answer grid as fallback."""
    if not answer or not isinstance(answer, list):
        return None
    try:
        n = len(answer)
        if n > 0 and all(isinstance(row, list) and len(row) == n for row in answer):
            return n
    except Exception:
        return None
    return None


def format_grid_as_text(grid: List[List[int]]) -> str:
    # Zero means empty; render as '0'
    lines: List[str] = []
    for row in grid:
        line = " ".join(str(int(v)) for v in row)
        lines.append(line)
    return "\n".join(lines)


def format_solution_grid_as_text(grid: List[List[int]]) -> str:
    lines: List[str] = []
    for row in grid:
        line = " ".join(str(int(v)) for v in row)
        lines.append(line)
    return "\n".join(lines)


def build_single_attempt_prompt(question_grid: List[List[int]], size: int) -> List[Dict[str, str]]:
    puzzle_block = format_grid_as_text(question_grid)

    # Generate examples tailored to the board size
    if size == 4:
        example_partials = (
            "A partial attempt: <answer>```\n"
            "1 2 0 3\n"
            "3 0 2 0\n"
            "0 1 3 2\n"
            "2 3 1 0\n"
            "```</answer>\n"
        )
        example_full = (
            "A complete attempt: <answer>```\n"
            "2 4 3 1\n"
            "3 1 2 4\n"
            "4 3 1 2\n"
            "1 2 4 3\n"
            "```</answer>\n"
        )
    elif size == 9:
        example_partials = (
            "A partial attempt: <answer>```\n"
            "5 3 0 0 7 0 0 0 0\n"
            "6 0 0 1 9 5 0 0 0\n"
            "0 9 8 0 0 0 0 6 0\n"
            "8 0 0 0 6 0 0 0 3\n"
            "4 0 0 8 0 3 0 0 1\n"
            "7 0 0 0 2 0 0 0 6\n"
            "0 6 0 0 0 0 2 8 0\n"
            "0 0 0 4 1 9 0 0 5\n"
            "0 0 0 0 8 0 0 7 9\n"
            "```</answer>\n"
        )
        example_full = (
            "A complete attempt: <answer>```\n"
            "5 3 4 6 7 8 9 1 2\n"
            "6 7 2 1 9 5 3 4 8\n"
            "1 9 8 3 4 2 5 6 7\n"
            "8 5 9 7 6 1 4 2 3\n"
            "4 2 6 8 5 3 7 9 1\n"
            "7 1 3 9 2 4 8 5 6\n"
            "9 6 1 5 3 7 2 8 4\n"
            "2 8 7 4 1 9 6 3 5\n"
            "3 4 5 2 8 6 1 7 9\n"
            "```</answer>\n"
        )
    else:
        # Generic examples for other sizes
        raise ValueError(f"Unsupported board size: {size}")

    content = (
        f"You are provided with a {size}x{size} Sudoku puzzle. Empty cells are '0'.\n"
        "Rules:\n"
        f"- Each row and column must contain numbers 1..{size} without repeats.\n"
        f"- Each subgrid must contain numbers 1..{size} without repeats.\n\n"
        "Task:\n"
        "- Find a valid solution. If multiple solutions exist, provide one.\n\n"
        "Answer format:\n"
        "- Think step by step inside <think>...</think>.\n"
        "- Then, inside <answer>...</answer>, output EITHER:\n"
        "  (a) a complete solved board as a fenced code block (```), OR\n"
        "  (b) a partially filled board in the same grid format with 0 for unknown cells.\n"
        "  Do not include prose inside <answer>.\n\n"
        "**Scoring:**\n"
        "- **Accuracy is prioritized over completeness.**\n"
        "- Your score is the number of correctly filled cells.\n"
        "- **If ANY filled cell is incorrect, the total score is 0.**\n"
        "- **DO NOT GUESS.** If you are uncertain about any cell, leave it as '0'. Output a partial board instead of an incorrect full one.\n\n"
        "Examples (illustrative):\n"
        f"{example_partials}{example_full}"
        "Here is the puzzle:\n"
        "```\n"
        f"{puzzle_block}\n"
        "```\n\n"
        "Now, think in <think>...</think>, then give your final output in <answer>...</answer>."
    )

    return [{"content": content, "role": "user"}]


def build_multi_attempt_prompt(question_grid: List[List[int]], size: int, num_attempts: int) -> List[Dict[str, str]]:
    puzzle_block = format_grid_as_text(question_grid)
    attempt_tags = [f"<attempt-{i}></attempt-{i}>" for i in range(1, num_attempts + 1)]
    if num_attempts > 3:
        preview = attempt_tags[:2] + ["..."] + attempt_tags[-1:]
    else:
        preview = attempt_tags
    attempts_format = ", ".join(preview)

    # Generate examples tailored to the board size
    if size == 4:
        example_attempts = (
            "A partial attempt: <attempt>```\n1 2 0 3\n3 0 2 0\n0 1 3 2\n2 3 1 0\n```</attempt>\n"
            "A complete attempt: <attempt>```\n2 4 3 1\n3 1 2 4\n4 3 1 2\n1 2 4 3\n```</attempt>\n"
        )
    elif size == 9:
        example_attempts = (
            "A partial attempt: <attempt>```\n5 3 0 0 7 0 0 0 0\n6 0 0 1 9 5 0 0 0\n0 9 8 0 0 0 0 6 0\n8 0 0 0 6 0 0 0 3\n4 0 0 8 0 3 0 0 1\n7 0 0 0 2 0 0 0 6\n0 6 0 0 0 0 2 8 0\n0 0 0 4 1 9 0 0 5\n0 0 0 0 8 0 0 7 9\n```</attempt>\n"
            "A complete attempt: <attempt>```\n5 3 4 6 7 8 9 1 2\n6 7 2 1 9 5 3 4 8\n1 9 8 3 4 2 5 6 7\n8 5 9 7 6 1 4 2 3\n4 2 6 8 5 3 7 9 1\n7 1 3 9 2 4 8 5 6\n9 6 1 5 3 7 2 8 4\n2 8 7 4 1 9 6 3 5\n3 4 5 2 8 6 1 7 9\n```</attempt>\n"
        )
    else:
        raise ValueError(f"Unsupported board size: {size}")

    content = (
        f"You are provided with a {size}x{size} Sudoku puzzle. Empty cells are '0'.\n"
        "Rules:\n"
        f"- Each row and column must contain numbers 1..{size} without repeats.\n"
        f"- Each subgrid must contain numbers 1..{size} without repeats.\n\n"
        "Task:\n"
        "- Find a valid solution. If multiple solutions exist, provide one.\n\n"
        "Answer options per attempt:\n"
        "- In each attempt, output EITHER:\n"
        "  (a) a complete solved board as a fenced code block (```), OR\n"
        "  (b) a partially filled board in the same grid format with 0 for unknown cells.\n"
        "  Do not include prose inside attempts.\n"
        "  Partial boards are explicitly allowed in any attempt and preferred over incorrect full solutions.\n\n"
        "**Scoring:**\n"
        "- **Accuracy is prioritized over completeness.**\n"
        "- Your score is the number of correctly filled cells.\n"
        "- **If ANY filled cell is incorrect, the total score is 0.**\n"
        "- **DO NOT GUESS.** If you are uncertain about any cell, leave it as '0'. Output a partial board instead of an incorrect full one.\n\n"
        "Final score for this problem is the maximum across all attempts.\n\n"
        "Illustrative examples:\n"
        f"{example_attempts}"
        "Here is the puzzle:\n"
        "```\n"
        f"{puzzle_block}\n"
        "```"
        "Now, first think in <think></think>.\n"
        f"After </think>, output all {num_attempts} final attempts, each inside its own <attempt-i></attempt-i> tag (i = 1..{num_attempts}).\n"
        f"Your output format must be: <think> … </think> followed by {attempts_format}.\n\n"
    )

    return [{"content": content, "role": "user"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train.jsonl and test.jsonl (e.g., $HF_HOME/data/enigmata/sudoku2/en)",
    )
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Base local directory to save outputs; subdirs will be created for single and multi variants",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--train_size",
        type=int,
        default=-1,
        help="Max number of training rows to include (-1 for all)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=-1,
        help="Max number of test rows to include (-1 for all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle before downsampling",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=3,
        help="Number of attempts for the multi-attempt dataset (1-8)",
    )
    return parser.parse_args()


def select_split(dataset: datasets.Dataset, size: int) -> datasets.Dataset:
    if size is None or size < 0:
        return dataset
    if size == -1:
        return dataset
    if size > len(dataset):
        raise ValueError(f"Requested size ({size}) exceeds available rows ({len(dataset)}).")
    return dataset.select(range(size))


def map_row_to_output(
    example: Dict[str, Any],
    idx: int,
    split_label: str,
    variant: str,
    num_attempts: int,
) -> Dict[str, Any]:
    # Handle different meta formats (dict vs JSON string)
    meta_raw = example.get("meta", {})
    if isinstance(meta_raw, str):
        try:
            meta: Dict[str, Any] = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    else:
        meta = meta_raw or {}
    
    question: Optional[List[List[int]]] = meta.get("question")
    solution: Optional[List[List[int]]] = example.get("answer") or meta.get("answer")
    

    size = grid_size_from_question(question)

    if not question and solution:
        # Build a blanked question grid if missing, using zeros
        n = len(solution)
        question = [[0 for _ in range(n)] for _ in range(n)]

    if variant == "single":
        messages = build_single_attempt_prompt(question, size)
    else:
        messages = build_multi_attempt_prompt(question, size, num_attempts)

    # Determine data source based on task name
    task_name = example['task_name']
    data_source = task_name
    ability = example['ability']

    # Compute effective max attempts respected by the verifier
    effective_max_allowed = num_attempts if variant == "multi" else 1
    

    extra_info = {
        "split": split_label,
        "index": idx,
        "dataset_name": data_source,
        "task_name": task_name,
        "difficulty": meta.get('difficulty_level') or meta.get('difficulty'),
        "language": example['language'],
        "id": meta['id'],
        "holes": meta['holes'],
        "board_size": size,
        "question": question,
        "num_attempts": num_attempts if variant == "multi" else 1,
        "max_allowed_attempts": effective_max_allowed,
    }

    reward_model = {
        "style": "rule",
        "ground_truth": solution,
        "board_size": size,
        "question": question,
    }

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

    data_files = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }
    for split_name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split_name} file at: {path}")

    dataset_dict = datasets.load_dataset("json", data_files=data_files)
    train_base = dataset_dict["train"].shuffle(seed=args.seed)
    test_base = dataset_dict["test"].shuffle(seed=args.seed)

    train_raw = select_split(train_base, args.train_size if args.train_size is not None else -1)
    test_raw = select_split(test_base, args.test_size if args.test_size is not None else -1)

    # Build both variants
    def make_map_fn(split_label: str, variant: str):
        def _inner(example, idx):
            return map_row_to_output(
                example=example,
                idx=idx,
                split_label=split_label,
                variant=variant,
                num_attempts=args.num_attempts,
            )
        return _inner

    # Single-attempt
    train_single = train_raw.map(function=make_map_fn("train", "single"), with_indices=True)
    test_single = test_raw.map(function=make_map_fn("test", "single"), with_indices=True)

    # Multi-attempt
    train_multi = train_raw.map(function=make_map_fn("train", "multi"), with_indices=True)
    test_multi = test_raw.map(function=make_map_fn("test", "multi"), with_indices=True)

    # Report counts
    print(
        f"Loaded: train={len(train_base)}, test={len(test_base)}; "
        f"exported single: train={len(train_single)}, test={len(test_single)}; "
        f"exported multi({args.num_attempts}): train={len(train_multi)}, test={len(test_multi)}"
    )

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

    # Print a couple of sample rows
    if len(train_single) > 0:
        print("Sample single-attempt train row:")
        print(train_single[0])
    if len(train_multi) > 0:
        print("Sample multi-attempt train row:")
        print(train_multi[0])

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        single_hdfs = os.path.join(args.hdfs_dir, "single_attempt")
        multi_hdfs = os.path.join(args.hdfs_dir, f"multi_attempts_{args.num_attempts}")
        makedirs(single_hdfs)
        makedirs(multi_hdfs)
        copy(src=single_dir, dst=single_hdfs)
        copy(src=multi_dir, dst=multi_hdfs)

#  PYTHONPATH=. python custom/data_preprocessing/enigmata/partial_sudoku2_both.py --local_dir $HF_HOME/data/enigmata/partial_sudoku2 --test_size 256 --num_attempts 8 --data_dir $HF_HOME/data/enigmata/sudoku2/en
