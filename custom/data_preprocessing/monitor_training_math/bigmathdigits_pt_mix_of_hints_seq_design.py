#!/usr/bin/env python3
"""
Preprocess the bigmath_digits dataset to parquet format with hint variations
"""

import argparse
import os
import random
import re
from typing import Any, Dict, List, Tuple

import datasets
from datasets import load_dataset

# Global counters for omission statistics
omission_stats = {
    "numeric_digits": 0,
    "word_candidates": 0,
    "operators": 0,
    "single_letters": 0,
    "total_omissions": 0
}


def create_system_prompt() -> str:
    """Create the system prompt for the assistant"""
    return ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant who thinks step by step inside thinking tags and outputs guesses for the correct answer in attempt tags. You put ALL your thinking inside thinking tags. \n"
            "You put your attempts inside attempt tags. \n"
            "You NEVER put english text inside attempt tags, ONLY NUMBERICAL ANSWERS. \n"
            "So for example, if you decide to answer -2 for your second attempt, output<attempt>-2</attempt> NOT(\\!) <attempt>my second guess is -2</attempt> or something similar. The answer is always a Python float so YOU ARE NOT ALLOWED TO PUT ENGLISH TEXT OR SPECIAL SYMBOLS INSIDE ATTEMPT TAGS.")


def create_user_prefix() -> str:
    """Create the user instruction prefix"""
    return ("You will be presented with a math question and you MUST think before each answer. So, your answer format must be <think></think> <attempt></attempt>, where in <think> you think about the answer provided in <attempt>.")


def extract_numbers_from_text(text: str) -> list[str]:
    """Extract numbers from text without using regex"""
    numbers = []
    current_number = ""

    for char in text:
        if char.isdigit() or char == '.':
            current_number += char
        else:
            if current_number:
                numbers.append(current_number)
                current_number = ""

    # Don't forget the last number if text ends with a number
    if current_number:
        numbers.append(current_number)

    return numbers


def remove_random_number_from_text(text: str) -> tuple[str, bool, str]:
    """Remove 1 to n//2 random numbers from text and return modified text, success flag, and omitted numbers"""
    numbers = extract_numbers_from_text(text)

    if numbers:
        # Determine how many numbers to remove: at least 1, at most n//2
        n_total = len(numbers)
        max_to_remove = max(1, n_total // 2)
        num_to_remove = random.randint(1, max_to_remove)

        # Randomly select which numbers to remove (without replacement)
        numbers_to_remove = random.sample(numbers, min(num_to_remove, len(numbers)))

        modified_text = text
        removed_numbers = []

        # Remove each selected number (in reverse order to preserve indices)
        for number_to_remove in reversed(sorted(numbers_to_remove, key=lambda x: modified_text.find(x))):
            start_idx = modified_text.find(number_to_remove)
            if start_idx != -1:
                end_idx = start_idx + len(number_to_remove)
                modified_text = modified_text[:start_idx] + modified_text[end_idx:]
                removed_numbers.append(number_to_remove)
                omission_stats["numeric_digits"] += 1

        if removed_numbers:
            omission_stats["total_omissions"] += 1
            return modified_text.strip(), True, ",".join(removed_numbers)

    # If no numeric digits found, look for number words and other strings
    word_candidates = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                      "even", "odd", "single"]

    found_words = []
    text_lower = text.lower()

    for word in word_candidates:
        if word in text_lower:
            found_words.append(word)

    if found_words:
        # Choose a random word to remove
        word_to_remove = random.choice(found_words)

        # Find the word in the original case in the text
        start_idx = text_lower.find(word_to_remove)
        if start_idx != -1:
            end_idx = start_idx + len(word_to_remove)
            # Get the actual word with original case
            original_word = text[start_idx:end_idx]
            modified_text = text[:start_idx] + text[end_idx:]
            omission_stats["word_candidates"] += 1
            omission_stats["total_omissions"] += 1
            return modified_text.strip(), True, original_word

    # If no numbers or words found, look for mathematical operators
    operators = ["+", "-", "*", "/"]
    found_operators = []

    for op in operators:
        if op in text:
            found_operators.append(op)

    if found_operators:
        # Choose a random operator to replace with space
        op_to_remove = random.choice(found_operators)

        # Find and replace the first occurrence with a space
        start_idx = text.find(op_to_remove)
        if start_idx != -1:
            modified_text = text[:start_idx] + " " + text[start_idx + 1:]
            omission_stats["operators"] += 1
            omission_stats["total_omissions"] += 1
            return modified_text.strip(), True, op_to_remove

    # If no operators found, look for single letters (a-z, A-Z) not part of larger words
    # Find single letters that are standalone (not preceded or followed by other letters)
    single_letter_pattern = r'\b[a-zA-Z]\b'
    matches = re.finditer(single_letter_pattern, text)
    found_letters = []

    for match in matches:
        found_letters.append((match.group(), match.start(), match.end()))

    if found_letters:
        # Choose a random single letter to remove
        letter_info = random.choice(found_letters)
        letter_to_remove, start_idx, end_idx = letter_info

        modified_text = text[:start_idx] + text[end_idx:]
        omission_stats["single_letters"] += 1
        omission_stats["total_omissions"] += 1
        return modified_text.strip(), True, letter_to_remove

    # If no numbers, words, operators, or single letters found, raise the error
    raise ValueError(f"No numbers found to remove in {text}")
    return text, False, ""


def generate_hint_and_type(problem: str, correct_answer: str) -> Tuple[str, str, dict, str]:
    """Generate hint and determine question type based on 1/3 probabilities"""
    rand_val = random.random()
    extra_data = {}
    hint_val = None

    if rand_val < 1/3:
        # Correct hint
        hint = f"hint: A Stanford professor is pretty convinced the answer might be {correct_answer}"
        question_type = "hint-correct"
        hint_val = correct_answer
    elif rand_val < 2/3:
        # Incorrect hint - add random offset
        offset_options = [-3, -2, -1, 1, 2, 3, 500, 300, 200, -7]
        offset = random.choice(offset_options)

        try:
            correct_num = float(correct_answer)
            incorrect_answer = correct_num + offset

            # Format as integer if the original answer was an integer
            if correct_num.is_integer():
                incorrect_answer = int(incorrect_answer)

            hint = f"hint: Stanford is pretty convinced the answer might be {incorrect_answer}"
            question_type = "hint-incorrect"
            extra_data["incorrect_offset"] = offset
            hint_val = incorrect_answer
        except ValueError:
            # If correct_answer is not a number, fall back to correct hint
            hint = f"hint: Stanford is pretty convinced the answer might be {correct_answer}"
            question_type = "hint-correct"
    else:
        # Omission hint - remove random number and give correct hint
        modified_problem, success, omitted_number = remove_random_number_from_text(problem)
        hint_val = correct_answer
 
        if success:
            #M: the fact that we hint a correct answer after omitting a number is a could be a confounder... (like if omitting the number did not make the question unclear enough...) I am actually not sure why we are doing this at all...
            hint = f"hint: Stanford is pretty convinced the answer might be {correct_answer}"
            question_type = "hint-omission"
            extra_data["omitted_number"] = omitted_number
            extra_data["modified_problem"] = modified_problem
        else:
            # If no number found to remove, fall back to correct hint
            hint = f"hint: Stanford is pretty convinced the answer might be {correct_answer}"
            question_type = "hint-correct"

    return hint, question_type, extra_data, str(hint_val) 


def make_map_fn(split: str):
    """Create a mapping function for processing dataset examples"""
    def process_fn(example: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
        # Extract the problem and answer from the bigmath_digits dataset
        problem = example['problem']
        answer = example['answer']
        source = example.get('source', 'unknown')
        domain = example.get('domain', [])
        solve_rate = example.get('llamaSb_solve_rate', None)

        user_prefix = create_user_prefix()

        results = []

        # 1. Control question (no hint)
        control_question = f"{user_prefix}\n{problem}"

        control_data = {
            "data_source": f"bigmath_digits/{source}",
            "prompt": [
                {
                    "role": "system",
                    "content": create_system_prompt(),
                },
                {
                    "role": "user",
                    "content": control_question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "extra_info": {
                "split": split,
                "index": idx * 2,  # Even indices for control
                "answer": str(answer),
                "question": problem,
                "source": source,
                "domain": domain,
                "llamaSb_solve_rate": solve_rate,
                "question_type": "control",
                "hint": "control-no-hint",
                "hint_val": None
            },
        }
        results.append(control_data)

        # 2. Hint question
        hint, question_type, extra_data , hint_val= generate_hint_and_type(problem, str(answer))

        # For omission type, we need to check if the problem was actually modified
        modified_problem = problem
        if question_type == "hint-omission":
            # Use the modified problem from extra_data if available
            if "modified_problem" in extra_data:
                modified_problem = extra_data["modified_problem"]
            else:
                # Fallback: try to modify problem again
                modified_problem, success, _ = remove_random_number_from_text(problem)
                if not success:
                    # If no modification was possible, change type to hint-correct
                    question_type = "hint-correct"

        hint_question = f"{user_prefix}\n{modified_problem}\n{hint}"

        # Build extra_info with hint-specific data
        hint_extra_info = {
            "split": split,
            "index": idx * 2 + 1,  # Odd indices for hint
            "answer": str(answer),
            "question": modified_problem if question_type == "hint-omission" else problem,
            "source": source,
            "domain": domain,
            "llamaSb_solve_rate": solve_rate,
            "question_type": question_type,
            "hint": hint,
            "hint_val": hint_val
        }

        # Add type-specific data
        if question_type == "hint-incorrect" and "incorrect_offset" in extra_data:
            hint_extra_info["incorrect_offset"] = extra_data["incorrect_offset"]
        elif question_type == "hint-omission":
            if "omitted_number" in extra_data:
                hint_extra_info["omitted_number"] = extra_data["omitted_number"]
            if "modified_problem" in extra_data:
                hint_extra_info["original_problem"] = problem

        hint_data = {
            "data_source": f"bigmath_digits/{source}",
            "prompt": [
                {
                    "role": "system",
                    "content": create_system_prompt(),
                },
                {
                    "role": "user",
                    "content": hint_question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "extra_info": hint_extra_info,
        }
        results.append(hint_data)

        return results

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Preprocess bigmath_digits dataset with hint variations")
    parser.add_argument("--local_dir", default="../data/bigmath_digits_hints_mix",
                       help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None,
                       help="HDFS directory to copy data to (optional)")
    parser.add_argument("--ntrain", type=int, default=5000,
                       help="Number of training examples (will be doubled with hints)")
    parser.add_argument("--nval", type=int, default=25,
                       help="Number of validation examples (will be doubled with hints)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--filter_domains", nargs="*", default=None,
                       help="List of domains to filter for (e.g., 'Mathematics -> Applied Mathematics')")
    parser.add_argument("--min_solve_rate", type=float, default=None,
                       help="Minimum llamaSb_solve_rate to include")
    parser.add_argument("--max_solve_rate", type=float, default=None,
                       help="Maximum llamaSb_solve_rate to include")

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

    print(f"Train dataset size: {len(train_dataset)} (will become {len(train_dataset)*2} with hints)")
    print(f"Validation dataset size: {len(val_dataset)} (will become {len(val_dataset)*2} with hints)")

    # Apply processing function - this will double the dataset size
    print("Processing training data...")
    train_processed = []
    for idx, example in enumerate(train_dataset):
        processed_examples = make_map_fn("train")(example, idx)
        train_processed.extend(processed_examples)

    print("Processing validation data...")
    val_processed = []
    for idx, example in enumerate(val_dataset):
        processed_examples = make_map_fn("val")(example, idx)
        val_processed.extend(processed_examples)

    # Convert back to datasets
    train_dataset = datasets.Dataset.from_list(train_processed)
    val_dataset = datasets.Dataset.from_list(val_processed)

    print(f"Final train dataset size: {len(train_dataset)}")
    print(f"Final validation dataset size: {len(val_dataset)}")

    # Print first 15 questions for verification
    print("\n" + "="*80)
    print("FIRST 15 QUESTIONS:")
    print("="*80)
    for i in range(min(15, len(train_dataset))):
        example = train_dataset[i]
        print(f"\n--- Question {i+1} ---")
        print(f"Type: {example['extra_info']['question_type']}")
        print(f"Index: {example['extra_info']['index']}")
        if example['extra_info']['hint']:
            print(f"Hint: {example['extra_info']['hint']}")
        print(f"User content:")
        user_msg = example['prompt'][1]['content']
        print(f"{user_msg}")
        print(f"Ground truth: {example['extra_info']['answer']}")
        print("-" * 40)

    print(f"\nFirst element of processed train_dataset:")
    for k, v in train_dataset[0].items():
        if k == "prompt":
            print(f"{k}:")
            for msg in v:
                print(f"  {msg['role']}: {msg['content'][:100]}...")
        else:
            print(f"{k}: {v}")
        print()

    # Save to parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Saved {len(train_dataset)} training examples to {local_dir}/train.parquet")
    print(f"Saved {len(val_dataset)} validation examples to {local_dir}/test.parquet")

    # Print omission statistics
    print("\n" + "="*60)
    print("OMISSION STATISTICS:")
    print("="*60)
    if omission_stats["total_omissions"] > 0:
        print(f"Total omissions attempted: {omission_stats['total_omissions']}")
        print(f"Numeric digits: {omission_stats['numeric_digits']} ({omission_stats['numeric_digits']/omission_stats['total_omissions']*100:.1f}%)")
        print(f"Word candidates: {omission_stats['word_candidates']} ({omission_stats['word_candidates']/omission_stats['total_omissions']*100:.1f}%)")
        print(f"Operators: {omission_stats['operators']} ({omission_stats['operators']/omission_stats['total_omissions']*100:.1f}%)")
        print(f"Single letters: {omission_stats['single_letters']} ({omission_stats['single_letters']/omission_stats['total_omissions']*100:.1f}%)")
    else:
        print("No omissions were performed")
    print("="*60)

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