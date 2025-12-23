#!/usr/bin/env python3
"""
Script to convert the 'answer' field from int to string in a parquet dataset.
Modifies the dataset in-place.
"""

import argparse
import os
import sys
from pathlib import Path

import datasets


def convert_answer_to_string(dataset_path: str):
    """
    Read a parquet dataset, convert 'answer' field from int to string, and save in-place.
    
    Args:
        dataset_path: Path to the parquet file
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load the dataset
    dataset = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Check if 'answer' field exists
    if "answer" not in dataset.column_names:
        # Check if it's in extra_info
        if "extra_info" in dataset.column_names:
            print("'answer' not found at top level. Checking 'extra_info'...")
            # Check first example to see structure
            first_example = dataset[0]
            if "extra_info" in first_example and isinstance(first_example["extra_info"], dict):
                if "answer" in first_example["extra_info"]:
                    print("Found 'answer' in 'extra_info'. Converting...")
                    
                    def convert_extra_info_answer(example):
                        if "extra_info" in example and isinstance(example["extra_info"], dict):
                            if "answer" in example["extra_info"]:
                                answer = example["extra_info"]["answer"]
                                if isinstance(answer, int):
                                    example["extra_info"]["answer"] = str(answer)
                                elif answer is not None:
                                    example["extra_info"]["answer"] = str(answer)
                        return example
                    
                    dataset = dataset.map(convert_extra_info_answer)
                else:
                    raise ValueError("'answer' field not found in dataset or extra_info")
            else:
                raise ValueError("'answer' field not found in dataset")
        else:
            raise ValueError("'answer' field not found in dataset")
    else:
        # Convert top-level 'answer' field
        print("Found 'answer' at top level. Converting from int to string...")
        
        def convert_answer(example):
            if "answer" in example:
                answer = example["answer"]
                if isinstance(answer, int):
                    example["answer"] = str(answer)
                elif answer is not None:
                    example["answer"] = str(answer)
            return example
        
        dataset = dataset.map(convert_answer)
    
    # Verify conversion
    first_example = dataset[0]
    if "answer" in first_example:
        print(f"Sample 'answer' value after conversion: {first_example['answer']} (type: {type(first_example['answer']).__name__})")
    elif "extra_info" in first_example and isinstance(first_example["extra_info"], dict):
        if "answer" in first_example["extra_info"]:
            print(f"Sample 'answer' value in extra_info after conversion: {first_example['extra_info']['answer']} (type: {type(first_example['extra_info']['answer']).__name__})")
    
    # Save back to the same location
    print(f"Saving modified dataset to: {dataset_path}")
    
    # Create a temporary file first, then replace the original
    temp_path = dataset_path + ".tmp"
    dataset.to_parquet(temp_path)
    
    # Replace original with the modified version
    os.replace(temp_path, dataset_path)
    
    print(f"Successfully converted 'answer' field to string and saved to: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 'answer' field from int to string in a parquet dataset (in-place)"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the parquet dataset file"
    )
    
    args = parser.parse_args()
    
    try:
        convert_answer_to_string(args.dataset_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

