#!/usr/bin/env python3
"""
Script to convert the 'answer' field from int to string in a parquet dataset
and add a 'data_source' column extracted from the file path.
Modifies the dataset in-place.
"""

import argparse
import os
import sys
from pathlib import Path

import datasets
from datasets import Value


def extract_data_source_from_path(dataset_path: str) -> str:
    """
    Extract the data source name from the file path.
    
    For paths like: /HF_HOME/data/aime24/data/train-00000-of-00001.parquet
    Returns: "aime24"
    
    Args:
        dataset_path: Path to the parquet file
        
    Returns:
        The data source name (directory name after /data/)
    """
    path = Path(dataset_path)
    parts = path.parts
    
    # Look for "data" in the path and get the next directory
    for i, part in enumerate(parts):
        if part == "data" and i + 1 < len(parts):
            return parts[i + 1]
    
    # Fallback: use the parent directory name if "data" pattern not found
    return path.parent.name


def convert_answer_to_string(dataset_path: str, data_source: str = None):
    """
    Read a parquet dataset, convert 'answer' field from int to string, 
    add 'data_source' column, and save in-place.
    
    Args:
        dataset_path: Path to the parquet file
        data_source: Optional data source name. If not provided, extracted from path.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Extract data source from path if not provided
    if data_source is None:
        data_source = extract_data_source_from_path(dataset_path)
    
    print(f"Loading dataset from: {dataset_path}")
    print(f"Data source: {data_source}")
    
    # Load the dataset
    dataset = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Add data_source column if it doesn't exist or update it
    if "data_source" not in dataset.column_names:
        print(f"Adding 'data_source' column with value: {data_source}")
        dataset = dataset.add_column("data_source", [data_source] * len(dataset))
    else:
        print(f"Updating 'data_source' column with value: {data_source}")
        def update_data_source(example):
            example["data_source"] = data_source
            return example
        dataset = dataset.map(update_data_source)
        dataset = dataset.cast_column("data_source", Value("string"))
    
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
                    # Note: For nested fields in extra_info, we can't easily cast the schema
                    # but the values will be converted correctly
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
        
        # Explicitly cast the column to string type to update the schema
        if "answer" in dataset.features:
            dataset = dataset.cast_column("answer", Value("string"))
    
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
    
    print(f"Successfully converted 'answer' field to string, added 'data_source' column, and saved to: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 'answer' field from int to string and add 'data_source' column in a parquet dataset (in-place)"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the parquet dataset file"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Data source name (e.g., 'aime24'). If not provided, extracted from path."
    )
    
    args = parser.parse_args()
    
    try:
        convert_answer_to_string(args.dataset_path, data_source=args.data_source)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

