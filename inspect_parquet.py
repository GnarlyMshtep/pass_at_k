#!/usr/bin/env python3
"""
Script to inspect a parquet file and print its columns and their types.
"""

import argparse
import json
import sys
from typing import Any

import datasets
import pyarrow.parquet as pq


def get_type_str(value: Any) -> str:
    """Get a string representation of a value's type."""
    if value is None:
        return "None"
    type_name = type(value).__name__
    
    # Handle nested structures
    if isinstance(value, dict):
        return f"dict (keys: {list(value.keys())})"
    elif isinstance(value, list):
        if len(value) > 0:
            return f"list[{type(value[0]).__name__}] (length: {len(value)})"
        else:
            return "list[]"
    elif isinstance(value, tuple):
        if len(value) > 0:
            return f"tuple[{', '.join(type(v).__name__ for v in value)}]"
        else:
            return "tuple[]"
    
    return type_name


def format_value(value: Any, max_length: int = 200) -> str:
    """Format a value for display, handling nested structures."""
    if value is None:
        return "None"
    elif isinstance(value, (dict, list)):
        # Use JSON for structured data, but truncate if too long
        try:
            formatted = json.dumps(value, indent=2, ensure_ascii=False)
            if len(formatted) > max_length:
                formatted = formatted[:max_length] + "... (truncated)"
            return formatted
        except (TypeError, ValueError):
            # Fallback to string representation
            str_repr = str(value)
            if len(str_repr) > max_length:
                str_repr = str_repr[:max_length] + "... (truncated)"
            return str_repr
    else:
        str_repr = str(value)
        if len(str_repr) > max_length:
            str_repr = str_repr[:max_length] + "... (truncated)"
        return str_repr


def inspect_parquet(dataset_path: str, show_sample: bool = False, show_rows: int = 0):
    """
    Inspect a parquet file and print column names and types.
    
    Args:
        dataset_path: Path to the parquet file
        show_sample: Whether to show a sample value for each column
        show_rows: Number of rows to display (0 means don't show rows)
    """
    print(f"Inspecting parquet file: {dataset_path}")
    print("=" * 80)
    
    # Load using datasets library
    try:
        dataset = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
    except Exception as e:
        print(f"Error loading dataset with datasets library: {e}", file=sys.stderr)
        print("Trying with pyarrow directly...", file=sys.stderr)
        try:
            parquet_file = pq.ParquetFile(dataset_path)
            arrow_schema = parquet_file.schema_arrow
            print("\nSchema (from pyarrow):")
            print("-" * 80)
            for field in arrow_schema:
                print(f"  {field.name}: {field.type}")
            print(f"\nTotal rows: {parquet_file.metadata.num_rows}")
            return
        except Exception as e2:
            print(f"Error loading with pyarrow: {e2}", file=sys.stderr)
            sys.exit(1)
    
    print(f"\nTotal rows: {len(dataset)}")
    print(f"Number of columns: {len(dataset.column_names)}")
    print("\nColumns and types:")
    print("-" * 80)
    
    # Get first example to inspect types
    if len(dataset) > 0:
        first_example = dataset[0]
        
        for col_name in dataset.column_names:
            value = first_example[col_name]
            type_str = get_type_str(value)
            
            # Get the feature type from the dataset if available
            feature = dataset.features.get(col_name) if hasattr(dataset, 'features') else None
            feature_type = str(feature) if feature else "unknown"
            
            print(f"\n  Column: {col_name}")
            print(f"    Type: {type_str}")
            if feature:
                print(f"    Feature: {feature_type}")
            
            if show_sample:
                # Show sample value (truncate if too long)
                sample_str = str(value)
                if len(sample_str) > 200:
                    sample_str = sample_str[:200] + "... (truncated)"
                print(f"    Sample: {sample_str}")
            
            # For nested dicts, show structure
            if isinstance(value, dict):
                print(f"    Structure:")
                for key, val in value.items():
                    print(f"      - {key}: {get_type_str(val)}")
    else:
        print("\nDataset is empty - no rows to inspect.")
    
    # Show first N rows if requested
    if show_rows > 0 and len(dataset) > 0:
        print("\n" + "=" * 80)
        print(f"\nFirst {min(show_rows, len(dataset))} rows:")
        print("-" * 80)
        
        num_rows_to_show = min(show_rows, len(dataset))
        for i in range(num_rows_to_show):
            row = dataset[i]
            print(f"\nRow {i + 1}:")
            print("-" * 40)
            for col_name in dataset.column_names:
                value = row[col_name]
                formatted_value = format_value(value, max_length=500)
                print(f"  {col_name}: {formatted_value}")
    
    print("\n" + "=" * 80)
    
    # Also show pyarrow schema for more detailed type information
    try:
        parquet_file = pq.ParquetFile(dataset_path)
        # Get the Arrow schema from the Parquet schema
        arrow_schema = parquet_file.schema_arrow
        print("\nPyArrow Schema (detailed):")
        print("-" * 80)
        for field in arrow_schema:
            print(f"  {field.name}: {field.type}")
    except Exception as e:
        # Fallback: try alternative method
        try:
            parquet_file = pq.ParquetFile(dataset_path)
            arrow_schema = parquet_file.schema.to_arrow_schema()
            print("\nPyArrow Schema (detailed):")
            print("-" * 80)
            for field in arrow_schema:
                print(f"  {field.name}: {field.type}")
        except Exception as e2:
            print(f"\nCould not read PyArrow schema: {e2}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a parquet file and print its columns and types"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the parquet dataset file"
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Show a sample value for each column"
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        default=5,
        help="Number of rows to display (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        inspect_parquet(args.dataset_path, show_sample=args.show_sample, show_rows=args.show_rows)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

