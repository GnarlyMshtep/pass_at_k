#!/bin/bash
# Quick helper script to view parquet files in the output directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Check if a file was provided as argument
if [ -n "$1" ]; then
    FILE_PATH="$1"
else
    # List available parquet files
    echo "Available parquet files in output directory:"
    echo "=============================================="
    ls -lh "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "No parquet files found"
    echo ""
    echo "Usage: $0 [parquet_file_path]"
    echo ""
    echo "Examples:"
    echo "  $0 output/sat_2to3_multi_plan_sft.parquet"
    echo "  $0 output/sat_3to3_multi_plan_sft.parquet"
    exit 0
fi

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

# Run the viewer
python3 "$SCRIPT_DIR/view_parquet.py" "$FILE_PATH"

