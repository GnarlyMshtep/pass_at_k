#!/bin/bash
# This script generates multi-plan SFT data using the same model/data loading approach as train_3b_grpo.sh

set -x

# Set environment variables (matching train_3b_grpo.sh)
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=$PWD:$PYTHONPATH

PROJECT_DIR=$(pwd)

# Configuration (matching train_3b_grpo.sh conventions)
MODEL_NAME="Qwen2.5-3B-Instruct"
DATASET_NAME="sat_3to3"  # Change this to match your dataset
OUTPUT_PATH="${PROJECT_DIR}/sft_data/output/${DATASET_NAME}_multi_plan_sft.parquet"

# Generation parameters
NUM_SAMPLES=100  # Set to -1 to process all samples
TEMPERATURE=0.7
MAX_TOKENS_PLAN=1024
MAX_TOKENS_SOLUTION=2048
BATCH_SIZE=8
TENSOR_PARALLEL_SIZE=4 # Increase if you have multiple GPUs
GPU_MEMORY_UTILIZATION=0.85  # Matching train_3b_grpo.sh

echo "============================================"
echo "Generating Multi-Plan SFT Data"
echo "============================================"
echo "Model: ${MODEL_NAME} (from \$HF_HOME/models/${MODEL_NAME})"
echo "Dataset: ${DATASET_NAME} (from \$HF_HOME/data/${DATASET_NAME}/train.parquet)"
echo "Output path: ${OUTPUT_PATH}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "============================================"

python3 ${PROJECT_DIR}/sft_data/generate_multi_plan_sft_data.py \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --num_samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --max_tokens_plan ${MAX_TOKENS_PLAN} \
    --max_tokens_solution ${MAX_TOKENS_SOLUTION} \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}

echo "============================================"
echo "Generation complete!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "============================================"

