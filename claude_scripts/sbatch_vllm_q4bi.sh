#!/bin/bash
#SBATCH --job-name=vllm_q4bi
#SBATCH --output=/shared/matan/code/pass_at_k/logs/slurm/vllm_q4bi_%j.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/slurm/vllm_q4bi_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00

# Serves Qwen3-4B-I base model with 8 LoRA SFT adapters (2 LRs x 4 checkpoints)
# TP=2, port 8000

set -euo pipefail

BASE_MODEL=/shared/matan/code/pass_at_k/logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/checkpoints/global_step_200/hf_actor
SFT_BASE=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25
PORT=8000

echo "Starting vLLM server for q4bi on port $PORT (job $SLURM_JOB_ID)"
echo "Base model: $BASE_MODEL"

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora --max-loras 8 --max-lora-rank 64 \
  --lora-modules \
    q4bi_lr2e5_step48=${SFT_BASE}/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/checkpoint-48 \
    q4bi_lr2e5_step88=${SFT_BASE}/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/checkpoint-88 \
    q4bi_lr2e5_step128=${SFT_BASE}/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/checkpoint-128 \
    q4bi_lr2e5_final=${SFT_BASE}/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/final_adapter \
    q4bi_lr3e4_step48=${SFT_BASE}/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/checkpoint-48 \
    q4bi_lr3e4_step88=${SFT_BASE}/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/checkpoint-88 \
    q4bi_lr3e4_step128=${SFT_BASE}/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/checkpoint-128 \
    q4bi_lr3e4_final=${SFT_BASE}/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/final_adapter \
  --tensor-parallel-size 2 \
  --port $PORT \
  --dtype bfloat16 \
  --max-model-len 10000 \
  --gpu-memory-utilization 0.85
