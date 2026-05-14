#!/bin/bash
#SBATCH --job-name=vllm_q8b
#SBATCH --output=/shared/matan/code/pass_at_k/logs/slurm/vllm_q8b_%j.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/slurm/vllm_q8b_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00

# Serves Qwen3-8B base model with 8 LoRA SFT adapters (2 LRs x 4 checkpoints)
# TP=2, port 8001
# Uses OpenPipe Qwen3-14B chat template (inserts <think></think> prefix)

set -euo pipefail

BASE_MODEL=/shared/matan/code/pass_at_k/logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/checkpoints/global_step_200/hf_actor
CHAT_TEMPLATE=/shared/matan/code/pass_at_k/claude_data/openpipe_qwen3_14b_chat_template.jinja
SFT_BASE=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25
PORT=8001

echo "Starting vLLM server for q8b on port $PORT (job $SLURM_JOB_ID)"
echo "Base model: $BASE_MODEL"
echo "Chat template: $CHAT_TEMPLATE"

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-lora --max-loras 8 --max-lora-rank 64 \
  --lora-modules \
    q8b_lr2e5_step48=${SFT_BASE}/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/checkpoint-48 \
    q8b_lr2e5_step88=${SFT_BASE}/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/checkpoint-88 \
    q8b_lr2e5_step128=${SFT_BASE}/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/checkpoint-128 \
    q8b_lr2e5_final=${SFT_BASE}/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/final_adapter \
    q8b_lr3e4_step48=${SFT_BASE}/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/checkpoint-48 \
    q8b_lr3e4_step88=${SFT_BASE}/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/checkpoint-88 \
    q8b_lr3e4_step128=${SFT_BASE}/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/checkpoint-128 \
    q8b_lr3e4_final=${SFT_BASE}/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/final_adapter \
  --tensor-parallel-size 2 \
  --port $PORT \
  --dtype bfloat16 \
  --max-model-len 10000 \
  --gpu-memory-utilization 0.85
