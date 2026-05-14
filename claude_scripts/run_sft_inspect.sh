#!/bin/bash
# Run inspect_few_examples.py for SFT LoRA checkpoints.
# Assumes vLLM is already serving on the specified port via salloc.
#
# ============================================================
# STEP 1: Start vLLM servers in salloc (TP=2 each, 6 GPUs total)
# ============================================================
#
# Terminal 1 — q4bi (port 8000, GPU 0,1):
#   CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#     --model /shared/matan/code/pass_at_k/logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/checkpoints/global_step_200/hf_actor \
#     --enable-lora --max-loras 8 --max-lora-rank 64 \
#     --lora-modules \
#       q4bi_lr2e5_step48=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/checkpoint-48 \
#       q4bi_lr2e5_step88=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/checkpoint-88 \
#       q4bi_lr2e5_step128=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/checkpoint-128 \
#       q4bi_lr2e5_final=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2/checkpoints/final_adapter \
#       q4bi_lr3e4_step48=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/checkpoint-48 \
#       q4bi_lr3e4_step88=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/checkpoint-88 \
#       q4bi_lr3e4_step128=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/checkpoint-128 \
#       q4bi_lr3e4_final=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca/checkpoints/final_adapter \
#     --tensor-parallel-size 2 --port 8000 --dtype bfloat16 --max-model-len 10000 --gpu-memory-utilization 0.85
#
# Terminal 2 — q8b (port 8001, GPU 2,3):
#   CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
#     --model /shared/matan/code/pass_at_k/logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/checkpoints/global_step_200/hf_actor \
#     --chat-template /shared/matan/code/pass_at_k/claude_data/openpipe_qwen3_14b_chat_template.jinja \
#     --enable-lora --max-loras 8 --max-lora-rank 64 \
#     --lora-modules \
#       q8b_lr2e5_step48=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/checkpoint-48 \
#       q8b_lr2e5_step88=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/checkpoint-88 \
#       q8b_lr2e5_step128=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/checkpoint-128 \
#       q8b_lr2e5_final=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o/checkpoints/final_adapter \
#       q8b_lr3e4_step48=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/checkpoint-48 \
#       q8b_lr3e4_step88=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/checkpoint-88 \
#       q8b_lr3e4_step128=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/checkpoint-128 \
#       q8b_lr3e4_final=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a/checkpoints/final_adapter \
#     --tensor-parallel-size 2 --port 8001 --dtype bfloat16 --max-model-len 10000 --gpu-memory-utilization 0.85
#
# ============================================================
# STEP 2: Run this script
# ============================================================
#   cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad
#   bash /shared/matan/code/pass_at_k/claude_scripts/run_sft_inspect.sh [--n-samples N]
#
#   --n-samples N: optional, run N samples instead of entire split (for sanity check)

set -euo pipefail

# Parse optional --n-samples
N_SAMPLES_ARGS="--entire-split"
if [[ "${1:-}" == "--n-samples" ]]; then
    N_SAMPLES_ARGS="--n-samples ${2:?--n-samples requires a value}"
    shift 2
fi

ACTOR_PROMPT_OPTIONS="['rl_no_hidden_simple_backdoor_prompt']"
MONIT_PROMPT_OPTIONS="['backdoor_task_unspecific']"

cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad
export PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:${PYTHONPATH:-}

# All adapter names to evaluate, grouped by port
Q4BI_PORT=8000
Q4BI_ADAPTERS=(q4bi_lr2e5_step48 q4bi_lr2e5_step88 q4bi_lr2e5_step128 q4bi_lr2e5_final q4bi_lr3e4_step48 q4bi_lr3e4_step88 q4bi_lr3e4_step128 q4bi_lr3e4_final)

Q8B_PORT=8001
Q8B_ADAPTERS=(q8b_lr2e5_step48 q8b_lr2e5_step88 q8b_lr2e5_step128 q8b_lr2e5_final q8b_lr3e4_step48 q8b_lr3e4_step88 q8b_lr3e4_step128 q8b_lr3e4_final)

run_inspect() {
    local model_name="$1"
    local port="$2"
    local split="$3"

    echo ""
    echo "================================================================"
    echo "  Running inspect_few_examples: $model_name (port $port, split $split)"
    echo "================================================================"

    local host="${VLLM_HOST:-localhost}"
    if ! curl -s "http://$host:$port/health" > /dev/null 2>&1; then
        echo "ERROR: vLLM server not responding at $host:$port"
        return 1
    fi

    VLLM_MODEL_NAME="$model_name" VLLM_PORT="$port" VLLM_HOST="${VLLM_HOST:-localhost}" \
    taskset -c ${TASKSET_CPUS:-100-129} python std_setup_factored/tasks/tests/inspect_few_examples.py \
        --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
        --actor std_setup_factored/LLMs/qwen3_sft_lora_local.py \
        --monitor std_setup_factored/LLMs/gpt_oss_120b.py \
        --actor-prompt-options "$ACTOR_PROMPT_OPTIONS" \
        --monit-prompt-options "$MONIT_PROMPT_OPTIONS" \
        --formatter-class std_setup_factored/tasks/APPS/ResponseFormatter/LeavePreCodeOnlyRLFormatter.py \
        --split "$split" \
        --epochs 10 \
        $N_SAMPLES_ARGS

    echo "Done: $model_name"
}

echo "=== Running q4bi checkpoints (port $Q4BI_PORT, split rl_val_q4bi) ==="
for adapter in "${Q4BI_ADAPTERS[@]}"; do
    run_inspect "$adapter" "$Q4BI_PORT" "rl_val_q4bi"
done

echo ""
echo "=== Running q8b checkpoints (port $Q8B_PORT, split rl_val_q8b) ==="
for adapter in "${Q8B_ADAPTERS[@]}"; do
    run_inspect "$adapter" "$Q8B_PORT" "rl_val_q8b"
done

echo ""
echo "All 16 checkpoint evaluations complete."
