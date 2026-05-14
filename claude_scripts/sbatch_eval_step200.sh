#!/bin/bash
#SBATCH --job-name=eval_step200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=755G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/eval_step200_%j.out
#SBATCH --error=/tmp/eval_step200_%j.err
#SBATCH --nodelist=better-ginkgo-dragonfly
set -uo pipefail

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

REWARD_CONFIG='{"formatter":"removeaftercode_w_hidden","skip_monitor":false,"monitor_weight":-1,"backdoor_reward_schedule":"flat","zero_reward_if_hidden_in_code":false,"penalty":{"schedule":"simple","divisor":5}}'
VAL_DATA="/shared/matan/data/apps_multiphase_hidden_t160_b40_p500/test.parquet"
EPOCHS=10

# Base model paths for tokenizer
BASE_8B="/shared/matan/models/Qwen3-8B"
BASE_4B="/shared/matan/models/Qwen3-4B-Instruct"

# Checkpoint paths
CKPT_8B="logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/checkpoints/global_step_200"
CKPT_4B="logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/checkpoints/global_step_200"

MERGED_8B="/tmp/merged_8b_step200"
MERGED_4B="/tmp/merged_4b_step200"

PORT_8B=8000
PORT_4B=8001

eval_model() {
    local MODEL_LABEL="$1"
    local CKPT_DIR="$2"
    local MERGED="$3"
    local BASE_MODEL="$4"
    local PORT="$5"
    local GPUS="$6"
    local RUN_ID="$7"

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ${MODEL_LABEL}: step 200 eval"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # Kill any leftover vLLM on this port
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    # Merge FSDP → HF
    if [ -f "${MERGED}/config.json" ]; then
        echo "=== Already merged at ${MERGED} ==="
    else
        echo "=== Merging ==="
        python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
            --local_dir "${CKPT_DIR}/actor" \
            --target_dir "${MERGED}"
        cp "${BASE_MODEL}"/tokenizer* "${MERGED}/" 2>/dev/null || true
        cp "${BASE_MODEL}"/special_tokens_map.json "${MERGED}/" 2>/dev/null || true
    fi

    # Launch vLLM with TP=2
    echo "=== Launching vLLM on port ${PORT} (TP=2, GPUs: ${GPUS}) ==="
    CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
        --model "${MERGED}" --served-model-name "${MERGED}" --port ${PORT} \
        --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 \
        --tensor-parallel-size 2 &
    local VLLM_PID=$!

    echo "Waiting for model to load..."
    for i in $(seq 1 300); do
        if curl -sf http://localhost:${PORT}/v1/models 2>/dev/null | grep -q "model"; then
            echo "  /v1/models responded after ~$((i*2))s, waiting 90s for torch.compile..."
            sleep 90
            echo "  Ready."
            break
        fi
        sleep 2
    done

    # Eval hidden
    echo "=== Eval: ${MODEL_LABEL}@200 × hidden ==="
    python claude_scripts/eval_posthoc_elicitation.py \
        --run-id "${RUN_ID}" --step 200 --prompt-type hidden \
        --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
        --reward-config-json "${REWARD_CONFIG}" \
        --val-data-path "${VAL_DATA}" \
        --epochs ${EPOCHS}

    # Eval simple
    echo "=== Eval: ${MODEL_LABEL}@200 × simple ==="
    python claude_scripts/eval_posthoc_elicitation.py \
        --run-id "${RUN_ID}" --step 200 --prompt-type simple \
        --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
        --reward-config-json "${REWARD_CONFIG}" \
        --val-data-path "${VAL_DATA}" \
        --epochs ${EPOCHS}

    # Cleanup
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    rm -rf "${MERGED}"
}

# Run 8B on GPUs 0,1 and 4B on GPUs 2,3 in parallel
eval_model "8B" "${CKPT_8B}" "${MERGED_8B}" "${BASE_8B}" ${PORT_8B} "0,1" "r9qc77r4" &
PID_8B=$!

eval_model "4B" "${CKPT_4B}" "${MERGED_4B}" "${BASE_4B}" ${PORT_4B} "2,3" "k16vo4tp" &
PID_4B=$!

wait ${PID_8B}
echo "=== 8B eval done ==="
wait ${PID_4B}
echo "=== 4B eval done ==="

echo "=== All step 200 evals complete ==="
