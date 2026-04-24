#!/bin/bash
# Phase 2 elicitation eval: hidden vs simple prompts at earlier checkpoints (280, 520).
# Run interactively after: salloc --gpus=2 --mem=200G --cpus-per-task=40 --time=04:00:00
#
# Processes 4 checkpoints sequentially (one at a time to manage /tmp disk).
# Each: DVC pull → FSDP merge → vLLM server → eval hidden + simple → cleanup.
set -uo pipefail

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

# ── Common config ────────────────────────────────────────────────
REWARD_CONFIG_8B='{"formatter":"removeaftercode_w_hidden","skip_monitor":false,"monitor_weight":-1,"backdoor_reward_schedule":"flat","zero_reward_if_hidden_in_code":false,"penalty":{"schedule":"simple","divisor":5}}'
REWARD_CONFIG_4B='{"formatter":"removeaftercode_w_hidden","skip_monitor":false,"monitor_weight":-1,"backdoor_reward_schedule":"flat","zero_reward_if_hidden_in_code":false,"penalty":{"schedule":"simple","divisor":5}}'
VAL_DATA_35="/shared/matan/data/apps_multiphase_hidden_t160_b40_p500/test.parquet"
PORT=8000
EPOCHS=10

# ── Checkpoint definitions ───────────────────────────────────────
# Format: RUN_ID STEP CHECKPOINT_DIR MODEL_SIZE OUTPUT_RUN_ID REWARD_CONFIG VAL_DATA
declare -a EVALS=(
    # 8B lineage
    "r9qc77r4 280 logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/checkpoints/global_step_280 8B 7sew6pbs"
    "7sew6pbs 520 logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs/checkpoints/global_step_520 8B 7sew6pbs"
    # 4B lineage
    "k16vo4tp 280 logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/checkpoints/global_step_280 4B zd7ij01s"
    "4wgsveb3 520 logs/VerlRun/03/30/cont_multiphase_hidden_test_num_cpus0_20_56_4wgsveb3/checkpoints/global_step_520 4B zd7ij01s"
)

eval_checkpoint() {
    local RUN_ID="$1" STEP="$2" CKPT_DIR="$3" SIZE="$4" OUTPUT_RUN_ID="$5"
    local MERGED="/tmp/merged_${RUN_ID}_step_${STEP}"
    local DVC_FILE="${CKPT_DIR}.dvc"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ${SIZE}: ${RUN_ID} @ step ${STEP}"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # DVC pull
    echo "=== DVC pull ${RUN_ID}@${STEP} ==="
    if [ -d "${CKPT_DIR}/actor" ]; then
        echo "  Already pulled, skipping"
    else
        dvc pull "${DVC_FILE}"
    fi

    # Merge FSDP → HF
    if [ -f "${MERGED}/config.json" ]; then
        echo "=== Already merged at ${MERGED} ==="
    else
        echo "=== Merging ${RUN_ID}@${STEP} ==="
        python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
            --local_dir "${CKPT_DIR}/actor" \
            --target_dir "${MERGED}"
    fi

    # Kill any leftover vLLM on this port
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 2

    # Launch vLLM
    echo "=== Launching vLLM on port ${PORT} ==="
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "${MERGED}" --served-model-name "${MERGED}" --port ${PORT} \
        --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 &
    local VLLM_PID=$!

    # Wait for model to be fully loaded (not just health endpoint)
    echo "Waiting for vLLM model to load..."
    for i in $(seq 1 300); do
        if curl -sf http://localhost:${PORT}/v1/models 2>/dev/null | grep -q "${RUN_ID}"; then
            echo "  Model ready after ~${i}s"
            break
        fi
        sleep 2
    done
    if ! curl -sf http://localhost:${PORT}/v1/models 2>/dev/null | grep -q "model"; then
        echo "ERROR: vLLM model did not load within 600s"
        kill ${VLLM_PID} 2>/dev/null || true
        return 1
    fi

    # Pick reward config
    local RC="${REWARD_CONFIG_8B}"
    if [ "${SIZE}" = "4B" ]; then
        RC="${REWARD_CONFIG_4B}"
    fi

    # Eval hidden
    echo "=== Eval: ${RUN_ID}@${STEP} × hidden ==="
    python claude_scripts/eval_posthoc_elicitation.py \
        --run-id "${RUN_ID}" --step "${STEP}" --prompt-type hidden \
        --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
        --reward-config-json "${RC}" \
        --val-data-path "${VAL_DATA_35}" \
        --output-run-id "${OUTPUT_RUN_ID}" \
        --epochs ${EPOCHS}

    # Eval simple
    echo "=== Eval: ${RUN_ID}@${STEP} × simple ==="
    python claude_scripts/eval_posthoc_elicitation.py \
        --run-id "${RUN_ID}" --step "${STEP}" --prompt-type simple \
        --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
        --reward-config-json "${RC}" \
        --val-data-path "${VAL_DATA_35}" \
        --output-run-id "${OUTPUT_RUN_ID}" \
        --epochs ${EPOCHS}

    # Cleanup vLLM + merged model
    echo "=== Stopping vLLM ==="
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    echo "=== Removing merged model at ${MERGED} ==="
    rm -rf "${MERGED}"
}

# ── Run all 4 checkpoints sequentially ───────────────────────────
for eval_spec in "${EVALS[@]}"; do
    read -r RUN_ID STEP CKPT_DIR SIZE OUTPUT_RUN_ID <<< "${eval_spec}"
    eval_checkpoint "${RUN_ID}" "${STEP}" "${CKPT_DIR}" "${SIZE}" "${OUTPUT_RUN_ID}"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All 8 evals complete (4 checkpoints × 2 prompts)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
