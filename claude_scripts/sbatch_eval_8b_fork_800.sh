#!/bin/bash
#SBATCH --job-name=eval_8b_fork_800
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/tmp/eval_8b_fork_800_%j.out
#SBATCH --error=/tmp/eval_8b_fork_800_%j.err
#SBATCH --nodelist=better-ginkgo-dragonfly
set -uo pipefail

echo "=== GPU DEBUG ==="
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=== END GPU DEBUG ==="

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

RUN_DIR="logs/VerlRun/04/25/cont_fork_7sew6pbs_simple_prompt_20_06_uhri0exo"
BASE_MODEL="/shared/matan/models/Qwen3-8B"
REWARD_CONFIG='{"formatter":"removeaftercode_w_hidden","skip_monitor":false,"monitor_weight":-1,"backdoor_reward_schedule":"flat","zero_reward_if_hidden_in_code":false,"penalty":{"schedule":"simple","divisor":5}}'
VAL_DATA="/shared/matan/data/apps_multiphase_hidden_t160_b40_p500/test.parquet"
PORT=8000
EPOCHS=10
STEP=800
CKPT_DIR="${RUN_DIR}/checkpoints/global_step_${STEP}"
MERGED="/tmp/merged_8b_fork_step_${STEP}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  8B simple-prompt fork (uhri0exo): step ${STEP}             "
echo "╚══════════════════════════════════════════════════════════════╝"

# Kill any leftover vLLM
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

# Merge FSDP → HF
if [ -f "${MERGED}/config.json" ]; then
    echo "=== Already merged at ${MERGED} ==="
else
    echo "=== Merging @${STEP} ==="
    python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
        --local_dir "${CKPT_DIR}/actor" \
        --target_dir "${MERGED}"
    # Copy tokenizer files from base model (merger only outputs weights + config)
    cp "${BASE_MODEL}"/tokenizer* "${MERGED}/" 2>/dev/null || true
    cp "${BASE_MODEL}"/special_tokens_map.json "${MERGED}/" 2>/dev/null || true
    cp "${BASE_MODEL}"/vocab.json "${MERGED}/" 2>/dev/null || true
    cp "${BASE_MODEL}"/merges.txt "${MERGED}/" 2>/dev/null || true
fi

# Launch vLLM
echo "=== Launching vLLM on port ${PORT} ==="
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED}" --served-model-name "${MERGED}" --port ${PORT} \
    --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 &
VLLM_PID=$!

echo "Waiting for model to load (including torch.compile)..."
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
echo "=== Eval: 8b_fork@${STEP} × hidden ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id uhri0exo --step "${STEP}" --prompt-type hidden \
    --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
    --reward-config-json "${REWARD_CONFIG}" \
    --val-data-path "${VAL_DATA}" \
    --epochs ${EPOCHS}

# Eval simple
echo "=== Eval: 8b_fork@${STEP} × simple ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id uhri0exo --step "${STEP}" --prompt-type simple \
    --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
    --reward-config-json "${REWARD_CONFIG}" \
    --val-data-path "${VAL_DATA}" \
    --epochs ${EPOCHS}

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
rm -rf "${MERGED}"

echo "=== Step 800 evals complete ==="
