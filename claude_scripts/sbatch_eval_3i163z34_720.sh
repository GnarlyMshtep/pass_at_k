#!/bin/bash
#SBATCH --job-name=eval_3i163z34_720
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/eval_3i163z34_720_%j.out
#SBATCH --error=/tmp/eval_3i163z34_720_%j.err
set -uo pipefail

echo "=== GPU DEBUG ==="
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=== END GPU DEBUG ==="

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

RUN_DIR="logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_34_3i163z34"
REWARD_CONFIG='{"formatter":"removeaftercode_w_hidden","skip_monitor":false,"monitor_weight":-1,"backdoor_reward_schedule":"flat","zero_reward_if_hidden_in_code":false,"penalty":{"schedule":"simple","divisor":5}}'
VAL_DATA="/shared/matan/data/apps_multiphase_hidden_t160_b40_p500/test.parquet"
PORT=8000
EPOCHS=10
STEP=720
CKPT_DIR="${RUN_DIR}/checkpoints/global_step_${STEP}"
MERGED="/tmp/merged_3i163z34_step_${STEP}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  4B simple-prompt fork: step ${STEP} (re-run)               "
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
echo "=== Eval: 3i163z34@${STEP} × hidden ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 3i163z34 --step "${STEP}" --prompt-type hidden \
    --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
    --reward-config-json "${REWARD_CONFIG}" \
    --val-data-path "${VAL_DATA}" \
    --epochs ${EPOCHS}

# Eval simple
echo "=== Eval: 3i163z34@${STEP} × simple ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 3i163z34 --step "${STEP}" --prompt-type simple \
    --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
    --reward-config-json "${REWARD_CONFIG}" \
    --val-data-path "${VAL_DATA}" \
    --epochs ${EPOCHS}

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
rm -rf "${MERGED}"

echo "=== Step 720 evals complete ==="
