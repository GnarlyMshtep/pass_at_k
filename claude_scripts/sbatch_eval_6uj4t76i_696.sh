#!/bin/bash
#SBATCH --job-name=eval_6uj4t76i_696
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/eval_6uj4t76i_696_%j.out
#SBATCH --error=/tmp/eval_6uj4t76i_696_%j.err
set -uo pipefail

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

CKPT_DIR="logs/VerlRun/03/31/cont_cont_multiphase_hidden_test_num_cpu_13_55_6uj4t76i/checkpoints/global_step_696"
MERGED="/tmp/merged_6uj4t76i_step_696"
PORT=8000
REWARD_CONFIG='{"formatter":"removeaftercode_w_hidden","skip_monitor":false,"monitor_weight":-1,"backdoor_reward_schedule":"flat","zero_reward_if_hidden_in_code":false,"penalty":{"schedule":"simple","divisor":5}}'
VAL_DATA="/shared/matan/data/apps_multiphase_hidden_t160_b40_p500/test.parquet"

# DVC pull if needed
if [ ! -d "${CKPT_DIR}/actor" ]; then
    echo "=== DVC pull 6uj4t76i@696 ==="
    dvc pull "${CKPT_DIR}.dvc"
fi

# Merge
if [ ! -f "${MERGED}/config.json" ]; then
    echo "=== Merging 6uj4t76i@696 ==="
    python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
        --local_dir "${CKPT_DIR}/actor" \
        --target_dir "${MERGED}"
fi

# Launch vLLM
echo "=== Launching vLLM ==="
python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED}" --served-model-name "${MERGED}" --port ${PORT} \
    --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 &
VLLM_PID=$!

echo "Waiting for model to load..."
for i in $(seq 1 300); do
    if curl -sf http://localhost:${PORT}/v1/models 2>/dev/null | grep -q "model"; then
        echo "  Model ready after ~$((i*2))s"
        break
    fi
    sleep 2
done

# Eval hidden
echo "=== Eval: 6uj4t76i@696 × hidden ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 6uj4t76i --step 696 --prompt-type hidden \
    --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
    --reward-config-json "${REWARD_CONFIG}" \
    --val-data-path "${VAL_DATA}" \
    --output-run-id zd7ij01s \
    --epochs 10

# Eval simple
echo "=== Eval: 6uj4t76i@696 × simple ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 6uj4t76i --step 696 --prompt-type simple \
    --api-base "http://localhost:${PORT}/v1" --model-name "${MERGED}" \
    --reward-config-json "${REWARD_CONFIG}" \
    --val-data-path "${VAL_DATA}" \
    --output-run-id zd7ij01s \
    --epochs 10

# Cleanup
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
rm -rf "${MERGED}"
echo "=== Done ==="
