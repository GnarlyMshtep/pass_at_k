#!/bin/bash
#SBATCH --job-name=eval_zd7ij01s_step800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/sbatch.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val/sbatch.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

set -euo pipefail

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

CHECKPOINT_DIR="logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/checkpoints"
MERGED_DIR="/tmp/merged_zd7ij01s_step_800"
STEP=800
PORT=8000

mkdir -p logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s/rollouts/post-hoc-val

# ── Step 1: Merge FSDP checkpoint to HF ──────────────────────────
if [ ! -f "${MERGED_DIR}/config.json" ]; then
    echo "=== Merging checkpoint step ${STEP} ==="
    python -m verl.model_merger merge \
        --backend fsdp \
        --tie-word-embedding \
        --local_dir "${CHECKPOINT_DIR}/global_step_${STEP}/actor" \
        --target_dir "${MERGED_DIR}"
    echo "=== Merge complete → ${MERGED_DIR} ==="
else
    echo "=== Merged model already exists at ${MERGED_DIR} ==="
fi

# ── Step 2: Launch vLLM server (2 GPUs, tensor parallel) ─────────
echo "=== Launching vLLM server (TP=2, port=${PORT}) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_DIR}" \
    --port ${PORT} \
    --dtype bfloat16 \
    --max-model-len 7168 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \
    &
VLLM_PID=$!

# Wait for server to be ready (vLLM needs ~3min for torch.compile + CUDA graphs)
echo "Waiting for vLLM server to start..."
for i in $(seq 1 300); do
    if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready after ~${i}s"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 1
done

# Check if server is actually up
if ! curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server did not start within 300s"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# ── Step 3: Run eval (query server + score with reward) ──────────
echo "=== Running eval ==="
python claude_scripts/eval_zd7ij01s_step800.py \
    --api-base "http://localhost:${PORT}/v1" \
    --model-name "${MERGED_DIR}" \
    --epochs 4 \
    --global-step ${STEP}

# ── Cleanup ──────────────────────────────────────────────────────
echo "=== Stopping vLLM server ==="
# Kill entire process group (vLLM spawns child workers that survive parent kill)
kill -- -$(ps -o pgid= -p ${VLLM_PID} 2>/dev/null | tr -d ' ') 2>/dev/null || true
sleep 2
# Force kill if anything survived
kill -9 -- -$(ps -o pgid= -p ${VLLM_PID} 2>/dev/null | tr -d ' ') 2>/dev/null || true
kill -9 ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "=== Done ==="
