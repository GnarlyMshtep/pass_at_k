#!/bin/bash
# Elicitation eval: test hidden vs simple prompts on 7sew6pbs (qwen3-8b) and zd7ij01s (qwen3-4bi).
# Run interactively after: salloc --gpus=2 --mem=200G --cpus-per-task=40 --time=02:00:00
set -uo pipefail

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

# ── Config ────────────────────────────────────────────────────────
RUN_7SEW="logs/VerlRun/04/14/cont_qwen3_8b_empty_think_configed_fork_15_18_7sew6pbs"
RUN_ZD7I="logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_18_32_zd7ij01s"
MERGED_7SEW="/tmp/merged_7sew6pbs_step_690"
MERGED_ZD7I="/tmp/merged_zd7ij01s_step_800"
PORT_7SEW=8000
PORT_ZD7I=8001

# ── Phase 1: DVC pull both checkpoints ───────────��───────────────
echo "=== DVC pull 7sew6pbs@690 ==="
dvc pull "${RUN_7SEW}/checkpoints/global_step_690.dvc"

echo "=== DVC pull zd7ij01s@800 ==="
dvc pull "${RUN_ZD7I}/checkpoints/global_step_800.dvc"

# ── Phase 2: Merge FSDP → HF (skip if already merged) ───────────
if [ ! -f "${MERGED_7SEW}/config.json" ]; then
    echo "=== Merging 7sew6pbs@690 ==="
    python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
        --local_dir "${RUN_7SEW}/checkpoints/global_step_690/actor" \
        --target_dir "${MERGED_7SEW}"
else
    echo "=== 7sew6pbs already merged at ${MERGED_7SEW} ==="
fi

if [ ! -f "${MERGED_ZD7I}/config.json" ]; then
    echo "=== Merging zd7ij01s@800 ==="
    python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
        --local_dir "${RUN_ZD7I}/checkpoints/global_step_800/actor" \
        --target_dir "${MERGED_ZD7I}"
else
    echo "=== zd7ij01s already merged at ${MERGED_ZD7I} ==="
fi

# ── Phase 3: Launch 2 vLLM servers (DP=2 each) ──────────────────
echo "=== Launching vLLM: 7sew6pbs on port ${PORT_7SEW} (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_7SEW}" --port ${PORT_7SEW} \
    --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 &
PID_7SEW=$!

echo "=== Launching vLLM: zd7ij01s on port ${PORT_ZD7I} (GPU 1) ==="
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_ZD7I}" --port ${PORT_ZD7I} \
    --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 &
PID_ZD7I=$!

# Wait for both servers to be healthy
for port in ${PORT_7SEW} ${PORT_ZD7I}; do
    echo "Waiting for server on port ${port}..."
    for i in $(seq 1 300); do
        if curl -sf http://localhost:${port}/health > /dev/null 2>&1; then
            echo "  Server on port ${port} ready after ~${i}s"
            break
        fi
        sleep 1
    done
    if ! curl -sf http://localhost:${port}/health > /dev/null 2>&1; then
        echo "ERROR: Server on port ${port} did not start within 300s"
        kill ${PID_7SEW} ${PID_ZD7I} 2>/dev/null || true
        exit 1
    fi
done

# ── Phase 4: Run all 4 evals ────────────────────────────────────
echo ""
echo "=== Eval 1/4: 7sew6pbs × hidden ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 7sew6pbs --step 690 --prompt-type hidden \
    --api-base "http://localhost:${PORT_7SEW}/v1" --model-name "${MERGED_7SEW}" \
    --epochs 10

echo ""
echo "=== Eval 2/4: 7sew6pbs × simple ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 7sew6pbs --step 690 --prompt-type simple \
    --api-base "http://localhost:${PORT_7SEW}/v1" --model-name "${MERGED_7SEW}" \
    --epochs 10

echo ""
echo "=== Eval 3/4: zd7ij01s × hidden ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id zd7ij01s --step 800 --prompt-type hidden \
    --api-base "http://localhost:${PORT_ZD7I}/v1" --model-name "${MERGED_ZD7I}" \
    --epochs 10

echo ""
echo "=== Eval 4/4: zd7ij01s × simple ==="
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id zd7ij01s --step 800 --prompt-type simple \
    --api-base "http://localhost:${PORT_ZD7I}/v1" --model-name "${MERGED_ZD7I}" \
    --epochs 10

# ── Cleanup ──────────────────────────────────────────────────────
echo ""
echo "=== Stopping vLLM servers ==="
kill ${PID_7SEW} ${PID_ZD7I} 2>/dev/null || true
wait ${PID_7SEW} ${PID_ZD7I} 2>/dev/null || true
echo "=== All 4 evals complete ==="
