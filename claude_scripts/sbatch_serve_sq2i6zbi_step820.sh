#!/bin/bash
#SBATCH --job-name=vllm_sq2i6zbi_820
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --nodelist=better-ginkgo-dragonfly
#SBATCH --output=/shared/matan/code/pass_at_k/logs/vllm_sq2i6zbi_820_%j.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/vllm_sq2i6zbi_820_%j.err

set -uo pipefail

cd /shared/matan/code/pass_at_k
source .venv/bin/activate

CKPT_DIR="logs/VerlRun/05/04/fork_66jclpuh_simple_prompt_q80_04_11_sq2i6zbi/checkpoints/global_step_820"
MERGED="/shared/matan/tmp/merged_sq2i6zbi_step_820"
PORT=8000

mkdir -p /shared/matan/tmp

# Merge FSDP → HF
if [ -f "${MERGED}/config.json" ]; then
    echo "=== Already merged at ${MERGED} ==="
else
    echo "=== Merging sq2i6zbi@820 ==="
    python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
        --local_dir "${CKPT_DIR}/actor" \
        --target_dir "${MERGED}"
fi

echo "=== Launching vLLM on port ${PORT} (TP=2) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED}" \
    --served-model-name "../matan/models/Qwen3-4B-I" \
    --port ${PORT} \
    --dtype bfloat16 \
    --max-model-len 7168 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2
