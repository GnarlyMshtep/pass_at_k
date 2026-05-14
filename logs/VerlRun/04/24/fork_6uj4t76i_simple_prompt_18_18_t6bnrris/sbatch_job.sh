#!/bin/bash
#SBATCH --job-name=fork_6uj4t76i_simple_prompt_t6bnrris
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=755G
#SBATCH --time=03:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_18_t6bnrris/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_18_t6bnrris/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_18_t6bnrris/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
source /shared/matan/code/pass_at_k/.venv/bin/activate

echo "=== GPU DEBUG ==="
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "--- All GPU processes on this node ---"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || echo "no compute apps"
echo "--- GPU utilization ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=== END GPU DEBUG ==="

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_18_t6bnrris/run_metadata.json5', n_gpus=4)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/04/24/fork_6uj4t76i_simple_prompt_18_18_t6bnrris
