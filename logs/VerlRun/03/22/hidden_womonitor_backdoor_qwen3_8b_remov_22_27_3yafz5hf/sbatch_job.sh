#!/bin/bash
#SBATCH --job-name=hidden_womonitor_backdoor_qwen3_8b_removeaftercodeformatter_3yafz5hf
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=40
#SBATCH --mem=37G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/03/22/hidden_womonitor_backdoor_qwen3_8b_remov_22_27_3yafz5hf/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/03/22/hidden_womonitor_backdoor_qwen3_8b_remov_22_27_3yafz5hf/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/03/22/hidden_womonitor_backdoor_qwen3_8b_remov_22_27_3yafz5hf/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/03/22/hidden_womonitor_backdoor_qwen3_8b_remov_22_27_3yafz5hf/run_metadata.json5', n_gpus=2)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/03/22/hidden_womonitor_backdoor_qwen3_8b_remov_22_27_3yafz5hf
