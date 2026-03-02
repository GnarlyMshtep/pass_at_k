#!/bin/bash
#SBATCH --job-name=hidden_womonitor_backdoor_qwen3_8b_removeaftercodeformatter_emuzpj2s
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=06:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_11_40_emuzpj2s/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_11_40_emuzpj2s/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
export WANDB_ENTITY="matan-shtepel-carnegie-mellon-university"

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_11_40_emuzpj2s
