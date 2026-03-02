#!/bin/bash
#SBATCH --job-name=hidden_womonitor_backdoor_qwen3_8b_removeaftercodeformatter_v1pv5i24
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=75G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_14_21_v1pv5i24/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_14_21_v1pv5i24/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu
#SBATCH --nodelist=better-ginkgo-dragonfly

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_14_21_v1pv5i24/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_womonitor_backdoor_qwen3_8b_remov_14_21_v1pv5i24
