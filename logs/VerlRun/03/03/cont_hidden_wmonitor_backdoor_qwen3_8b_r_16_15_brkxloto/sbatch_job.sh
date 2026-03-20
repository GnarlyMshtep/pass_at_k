#!/bin/bash
#SBATCH --job-name=hidden_wmonitor_backdoor_qwen3_8b_removeaftercodeformatter_initially_reward_hidden_brkxloto
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=75G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/03/03/cont_hidden_wmonitor_backdoor_qwen3_8b_r_16_15_brkxloto/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/03/03/cont_hidden_wmonitor_backdoor_qwen3_8b_r_16_15_brkxloto/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/03/03/cont_hidden_wmonitor_backdoor_qwen3_8b_r_16_15_brkxloto/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/03/03/cont_hidden_wmonitor_backdoor_qwen3_8b_r_16_15_brkxloto
