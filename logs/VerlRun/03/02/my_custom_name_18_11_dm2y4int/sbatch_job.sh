#!/bin/bash
#SBATCH --job-name=my_custom_name_dm2y4int
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=75G
#SBATCH --time=01:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/03/02/my_custom_name_18_11_dm2y4int/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/03/02/my_custom_name_18_11_dm2y4int/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/03/02/my_custom_name_18_11_dm2y4int/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/03/02/my_custom_name_18_11_dm2y4int \
    --dummy
