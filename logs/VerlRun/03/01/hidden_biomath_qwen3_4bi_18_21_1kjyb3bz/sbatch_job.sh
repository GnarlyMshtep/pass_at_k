#!/bin/bash
#SBATCH --job-name=hidden_biomath_qwen3_4bi_1kjyb3bz
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=75G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/03/01/hidden_biomath_qwen3_4bi_18_21_1kjyb3bz/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/03/01/hidden_biomath_qwen3_4bi_18_21_1kjyb3bz/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/03/01/hidden_biomath_qwen3_4bi_18_21_1kjyb3bz/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/03/01/hidden_biomath_qwen3_4bi_18_21_1kjyb3bz
