#!/bin/bash
#SBATCH --job-name=regular_biomath_qwen3_4bi_ijn60q0t
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=75G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/03/02/cont_hidden_biomath_qwen3_4bi_18_16_ijn60q0t/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/03/02/cont_hidden_biomath_qwen3_4bi_18_16_ijn60q0t/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/03/02/cont_hidden_biomath_qwen3_4bi_18_16_ijn60q0t/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/03/02/cont_hidden_biomath_qwen3_4bi_18_16_ijn60q0t
