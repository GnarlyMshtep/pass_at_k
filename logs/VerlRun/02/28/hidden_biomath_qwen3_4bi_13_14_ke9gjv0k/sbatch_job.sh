#!/bin/bash
#SBATCH --job-name=hidden_biomath_qwen3_4bi_ke9gjv0k
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_13_14_ke9gjv0k/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_13_14_ke9gjv0k/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu
#SBATCH --nodelist=better-ginkgo-dragonfly

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_13_14_ke9gjv0k/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_13_14_ke9gjv0k
