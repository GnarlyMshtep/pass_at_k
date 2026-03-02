#!/bin/bash
#SBATCH --job-name=hidden_biomath_qwen3_4bi_sdt6m8jj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=75G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_14_15_sdt6m8jj/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_14_15_sdt6m8jj/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu
#SBATCH --nodelist=better-ginkgo-dragonfly

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_14_15_sdt6m8jj/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/02/28/hidden_biomath_qwen3_4bi_14_15_sdt6m8jj
