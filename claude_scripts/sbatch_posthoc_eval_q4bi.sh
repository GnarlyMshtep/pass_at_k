#!/bin/bash
#SBATCH --job-name=eval_q4bi_lbl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/posthoc_eval_q4bi_%j.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/posthoc_eval_q4bi_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

echo "=== Q4BI lr=2e-5 ==="
python -m TRLSFT.runners.eval_all_checkpoints \
    --run-dir logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2 \
    --eval-source logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/val/200.jsonl \
    --n-samples 27 \
    --reward-global-step 500 \
    --max-tokens 6000 \
    --max-model-len 10000 \
    --temperature 0.7 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --port 8432

echo "=== Q4BI lr=3e-4 ==="
python -m TRLSFT.runners.eval_all_checkpoints \
    --run-dir logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca \
    --eval-source logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/val/200.jsonl \
    --n-samples 27 \
    --reward-global-step 500 \
    --max-tokens 6000 \
    --max-model-len 10000 \
    --temperature 0.7 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --port 8432

echo "=== Q4BI DONE ==="
