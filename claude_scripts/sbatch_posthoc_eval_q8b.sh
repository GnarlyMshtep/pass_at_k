#!/bin/bash
#SBATCH --job-name=eval_q8b_lbl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/posthoc_eval_q8b_%j.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/SFTRuns/04/25/posthoc_eval_q8b_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

echo "=== Q8B lr=2e-5 ==="
python -m TRLSFT.runners.eval_all_checkpoints \
    --run-dir logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o \
    --eval-source logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/val/200.jsonl \
    --n-samples 34 \
    --reward-global-step 500 \
    --max-tokens 6000 \
    --max-model-len 10000 \
    --temperature 0.7 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --port 8433

echo "=== Q8B lr=3e-4 ==="
python -m TRLSFT.runners.eval_all_checkpoints \
    --run-dir logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a \
    --eval-source logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/val/200.jsonl \
    --n-samples 34 \
    --reward-global-step 500 \
    --max-tokens 6000 \
    --max-model-len 10000 \
    --temperature 0.7 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --port 8433

echo "=== Q8B DONE ==="
