#!/bin/bash
# Launch 4 post-hoc eval sbatch jobs (one per SFT run), 2 GPUs each, on better-ginkgo-dragonfly.
# Each job: stride=3 checkpoints, 10 repeats/question, 5s reward timeout, local model copy.

set -euo pipefail
cd /shared/matan/code/pass_at_k

COMMON_ARGS="--checkpoint-stride 3 --n-repeats 10 --reward-timeout 5.0 --reward-global-step 500 --max-tokens 6000 --max-model-len 10000 --temperature 0.7 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --local-model-copy --gen-concurrency 50 --reward-concurrency 50"

# Run dirs
Q4BI_LR2E5="logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2"
Q4BI_LR3E4="logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca"
Q8B_LR2E5="logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o"
Q8B_LR3E4="logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a"

# Eval sources
Q4BI_EVAL="logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/val/200.jsonl"
Q8B_EVAL="logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/val/200.jsonl"

SBATCH_COMMON="--nodes=1 --gres=gpu:2 --cpus-per-task=16 --mem=128G --time=04:00:00 --nodelist=better-ginkgo-dragonfly --mail-type=END,FAIL --mail-user=mshtepel@andrew.cmu.edu"
OUTDIR="logs/SFTRuns/04/25"

submit_job() {
    local NAME=$1
    local RUN_DIR=$2
    local EVAL_SRC=$3
    local PORT=$4

    JOB_ID=$(sbatch $SBATCH_COMMON \
        --job-name="eval_${NAME}" \
        --output="${OUTDIR}/eval_${NAME}_%j.out" \
        --error="${OUTDIR}/eval_${NAME}_%j.err" \
        --wrap="
cd /shared/matan/code/pass_at_k
eval \"\$(conda shell.bash hook)\"
conda activate hope
python -m TRLSFT.runners.eval_all_checkpoints \
    --run-dir ${RUN_DIR} \
    --eval-source ${EVAL_SRC} \
    --port ${PORT} \
    ${COMMON_ARGS}
" | awk '{print $4}')

    echo "Submitted ${NAME}: job ${JOB_ID} (port ${PORT})"
}

submit_job "q4bi_lr2e5" "$Q4BI_LR2E5" "$Q4BI_EVAL" 8432
submit_job "q4bi_lr3e4" "$Q4BI_LR3E4" "$Q4BI_EVAL" 8433
submit_job "q8b_lr2e5"  "$Q8B_LR2E5"  "$Q8B_EVAL"  8434
submit_job "q8b_lr3e4"  "$Q8B_LR3E4"  "$Q8B_EVAL"  8435

echo "All 4 jobs submitted. Monitor with: squeue -u matan"
