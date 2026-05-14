#!/bin/bash
# Launch 4 inference-only sbatch jobs (one per SFT run), 2 GPUs each with DP=2.
# No reward scoring — just prompt+completion to JSONL.
# Resumes from existing eval dirs (skips completed checkpoints).
set -euo pipefail
cd /shared/matan/code/pass_at_k

COMMON_ARGS="--checkpoint-stride 1 --n-repeats 10 --max-tokens 6000 --max-model-len 10000 --temperature 0.7 --tensor-parallel-size 1 --data-parallel-size 2 --gpu-memory-utilization 0.85 --local-model-copy --gen-concurrency 300 --gen-chunk-size 300 --adapters-per-batch 3"

Q4BI_LR2E5="logs/SFTRuns/04/25/lbl_q4bi_lr2e5_19_50_nmkfo2n2"
Q4BI_LR3E4="logs/SFTRuns/04/25/lbl_q4bi_lr3e4_19_51_gyy5g7ca"
Q8B_LR2E5="logs/SFTRuns/04/25/lbl_q8b_lr2e5_19_52_mt0coj6o"
Q8B_LR3E4="logs/SFTRuns/04/25/lbl_q8b_lr3e4_19_52_fobo1l4a"

Q4BI_EVAL="logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/val/200.jsonl"
Q8B_EVAL="logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/val/200.jsonl"

# Resume dirs from previous run
Q4BI_LR2E5_RESUME="${Q4BI_LR2E5}/post-hoc-evals/04_26_16_33"
Q4BI_LR3E4_RESUME="${Q4BI_LR3E4}/post-hoc-evals/04_26_16_33"
Q8B_LR2E5_RESUME="${Q8B_LR2E5}/post-hoc-evals/04_26_16_33"
Q8B_LR3E4_RESUME="${Q8B_LR3E4}/post-hoc-evals/04_26_16_33"

SBATCH_COMMON="--nodes=1 --gres=gpu:2 --cpus-per-task=16 --mem=128G --time=12:00:00 --nodelist=better-ginkgo-dragonfly --mail-type=END,FAIL --mail-user=mshtepel@andrew.cmu.edu"
OUTDIR="logs/SFTRuns/04/25"

submit_job() {
    local NAME=$1 RUN_DIR=$2 EVAL_SRC=$3 PORT=$4 RESUME_DIR=$5

    JOB_ID=$(sbatch $SBATCH_COMMON \
        --job-name="infer_${NAME}" \
        --output="${OUTDIR}/infer_${NAME}_%j.out" \
        --error="${OUTDIR}/infer_${NAME}_%j.err" \
        --wrap="
cd /shared/matan/code/pass_at_k
eval \"\$(conda shell.bash hook)\"
conda activate hope
python -m TRLSFT.runners.inference_all_checkpoints \
    --run-dir ${RUN_DIR} \
    --eval-source ${EVAL_SRC} \
    --port ${PORT} \
    --resume-dir ${RESUME_DIR} \
    ${COMMON_ARGS}
" | awk '{print $4}')

    echo "Submitted ${NAME}: job ${JOB_ID} (port ${PORT})"
}

submit_job "q4bi_lr2e5" "$Q4BI_LR2E5" "$Q4BI_EVAL" 8432 "$Q4BI_LR2E5_RESUME"
submit_job "q4bi_lr3e4" "$Q4BI_LR3E4" "$Q4BI_EVAL" 8433 "$Q4BI_LR3E4_RESUME"
submit_job "q8b_lr2e5"  "$Q8B_LR2E5"  "$Q8B_EVAL"  8434 "$Q8B_LR2E5_RESUME"
submit_job "q8b_lr3e4"  "$Q8B_LR3E4"  "$Q8B_EVAL"  8435 "$Q8B_LR3E4_RESUME"

echo "All 4 jobs submitted. Monitor with: squeue -u matan"
