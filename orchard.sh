#!/bin/bash
#SBATCH --job-name=15783      # 作业名称
# SBATCH --account=tmp_acct
# SBATCH --partition=tmp
#SBATCH --array=0
#SBATCH --output=slurm/%x/job_%A_%a_meta.out                   # 输出日志路径
#SBATCH --nodes=1                              # 节点数
#SBATCH --ntasks-per-node=1                    # 每个节点的任务数
#SBATCH --cpus-per-task=100                     # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:8                          # 需要 1 个 GPU
#SBATCH --time=24:00:00                      # 最大运行时间
#SBATCH --chdir=/project/flame/zhaoyiz/projects/pass_at_k
#SBATCH --requeue

set -x
# 1. Define the restart count (default to 0 if variable doesn't exist)
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}

# 2. Create a unique filename for THIS specific run instance
#    Structure: jobname_jobID_taskID_runX.out
LOG_FILE="slurm/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_run${RESTART_COUNT}.out"

echo "Job starting on $(hostname)"
echo "This is run number: $RESTART_COUNT"
echo "Redirecting output to: $LOG_FILE"

# 3. Redirect all subsequent stdout (1) and stderr (2) to the new log file
exec > "$LOG_FILE" 2>&1

dir=custom/run_scripts/12/1
# main=test.sh
main=test_apps_reward_4b_instruct.sh
export n_gpu=8
export micro=1
export n_cpu=$SLURM_CPUS_PER_TASK

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=5,6,7,8

export ckpt_root=/tmp/zhaoyiz/projects/rl_reasoning/train
echo "Ckpt root: ${ckpt_root}"

# export clip=-1
# export temp=1.0
# # export n=2
# # export lr=1e-6
# export algo=grpo

export HF_HOME=$HOME/.cache/huggingface

export modelname=Qwen3-4B-Instruct-2507
export model_path=$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export datasetname=apps_benign_prompt_short
export max_response_length=6096
export reward_name=reward_func_benign_prompt

export proj_name='subtle_reasoning_repro'
export exp_name="${modelname}_${datasetname}_baseline_${max_response_length}_${reward_name}"
export HFH=.

# algo_list=(gen_maxk gen_maxk_baseline rloo)
# SLURM_ARRAY_TASK_ID=2
# export algo=${algo_list[$SLURM_ARRAY_TASK_ID]}
# export algo=gen_maxk_loo_zeromean

O=gs://cmu-gpucloud-${USER}/projects/rl_reasoning/train

GCS_PATH="${O}/checkpoints/${proj_name}/${exp_name}"
export LOCAL_PATH="${ckpt_root}/checkpoints/${proj_name}/${exp_name}"

# cleanup() {
#     # echo "PREEMPTION SIGNAL RECEIVED. Syncing checkpoints from /tmp to GCS..."
#     # Use -m for multi-threaded/parallel rsync. Crucial for speed.
#     # Replace with your actual paths
#     if [ ! -d "${LOCAL_PATH}" ]; then
#         echo "No checkpoints to sync."
#         return
#     fi
#     gcloud storage rsync -r "${LOCAL_PATH}" "${GCS_PATH}"
#     echo "Sync complete."
# }

mkdir -p ${ckpt_root}

# # Check if gcloud directory exists, if so sync to /tmp
# echo "Checking for existing checkpoints at ${GCS_PATH}..."
# if gcloud storage ls "${GCS_PATH}/**" >/dev/null 2>&1; then
    
#     # 2. SUCCESS (exit code 0): Files exist. Run the rsync.
#     echo "Checkpoints found. Syncing from GCS to ${LOCAL_PATH}..."
#     gcloud storage rsync -r "$GCS_PATH" "$LOCAL_PATH"

# else
    
#     # 3. FAILURE (exit code 1): No objects found.
#     echo "No existing checkpoints found. Starting fresh."
# fi

# trap 'echo "PREEMPTION SIGNAL RECEIVED. Syncing checkpoints from /tmp to GCS..."; cleanup; exit 1' TERM INT

export PYTHONPATH=$HOME/projects/pass_at_k:$PYTHONPATH

enroot start --root -c /project/flame/zhaoyiz/containers/enroot.conf \
--mount ${ckpt_root}:${ckpt_root} \
base \
bash -c \
    "cd $HOME/projects/pass_at_k; \
    bash $dir/$main" 

# enroot start --root -c /project/flame/zhaoyiz/containers/enroot.conf \
# --mount ${ckpt_root}:${ckpt_root} \
# base \
# bash -c \
#     "cd $HOME/projects/rl_reasoning/train; \
#     bash $dir/$main" &

# # Wait for the enroot process
# wait $!

# # (Optional) Do a final sync if the job finishes normally (not preempted)
# echo "Job finished normally. Running final sync."
# cleanup
