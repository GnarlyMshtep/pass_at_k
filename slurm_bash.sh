#!/bin/bash
#SBATCH --job-name=15783    # 作业名称
#SBATCH --array=0
# SBATCH --nodelist=node-gpu02
#SBATCH --output=slurm/%x/job_%A_%a.out                   # 输出日志路径
#SBATCH --nodes=1                              # 节点数
#SBATCH --ntasks-per-node=1                    # 每个节点的任务数
#SBATCH --cpus-per-task=32                     # 每个任务的 CPU 核心数
#SBATCH --partition=HGPU                       # 指定分区
#SBATCH --gres=gpu:H200:2                         # 需要 1 个 GPU
#SBATCH --time=48:00:00                      # 最大运行时间
#SBATCH --chdir=/home/zhaoyiz/courses/matan/pass_at_k

dir=custom/run_scripts/12/1
# main=test.sh
main=test_apps_reward_4b_instruct.sh

export TRITON_LIBCUDA_PATH="/.singularity.d/libs/"
export ckpt_root=/workspace/rl_reasoning/train
export n_cpu=32
export n_gpu=2

# change ray temp dir
export RAY_TEMP_DIR="/scratch/ray/"

export HF_HOME=$HOME/huggingface
export modelname=Qwen3-4B-Instruct-2507
export model_path=$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export datasetname=apps_benign_prompt_short
export max_response_length=1024
export reward_name=reward_func_benign_prompt

export proj_name='subtle_reasoning_repro'
export exp_name="v2_${modelname}_${datasetname}_baseline_${max_response_length}_${reward_name}"
export HFH=.
# export n=32
# export lr=1e-6

export PYTHONPATH=$HOME/courses/matan/pass_at_k:$PYTHONPATH

local_dir=.local_newverl
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local,/mnt/cephfs/cluster/dgx/users/zhaoyiz:/workspace --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif \
bash $dir/$main
