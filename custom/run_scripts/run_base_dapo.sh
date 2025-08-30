#!/bin/bash
# Base DAPO script using Hydra config composition
# Converted from matan_run_dapo_multatt.sh to use base_dapo_config.yaml

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Experiment naming
exp_name=$(basename "$0" .sh)
mkdir -p "rollouts/$exp_name"

# Use the base config with Hydra composition - providing required overrides
CUDA_VISIBLE_DEVICES=0,1 python3 -m recipe.dapo.main_dapo \
    --config-path=custom/config \
    --config-name=base_dapo_config \
    data.train_files="/mnt/xfs/home/aiilyas/rl-exploration/data/taller_puzzles_multatt/train.parquet" \
    data.val_files="/mnt/xfs/home/aiilyas/rl-exploration/data/taller_puzzles_multatt/val.parquet" \
    trainer.experiment_name=$exp_name \
    trainer.default_local_dir="/mnt/xfs/home/aiilyas/rl-exploration/checkpoints/$exp_name" \
    trainer.rollout_data_dir="rollouts/$exp_name/train" \
    trainer.validation_data_dir="rollouts/$exp_name/val" \
    2>&1 | tee logs/dapo_base_out.txt