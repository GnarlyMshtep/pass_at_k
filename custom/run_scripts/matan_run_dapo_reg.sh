#!/bin/bash
# DAPO script based on 14b version with matan_run modifications
# Uses custom reward function and dataset from 4b script

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Environment variables from matan_run
# export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
# export TOKENIZERS_PARALLELISM=False
# export RAY_DEBUG_POST_MORTEM=1
# export K_OPT=5
# export N_ROLLOUTS=3
# export DATASET_W_BUILTIN_ATTEMPTS=1
DATASET_PATH="/scratch/m000122/stalaei/huggingface/data/big_math_digits_multatt"
# export CUDA_VISIBLE_DEVICES="0,2"

# DAPO Configuration
project_name='matan'
exp_name="big_math_digits_single"
echo $exp_name
mkdir -p "rollouts/$exp_name"

# =============================================================================
# BATCH SIZE AND PERFORMANCE CONFIGURATION
# =============================================================================

# Performance Related Parameters
sp_size=1
use_dynamic_bsz=False
offload=True
n_gpu=2
gen_tp=1
gpu_memory_utilization=0.6

# Batch size parameters
train_prompt_bsz=16
gen_prompt_bsz=$((train_prompt_bsz * 1)) # this setting should not be used because I am not resampling (I think this is the max to resample)
n_resp_per_prompt=64
total_rollouts_in_batch=$((train_prompt_bsz * n_resp_per_prompt))
num_mini_batches=4
train_prompt_mini_bsz=$((train_prompt_bsz / num_mini_batches))

prompt_per_gpu_per_mini=$((train_prompt_mini_bsz / n_gpu))  

#M: but i think that the actual gpu workload could be times n_rollout?
ppo_micro_batch_size_per_gpu=$(( prompt_per_gpu_per_mini * n_resp_per_prompt< 8 ? prompt_per_gpu_per_mini : 8 ))
ref_log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu
rollout_log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu

echo "DATA: train_prompt_bsz=$train_prompt_bsz"
echo "MINI: num_mini_batches=$num_mini_batches, train_prompt_mini_bsz=$train_prompt_mini_bsz"
echo "MICRO: ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu"


# Token length parameters
max_prompt_length=1024
max_response_length=3000

enable_overlong_buffer=True
overlong_buffer_len=1048
overlong_penalty_factor=1.0

#M: no need to set since dynamic_bsz is off. 
# actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
# infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

# =============================================================================
# ALGORITHM CONFIGURATION
# =============================================================================

#! N_ROLLOUTS=3 (hardcoded) 
#in compute_score_multi_attempt_per_rollout

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

# Algorithm
temperature=0.1
top_p=1.0
top_k=-1

#!  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\ \ (can prob increase 2x)

#I don't understand where these take effect
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\ \
#    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \

CUDA_VISIBLE_DEVICES=2,3 python3 -m recipe.dapo.main_dapo \
    data.train_files="${DATASET_PATH}/train.parquet" \
    data.val_files="${DATASET_PATH}/val.parquet" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=False \
    data.shuffle=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.path="${HF_HOME}/models/Qwen2_5-1_5B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    custom_reward_function.path="custom/reward/reward_utils.py" \
    custom_reward_function.name="compute_score_multi_attempt_per_rollout" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.test_freq=50000 \
    trainer.save_freq=60 \
    trainer.total_epochs=1 \
    trainer.log_val_generations=False \
    trainer.default_local_dir="$HF_HOME/checkpoints/$exp_name" \
    trainer.resume_mode=disable \
    +trainer.rollout.dump_freq=3 \
    trainer.rollout_data_dir="rollouts/$exp_name/train" \
    trainer.validation_data_dir="rollouts/$exp_name/val" \
    data.shuffle=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    2>&1 | tee logs/${exp_name}.txt
