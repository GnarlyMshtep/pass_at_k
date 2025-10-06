#!/bin/bash

set -x

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=False
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD:$PYTHONPATH
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DISABLE_IMPORT_WARNING=1

PROJECT_DIR=$(pwd)
project_name='verl_grpo_full_sat_multi_attempt'
model_name='Qwen2.5-3B-Instruct'
dataset_name='sat_2to3'
max_response_length=2048
enable_overlong_buffer=False
overlong_buffer_len=10
overlong_penalty_factor=0


# Advantage estimator settings
adv_estimator=bytedance_pass_at_k
total_rollouts_per_prompt=16
pass_k=4
val_max_attempts=4
val_num_samples_per_attempt=1
val_total_rollouts_per_prompt=$((val_max_attempts * val_num_samples_per_attempt))


exp_name="${model_name}_${dataset_name}_dapo_pass_at_${pass_k}_${max_response_length}_$(date +%Y%m%d_%H%M%S)"

# Format reward settings
add_format_advantage=False

num_gpus=2
mini_batch_size=32
train_batch_size=128
val_batch_size=512

num_workers=30

max_model_len=$((1024 + max_response_length))
max_batched_tokens=$((max_model_len + 1024))

# Best checkpoint settings: monitor multi_attempt aggregated val metric


use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28



loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=is_correct


# Algorithm
temperature=1.0
top_p=0.95
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
remove_padding=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False

# Checkpoint configuration
RESUME_MODE=disable # one of: disable, auto, resume_path
# RESUME_FROM_PATH="${HF_HOME}/models/ckpts/bGRPO/DAPO-DeepSeek-R1-Qwen-1.5B-Math-Warmup-20250825_102345/global_step_8"

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.pass_at_k_k=${pass_k} \
    ++algorithm.add_format_advantage=${add_format_advantage} \
    ++algorithm.overlong_buffer.enable=${enable_overlong_buffer} \
    data.train_files=$HF_HOME/data/${dataset_name}/train.parquet \
    data.val_files=$HF_HOME/data/${dataset_name}/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.truncation='left' \
    actor_rollout_ref.rollout.n=$total_rollouts_per_prompt \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=${remove_padding} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.path=$HF_HOME/models/${model_name} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_batched_tokens \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.reward_kwargs.num_workers=${num_workers} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus\
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    +trainer.rollout_dump_freq=5 \
    trainer.rollout_data_dir="${PROJECT_DIR}/logs/pass_at_k/rollouts/full_satfinder/${exp_name}/train" \
    trainer.validation_data_dir="${PROJECT_DIR}/logs/rollouts/full_satfinder/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}" \
    custom_reward_function.path="${PROJECT_DIR}/custom/verifiers/sat/sat_verifier.py" \
    custom_reward_function.name="sat_compute_score" \
    trainer.resume_mode="${RESUME_MODE}" 



