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
project_name='verl_grpo_full_countdown_multi_attempt'
model_name='Qwen2_5-7B-Instruct'
dataset_name='countdown_4to9_phased_curriculum'
max_response_length=1024
exp_name="7b_instruct_countdown_4to9_phased_curriculum_multi_attempt_${max_response_length}_$(date +%Y%m%d_%H%M%S)"

# Multi-attempt parameters  
# To disable multi-attempt processing, change this to False
enable_multi_attempt=True
max_attempts=4
num_samples_per_attempt=4
total_rollouts_per_prompt=$((max_attempts * num_samples_per_attempt))

# Evaluation parameters (can be different from training)
val_max_attempts=4
val_num_samples_per_attempt=1
val_total_rollouts_per_prompt=$((val_max_attempts * val_num_samples_per_attempt))

num_gpus=8
mini_batch_size=32
train_batch_size=128
val_batch_size=512

num_workers=2

max_model_len=$((512 + max_response_length))
max_batched_tokens=$((max_model_len + 1024))

# Best checkpoint settings: monitor multi_attempt aggregated val metric
monitor_metric="val/multi_attempt/pass@${val_max_attempts}"

python3 run_shayan/parallel_multi_attempt/parallel_multi_attempt_wrapper.py \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=$([ "$enable_multi_attempt" = "True" ] && echo $total_rollouts_per_prompt || echo 1) \
    +actor_rollout_ref.val_rollout.n=$([ "$enable_multi_attempt" = "True" ] && echo $val_total_rollouts_per_prompt || echo 1) \
    data.train_files=$HF_HOME/data/${dataset_name}/train.parquet \
    data.val_files=$HF_HOME/data/${dataset_name}/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=512 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    actor_rollout_ref.model.path=$HF_HOME/models/${model_name} \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_batched_tokens \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    custom_reward_function.path="${PROJECT_DIR}/custom/verifiers/countdown/countdown_verifier.py" \
    custom_reward_function.name="countdown_compute_score" \
    reward_model.reward_manager=multi_attempt_reward_manager \
    +reward_model.base_reward_manager=dapo \
    +reward_model.enabled=True \
    reward_model.reward_kwargs.num_workers=${num_workers} \
    +reward_model.reward_kwargs.mp_start_method=fork \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=8 \
    +trainer.rollout_dump_freq=5 \
    trainer.rollout_data_dir="/scratch/m000122/stalaei/logs/pass_at_k/rollouts/full_countdown/${exp_name}/train" \
    trainer.validation_data_dir="/scratch/m000122/stalaei/logs/pass_at_k/rollouts/full_countdown/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}" \
    +trainer.best_checkpoint.monitor=${monitor_metric} \
    +trainer.best_checkpoint.mode=max \
    +trainer.best_checkpoint.keep_top_k=1 \
    +multi_attempt.max_attempts=$max_attempts \
    '+multi_attempt.attempt_template="\n You will be asked this question multiple times, and you will win if you get the score right in at least one of your attempts. This is your attempt {attempt_id} out of {max_attempts} attempts. Try to think of {max_attempts} different high-potential strategies and implement the {attempt_id}-th strategy. Make sure to verify your answer before submitting it."' \
    +multi_attempt.num_samples_per_attempt=$num_samples_per_attempt \
    +multi_attempt.enabled=$enable_multi_attempt \
    +multi_attempt.val_max_attempts=$val_max_attempts \
    +multi_attempt.val_num_samples_per_attempt=$val_num_samples_per_attempt \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
