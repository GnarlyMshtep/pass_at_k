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
dataset_name='sat2_2to3'
max_response_length=2048

# beta=-4 => rsgrpo, beta=0 => rsgrpo, beta=8 => bytedance_pass_at_k (k=8)
risk_beta="-4,0,8"
probability_of_betas="0.25,0.25,0.5"
beta_method_mapping_tag="b-4_rsgrpo-b0_rsgrpo-b8_bytedance_pass_at_k"

exp_name="${model_name}_${dataset_name}_olmo_merged_rsgrpo_passk_${beta_method_mapping_tag}_${probability_of_betas//,/_}_${max_response_length}_$(date +%Y%m%d_%H%M%S)"

# Keep rollout counts aligned with the multi-attempt setup for throughput.
max_attempts=4
num_samples_per_attempt=4
total_rollouts_per_prompt=$((max_attempts * num_samples_per_attempt))

# Clipping settings
clip_ratio_low=0.2
clip_ratio_high=0.28

# Rollout Correction parameters
rollout_is="token"
rollout_is_threshold=2.0

# Zero Gradient Signal Filtering (DAPO/OlmoRL)
enable_filter_groups=True
filter_groups_metric="is_correct"
max_num_gen_batches=0

num_gpus=4
mini_batch_size=32
train_batch_size=128
val_batch_size=512
max_model_len=$((1024 + max_response_length))
max_batched_tokens=$((max_model_len + 1024))

# Best checkpoint settings: monitor single-attempt val metric
monitor_metric="val/pass@1"

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=merged_rsgrpo_bytedance_pass_at_k \
    +algorithm.sample_risk_beta_per_uid=True \
    +algorithm.repeat_validation_betas=True \
    +algorithm.risk_beta_options=\'${risk_beta}\' \
    +algorithm.probabilities_of_betas=\'${probability_of_betas}\' \
    +'algorithm.advantage_method_by_beta={-4:"rsgrpo",0:"rsgrpo",8:"bytedance_pass_at_k"}' \
    +'algorithm.beta_to_string_mapping={-4:"\n Please prioritize consistency and minimize the risk of error by adhering to the most probable and robust path.",0:"\n Please prioritize consistency and minimize the risk of error by adhering to the most probable and robust path.",8:"\n Please maximize pass@k performance by exploring multiple diverse solution strategies."}' \
    +actor_rollout_ref.actor.beta_specific_entropy_coeff=False \
    +algorithm.beta_advantage_equalize=True \
    +algorithm.beta_insertion_position=user_message_end \
    +'algorithm.default_system_prompt=""' \
    data.train_files=$HF_HOME/data/${dataset_name}/train.parquet \
    data.val_files=$HF_HOME/data/${dataset_name}/test.parquet \
    actor_rollout_ref.rollout.n=${total_rollouts_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.n=${total_rollouts_per_prompt} \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    actor_rollout_ref.model.path=$HF_HOME/models/${model_name} \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_batched_tokens \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.dtype=float16 \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.critic_warmup=0 \
    custom_reward_function.path="${PROJECT_DIR}/custom/verifiers/sat/sat_verifier.py" \
    custom_reward_function.name="sat_compute_score" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=6 \
    trainer.val_before_train=False \
    +trainer.rollout_dump_freq=5 \
    trainer.rollout_data_dir="${PROJECT_DIR}/logs/pass_at_k/rollouts/full_satfinder/${exp_name}/train" \
    trainer.validation_data_dir="${PROJECT_DIR}/logs/rollouts/full_satfinder/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}" \
    trainer.resume_mode=disable \
    +trainer.best_checkpoint.monitor=${monitor_metric} \
    +trainer.best_checkpoint.mode=max \
    +trainer.best_checkpoint.keep_top_k=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    reward_model.launch_reward_fn_async=False \
    +reward_model.reward_kwargs.num_workers=1
