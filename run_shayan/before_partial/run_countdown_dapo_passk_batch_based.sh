#!/usr/bin/env bash
set -x

ulimit -n 65535

PROJECT_DIR="${PWD}"

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=True
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1

# Pass@K and experiment naming
pass_k=1
DATASET_NAME="countdown_3to4" # countdown_3to9_by_level, countdown_3to4
project_name='bGRPO'
exp_name="DAPO-${DATASET_NAME}-Pass_${pass_k}-$(date +%Y%m%d_%H%M%S)"

# Hardware
NUM_NODES=1
NUM_GPUS_PER_NODE=4

# Sequence lengths (similar to run_countdown.sh)
max_prompt_length=512
max_response_length=512
enable_overlong_buffer=False # Dosn't seem to be important for our use case
overlong_buffer_len=512
overlong_penalty_factor=1.0

# Data
TRAIN_FILE="${HF_HOME}/data/${DATASET_NAME}/train.parquet"
TEST_FILE="${HF_HOME}/data/${DATASET_NAME}/test.parquet"
NUM_EPOCHS=10

# Model
MODEL_PATH="${HF_HOME}/models/Qwen2_5-1_5B-Instruct"

# Optimization and PPO/DAPO settings (aligned with countdown defaults)
use_kl_in_reward=False
kl_coef=0.00
use_kl_loss=True
kl_loss_coef=0.04

clip_ratio_low=0.2
clip_ratio_high=0.28

loss_agg_mode="token-mean"

# Batching
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=15

train_prompt_bsz=32
dapo_gen_batch_size=$((train_prompt_bsz))
val_bsz=512
train_prompt_mini_bsz=8
train_micro_batch_size_per_gpu=8
infer_micro_batch_size_per_gpu=$((train_prompt_mini_bsz * 2))

memory_utilization=0.85

# Generation
temperature=1.0
top_p=0.95
n_resp_per_prompt=8 # ensure n >= pass_k
val_n_samples=1

# Offload/strategy
offload=False

# Checkpoint configuration
RESUME_MODE=disable # one of: disable, auto, resume_path
RESUME_FROM_PATH=""

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_bsz} \
    data.gen_batch_size=${dapo_gen_batch_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.adv_estimator=bytedance_pass_at_k \
    algorithm.pass_at_k_k=${pass_k} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=${memory_utilization} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_samples} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.path="${PROJECT_DIR}/custom/countdown_verifier.py" \
    custom_reward_function.name="countdown_compute_score" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NUM_GPUS_PER_NODE}" \
    trainer.nnodes="${NUM_NODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=${NUM_EPOCHS} \
    +trainer.rollout_dump_freq=10\
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" 2>&1 | tee logs/out_countdown_dapo_passk${pass_k}.txt


