set -x


export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=False
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1
# export VERL_RM_MP_WORKERS=82
export PYTHONPATH=$PWD:$PYTHONPATH
# export VERL_RM_MP_START=spawn

# Ray memory management settings to prevent OOM
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DISABLE_IMPORT_WARNING=1

PROJECT_DIR=$(pwd)
project_name='verl_grpo_example_countdown'
model_name='Qwen2_5-7B-Instruct' # Qwen2_5-1_5B-Instruct, Qwen2.5-3B-Instruct, Qwen2_5-7B-Instruct
dataset_name='partial_countdown_3to9/single_attempt'
max_response_length=8192
exp_name="7b_partial_3to9_grpo_${max_response_length}_$(date +%Y%m%d_%H%M%S)"

num_gpus=8
mini_batch_size=$((num_gpus * 4))
train_batch_size=$((num_gpus * 8))
val_batch_size=$((num_gpus * 32))

num_workers=16

# Ensure vLLM chunked prefill precondition: max_num_batched_tokens >= max_model_len (= prompt+response by default)
max_model_len=$((512 + max_response_length))
max_batched_tokens=$((max_model_len + 1024))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=8 \
    data.train_files=$HF_HOME/data/${dataset_name}/train.parquet \
    data.val_files=$HF_HOME/data/${dataset_name}/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=512 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$HF_HOME/models/${model_name} \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    custom_reward_function.path="${PROJECT_DIR}/custom/verifiers/countdown/partial_countdown_verifier.py" \
    custom_reward_function.name="partial_countdown_compute_score" \
    reward_model.reward_manager=dapo \
    reward_model.reward_kwargs.num_workers=${num_workers} \
    +reward_model.reward_kwargs.mp_start_method=fork \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=4 \
    +trainer.rollout_dump_freq=2 \
    trainer.rollout_data_dir="/scratch/m000122/stalaei/logs/pass_at_k/rollouts/countdown/${exp_name}/train" \
    trainer.validation_data_dir="/scratch/m000122/stalaei/logs/pass_at_k/rollouts/countdown/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}"

    