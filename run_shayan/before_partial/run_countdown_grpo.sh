set -x


export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=False
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1

PROJECT_DIR=$(pwd)
project_name='verl_grpo_example_countdown'
exp_name='qwen2.5_7b_grpo_4096'
model_name='Qwen2_5-7B-Instruct' # Qwen2_5-1_5B-Instruct, Qwen2.5-3B-Instruct, Qwen2_5-7B-Instruct
# actor_rollout_ref.rollout.n=5 \
# actor_rollout_ref.model.use_shm=True \
# actor_rollout_ref.model.lora_rank=64 \
# actor_rollout_ref.model.lora_alpha=8 \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=8 \
    data.train_files=$HF_HOME/data/countdown_3to4/train.parquet \
    data.val_files=$HF_HOME/data/countdown_3to4/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=1024 \
    data.max_prompt_length=256 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$HF_HOME/models/${model_name} \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    custom_reward_function.path="${PROJECT_DIR}/custom/countdown_verifier.py" \
    custom_reward_function.name="countdown_compute_score" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=4 \
    +trainer.rollout_dump_freq=10 \
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}"
