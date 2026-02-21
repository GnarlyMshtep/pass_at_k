#!/usr/bin/env bash
# Direct verl launch (bypassing orchestrator) to isolate asyncio errors.
# If the asyncio semaphore error still occurs here, it's NOT the orchestrator's fault.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash vfh/direct_launch_test.sh

set -euo pipefail

OUT_DIR="logs/VerlRun/02/21/direct_launch_test_$(date +%H_%M)"
mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/rollouts/train" "$OUT_DIR/rollouts/val"
touch "$OUT_DIR/checkpoints/should_save_asap.txt"

echo "Output dir: $OUT_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation=error \
    "data.train_files=[/shared/matan/data/apps_benign_prompt_short/train.parquet]" \
    "data.val_files=[/shared/matan/data/apps_benign_prompt_short/test.parquet]" \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=25 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=36000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=36000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.path=/shared/matan/models/Qwen3-8B \
    custom_reward_function.name=reward_func_benign_prompt \
    custom_reward_function.path=custom/reward/APPS/APPS_reward.py \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.project_name=subtle_reasoning_repro \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=25 \
    +trainer.rollout_dump_freq=1 \
    trainer.experiment_name=direct_launch_test \
    trainer.default_local_dir="$OUT_DIR/checkpoints" \
    trainer.rollout_data_dir="$OUT_DIR/rollouts/train" \
    trainer.validation_data_dir="$OUT_DIR/rollouts/val"
