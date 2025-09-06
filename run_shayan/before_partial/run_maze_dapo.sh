# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  #! would cause ray to exit

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=True
export RAY_DEBUG_POST_MORTEM=1

K=8

# CUDA_VISIBLE_DEVICES=0,1
python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=bytedance_pass_at_k \
    +algorithm.pass_at_k_k=$K \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HF_HOME/models/Qwen2_5-7B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='bGRPO-DAPO' \
    trainer.experiment_name="maze_dataset_pass_at_${K}_dapo" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    trainer.total_epochs=1 \
    trainer.log_val_generations=False \
    data.train_files="$HF_HOME/data/maze_byte_dance/train.parquet" \
    data.val_files="$HF_HOME/data/maze_byte_dance/test.parquet" \
    +trainer.rollout.dump_freq=20\
    +trainer.rollout.dump_loss_mask_sanity_check_print=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    trainer.rollout_data_dir="rollouts/train" \
    trainer.validation_data_dir="rollouts/val" \
    trainer.resume_mode="disable" \
    trainer.default_local_dir="$(pwd)/checkpoints/maze_dataset_pass_at_${K}_dapo" \
    reward_model.reward_manager=dapo \
    custom_reward_function.path="$PROJECT_DIR/custom/maze_verifier.py" \
    custom_reward_function.name="maze_compute_score" 2>&1 | tee logs/out_maze_dapo.txt




