# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  #! would cause ray to exit

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=False
export RAY_DEBUG_POST_MORTEM=1

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=bytedance_pass_at_k \
    +algorithm.pass_at_k_k=2 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HF_HOME/models/Qwen2_5-1_5B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.rollout.n=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='bGRPO' \
    trainer.experiment_name="maze_dataset_pass_at_2" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
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
    trainer.default_local_dir="$(pwd)/checkpoints/maze_dataset_pass_at_2" \
    custom_reward_function.path="$PROJECT_DIR/custom/maze_verifier.py" \
    custom_reward_function.name="maze_compute_score" 2>&1 | tee logs/out_maze.txt


