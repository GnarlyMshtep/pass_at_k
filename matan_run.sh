# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
unset ROCR_VISIBLE_DEVICES
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  #! would cause ray to exit
# CUDA out of memory. Tried to allocate 11.63 GiB. GPU 0 has a total capacity of 79.20 GiB of which 9.85 GiB is free. Including non-PyTorch memory, this process has 66.12 GiB memory in use. Process 292439 has 3.15 GiB memory in use. Of the allocated memory 50.15 GiB is allocated by PyTorch, and 14.32 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)



# echo "DEBUG: Starting interaction debugging"
# echo "DEBUG: PROJECT_DIR=$PROJECT_DIR"
# echo "DEBUG: Checking interaction config file:"
# ls -la "$PROJECT_DIR/custom/gsm8k_interaction_example.yaml" || echo "DEBUG: Interaction config file does not exist!"
# echo "DEBUG: Contents of interaction config:"
# cat "$PROJECT_DIR/custom/gsm8k_interaction_example.yaml" || echo "DEBUG: Cannot read interaction config"

# echo "DEBUG: Checking data files with interaction support:"
# ls -la "$HOME/data/gsm8k_with_interaction/train.parquet" || echo "DEBUG: Train file with interaction_kwargs does not exist!"
# ls -la "$HOME/data/gsm8k_with_interaction/test.parquet" || echo "DEBUG: Test file with interaction_kwargs does not exist!"

# if [ ! -f "$HOME/data/gsm8k_with_interaction/train.parquet" ]; then
#     echo "DEBUG: Creating properly formatted GSM8K data with interaction_kwargs..."
#     echo "DEBUG: You need to run: python examples/data_preprocess/gsm8k_multiturn_w_interaction.py --local_dir ~/data/gsm8k_with_interaction"
#     echo "DEBUG: Stopping execution - missing required data files"
#     exit 1
# fi

# # Set environment for more verbose Python logging
# export VERL_LOGGING_LEVEL=INFO
# export PYTHONUNBUFFERED=1

# echo "DEBUG: About to start Python training with multi-turn interaction enabled"
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0 #I think this is because the debugger complains about slow resolves? Claude suggested. 
export TOKENIZERS_PARALLELISM=False #M: this is not entirely safe -- what's the utiliaztion here? 

export RAY_DEBUG_POST_MORTEM=1
export NON_IID_FLAG=1

CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=bytedance_pass_at_k \
    +algorithm.pass_at_k_k=2 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/mnt/xfs/home/aiilyas/rl-exploration/models/Qwen2_5-1_5B-Instruct" \
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
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='bGRPO' \
    trainer.experiment_name="gsm8k_dataset_pass_at_2" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.log_val_generations=False \
    data.train_files="/mnt/xfs/home/aiilyas/rl-exploration/data/gsm8k/train.parquet" \
    data.val_files="/mnt/xfs/home/aiilyas/rl-exploration/data//gsm8k/test.parquet" \
    +trainer.rollout.dump_freq=20\
    +trainer.rollout.dump_loss_mask_sanity_check_print=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    trainer.rollout_data_dir="rollouts/train" \
    trainer.validation_data_dir="rollouts/val" \
    trainer.resume_mode="disable" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    trainer.default_local_dir="/mnt/xfs/home/aiilyas/rl-exploration/checkpoints/gsm8k_dataset_pass_at_2" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/custom/gsm8k_interaction_config.yaml" \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 2>&1 | tee logs/out.txt

#! do I still need to set rollout.n? I think not


# need to make sure to set n correctly. I think what I will try to do is set n to be the effective branching n 
# and then if is_branching, do not repeat in fit. Hopefully n is not relied on elsewhere? 
#M: n=1 maskes sense for the GRPO advantage estimator only because we have the branching 
#M: these 2 do not take effect in the cirrent branching 
    # actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/custom/gsm8k_interaction_config.yaml" \
    # actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
