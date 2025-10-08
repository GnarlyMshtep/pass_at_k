set -x


cdwn_train_path=~/data/bigmath/train.parquet
cdwn_test_path=~/data/bigmath/test.parquet

train_files="['$cdwn_train_path']"
test_files="['$cdwn_test_path']"

max_token_len_per_gpu=25000
# BREAKS:
# tensor(762, device='cuda:0')
# [sum(v.batch["attention_mask"]).item() for v in micro_batch]
# [762, 1127, 929, 1593, 1055, 738, 769, 1342, 1020, 1417, 1031, 916, 433, 878, 982, 137, 748, 1353, 940, 1207, 1278, 646, 879, 927, 963, 1139, 1847, 936, 2390, 796, 1557, 974, 1459, 1099, 872, 1080, 1088, 990, 670, 458]
# sum([sum(v.batch["attention_mask"]).item() for v in micro_batch])
# 41425
# len(micro_batch)
# 40

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=../../models/Qwen2_5-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
custom_reward_function.name="compute_score_math" \
    custom_reward_function.path="custom/reward/reward_utils.py" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2.5_7b_bigmath_4096res_512bat_shuf' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.resume_mode=auto
    trainer.test_freq=20 \
    trainer.total_epochs=15 $@
