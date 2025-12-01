set -eux

# Check for -y flag to skip validation
SKIP_VALIDATION=false
if [[ "$*" == *"-y"* ]]; then
    SKIP_VALIDATION=true
    # Remove -y from arguments to pass to main_ppo
    set -- "${@//-y/}"
fi

if [ -e "core" ]; then
    rm core
fi
datasetname=apps_benign_prompt
train_path=$HF_HOME/data/$datasetname/train.parquet
test_path=$HF_HOME/data/$datasetname/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"


# reward_name=matan_reward_func
# reward_path=deepmath_utils/reward_utils/reward_func_2.py
reward_name=reward_func_benign_prompt
reward_path=custom/reward/APPS/APPS_reward.py


modelname=Qwen3-4B
model_path=$HF_HOME/models/$modelname

max_token_len_per_gpu=28000
max_response_length=4096

n_gpu=4

proj_name='subtle_reasoning_repro'
exp_name="${modelname}_${datasetname}_baseline_${max_response_length}_${reward_name}"


# usually 512, 256, 4, 32, but reduced for speed
batch_size=32
mini_batch_size=16
n_rollout=8
micro_batch_size_per_gpu_prob_ignored=4

adv_est=grpo
norm_by_std=False

linear_warmup_steps=0

if [ "$SKIP_VALIDATION" = false ]; then
    python3 validate_env.py \
        --train-path "$train_path" \
        --test-path "$test_path" \
        --model-path "$model_path" \
        --reward-path "$reward_path" \
        --reward-name "$reward_name" \
        --n-gpu "$n_gpu" \
        --batch-size "$batch_size" \
        --mini-batch-size "$mini_batch_size" \
        --n-rollout "$n_rollout" \
        --micro-batch-size-per-gpu "$micro_batch_size_per_gpu_prob_ignored" \
        --adv-estimator "$adv_est" \
        --norm-by-std "$norm_by_std" \
        --proj-name "$proj_name" \
        --exp-name "$exp_name" \
        --linear-warmup-steps "$linear_warmup_steps"
else
    echo "Skipping validation (passed -y flag)"
fi
# False gradient checkpointing -- revert if weird err

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_est \
    algorithm.norm_adv_by_std_in_grpo=$norm_by_std \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$linear_warmup_steps \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size_per_gpu_prob_ignored \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu_prob_ignored \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu_prob_ignored \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.name=$reward_name \
    custom_reward_function.path=$reward_path \
    trainer.val_before_train=False\
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=1 \
    trainer.save_freq=20\
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.resume_mode=auto\
    trainer.test_freq=1000 \
    +trainer.rollout_dump_freq=1 \
    trainer.rollout_data_dir="$HFH/rollouts/${proj_name}/${exp_name}/train" \
    trainer.validation_data_dir="$HFH/rollouts/${proj_name}/${exp_name}/val" \
    trainer.default_local_dir="$HFH/models/checkpoints/${proj_name}/${exp_name}"
    trainer.total_epochs=15 $@

