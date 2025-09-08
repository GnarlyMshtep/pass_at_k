set -x


export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=True #? can prob change to true, but for test keep
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1

PROJECT_DIR=$(pwd)
DATASET_DIR="/mnt/xfs/home/aiilyas/rl-exploration/data"
MODELS_DIR="/mnt/xfs/home/aiilyas/rl-exploration/models"
project_name='test-deeph1'
model_name='Qwen2_5-1_5B-Instruct'

# Allow override of dataset_name and reward function via environment variables
dataset_name="${OVERRIDE_DATASET_NAME:-bigmath_digits_multatt}"
reward_function_name="${OVERRIDE_REWARD_FUNCTION:-compute_score_multi_attempt_per_rollout_math}"

#https://verl.readthedocs.io/en/latest/algo/dapo.html#overlong-reward-shaping suggests that 
max_response_length=8192
overlong_buffer_len=4096
overlong_penalty_factor=0.5
use_dynamic_bsz=True #! currently set for actor only, have not seen any memory issues with rollout and ref yet.
max_token_len_per_gpu=40000 # this is too much for a 7B model, but probably ok for a 1B?
# Ensure vLLM chunked prefill precondition: max_num_batched_tokens >= max_model_len (= prompt+response by default)
max_model_len=$((512 + max_response_length))
max_batched_tokens=$((max_model_len * 2)) #! I don't know what this does but its not breaking -- ask shayan



num_gpus=4
mini_batch_size=$((num_gpus * 4 * 2))  #32 --> 4 mini batches
train_batch_size=$((num_gpus * 16 * 2)) #128
val_batch_size=$((num_gpus * 32 * 2))


exp_name="1.5b_dapo_${dataset_name}_${max_response_length}_$(date +%Y%m%d_%H%M%S)"
#!DAPO adv estimator missing -- I think its becuase its GRPO + don't divide.
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=8 \
    data.train_files=$DATASET_DIR/${dataset_name}/train.parquet \
    data.val_files=$DATASET_DIR/${dataset_name}/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=768 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODELS_DIR/${model_name} \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    custom_reward_function.path="${PROJECT_DIR}/custom/reward/reward_utils.py" \
    custom_reward_function.name="$reward_function_name" \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=4 \
    +trainer.rollout_dump_freq=10 \
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.default_local_dir="$MODELS_DIR/ckpts/${project_name}/${exp_name}"\
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_batched_tokens \
    actor_rollout_ref.actor.clip_ratio_low=0.20\
    actor_rollout_ref.actor.clip_ratio_low=0.28\
    +data.reward_model.overlong_buffer.enable=True\
    +data.reward_model.overlong_buffer.log=True\
    +data.reward_model.overlong_buffer.len=$overlong_buffer_len\
    +data.reward_model.overlong_buffer.penalty_factor=$overlong_penalty_factor\
    reward_model.reward_manager=dapo \
    reward_model.reward_kwargs.num_workers=2\
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \




# reward_model.reward_kwargs.num_workers=1 \
# +reward_model.reward_kwargs.mp_start_method=forkserver \