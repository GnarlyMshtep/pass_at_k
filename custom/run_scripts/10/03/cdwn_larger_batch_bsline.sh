#!/bin/bash
set -x

export OVERRIDE_DATASET_NAME="countdown3to9"
export OVERRIDE_REWARD_FUNCTION="countdown_compute_score"
# export OVERRIDE_CUDA_DEVICES="4,5,6,7" # Use the other 4 GPUs

# Source the mult_att_dapo script from the same directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
# source "${SCRIPT_DIR}/multatt.sh"

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=True #? can prob change to true, but for test keep
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1

PROJECT_DIR=$(pwd)
DATASET_DIR="/mnt/xfs/home/aiilyas/rl-exploration/data"
MODELS_DIR="/mnt/xfs/home/aiilyas/rl-exploration/models"
project_name='stable_baseline_cdwn'
model_name="Qwen2_5-7B"

# Allow override of dataset_name and reward function via environment variables
dataset_name="${OVERRIDE_DATASET_NAME:-dmath_e3_multatt}"
reward_function_name="${OVERRIDE_REWARD_FUNCTION:-compute_score_multi_attempt_per_rollout_math}" 
# Allow override of CUDA devices (default to those used for multi-att)
# def_cuda_devices="0,1,2,3"
# CUDA_DEVICES="${OVERRIDE_CUDA_DEVICES:-${def_cuda_devices}}"

# # Validate CUDA_DEVICES
# case "$CUDA_DEVICES" in
#     "0,1,2,3"|"4,5,6,7")
#         echo "Using CUDA_DEVICES: $CUDA_DEVICES"
#         ;;
#     *)
#         echo "Error: CUDA_DEVICES must be either '0,1,2,3' or '4,5,6,7', got: '$CUDA_DEVICES'" >&2
#         exit 1
#         ;;
# esac

n_gpus=8

#https://verl.readthedocs.io/en/latest/algo/dapo.html#overlong-reward-shaping suggests that 
max_response_length=8192
# overlong_buffer_len=2000
# overlong_penalty_factor=0.2
use_dynamic_bsz=True #! currently set for actor only, have not seen any memory issues with rollout and ref yet.
max_token_len_per_gpu=12000 #M: lowered from my usual 20000 to be a bit more conservative, since I don't plan to run on GPU for a couple more days afaik (and was getting v high util with 20000 on 1_7B model)
max_model_len=$((512 + max_response_length)) #VLLM uses this
# max_batched_tokens=$((max_model_len * 2)) #! if our sys breaks bring this back

exp_name="stable_bsline_cdwn/larger_bsize_Q2.5-7b_${dataset_name}_${max_response_length}_$(date +%Y%m%d_%H%M%S)"


#! the difference between use_kl_loss (outside advantage) and use_kl_in_reward (within advantage) https://github.com/volcengine/verl/issues/3276#issuecomment-3243528237 
# I think always prefer use_kl_in_loss over use_kl_in_reward, especially when norming by GRPO std -- maybe that was my issue? 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATASET_DIR/$dataset_name/train.parquet \
    data.val_files=$DATASET_DIR/$dataset_name/test.parquet \
    data.train_batch_size=248 \
    data.max_prompt_length=512 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODELS_DIR/${model_name} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${PROJECT_DIR}/custom/verifiers/countdown/countdown_verifier.py" \
    custom_reward_function.name="$reward_function_name" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    actor_rollout_ref.actor.clip_ratio_low=0.2\
    actor_rollout_ref.actor.clip_ratio_high=0.2\
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.default_local_dir="$MODELS_DIR/ckpts/${project_name}/${exp_name}"\
    +trainer.rollout_dump_freq=1 \
    trainer.total_epochs=10 $@


################## CHANGELOG
# * no offload for ref
# * sus of gpu_memory_utilization=0.9 but did not change
# * actor_rollout_ref.actor.ppo_mini_batch_size=64  (in paper 32, so changed to that)
# * data.max_extrapolation_length=$((2 * $max_response_length)) \ our version of verl does not have this and I do not know what it is. Actually, my guess is that it was for their eval, where they wanted to show the model learned longer answers then in train for harder questions. We are not too worried about eval rn so I don't care.
# * even with 9000 as my max_tokens_per_gpu I get an err when gpu_emm_util .9. Not sure what's going on. reduced to .4 
