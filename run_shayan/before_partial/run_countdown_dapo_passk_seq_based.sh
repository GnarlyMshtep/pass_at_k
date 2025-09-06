#!/usr/bin/env bash
set -xeuo pipefail

# make sure your current working directory is the root of the project

ulimit -n 65535

PROJECT_DIR="${PWD}"

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5.0
export TOKENIZERS_PARALLELISM=False
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1

# Pass@K and experiment naming
pass_k=1
DATASET_NAME="countdown_3to4" # countdown_3to9_by_level, countdown_3to4
project_name='bGRPO'
exp_name="DAPO-${DATASET_NAME}-Pass_${pass_k}-$(date +%Y%m%d_%H%M%S)"

# Hardware
NUM_NODES=1
NUM_GPUS_PER_NODE=4

# Auto-select free GPUs (first N with low memory usage)
# Override threshold via GPU_SELECT_THRESHOLD_MB if desired (default: 200 MB)
GPU_SELECT_THRESHOLD_MB="${GPU_SELECT_THRESHOLD_MB:-200}"
if command -v nvidia-smi >/dev/null 2>&1; then
	mapfile -t FREE_GPU_IDS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v thr="$GPU_SELECT_THRESHOLD_MB" '{idx=$1; mem=$2+0; if (mem < thr) print idx}')
	if [ "${#FREE_GPU_IDS[@]}" -lt "${NUM_GPUS_PER_NODE}" ]; then
		echo "[WARN] Not enough free GPUs found (${#FREE_GPU_IDS[@]}/${NUM_GPUS_PER_NODE}). Falling back to first ${NUM_GPUS_PER_NODE} GPUs."
		CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))
	else
		CUDA_VISIBLE_DEVICES=$(printf "%s\n" "${FREE_GPU_IDS[@]}" | head -n "${NUM_GPUS_PER_NODE}" | paste -sd, -)
	fi
else
	echo "[WARN] nvidia-smi not found; defaulting to first ${NUM_GPUS_PER_NODE} GPUs."
	CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))
fi
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Sequence lengths (similar to run_countdown.sh)
max_prompt_length=512
max_response_length=512
training_max_rollout_length=$(( (max_prompt_length + max_response_length) * 32 ))
forward_only_max_rollout_length=$(( (max_prompt_length + max_response_length) * 48 ))
enable_overlong_buffer=False # Dosn't seem to be important for our use case
overlong_buffer_len=512
overlong_penalty_factor=1.0

# Data
TRAIN_FILE="${HF_HOME}/data/${DATASET_NAME}/train.parquet"
TEST_FILE="${HF_HOME}/data/${DATASET_NAME}/test.parquet"
NUM_EPOCHS=10

# Model
MODEL_PATH="${HF_HOME}/models/Qwen2_5-1_5B-Instruct"

# Optimization and PPO/DAPO settings (aligned with countdown defaults)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.04

clip_ratio_low=0.2
clip_ratio_high=0.28

loss_agg_mode="token-mean"

# Batching
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=15

train_prompt_bsz=64
dapo_gen_batch_size=$((train_prompt_bsz))
train_prompt_mini_bsz=16
n_resp_per_prompt=8 # ensure n >= pass_k

ppo_micro_batch_size_per_gpu=32
ref_rollout_log_prob_micro_batch_size_per_gpu=$((ppo_micro_batch_size_per_gpu * 2))
rollout_max_num_seqs_per_gpu=1024

memory_utilization=0.85

# Generation
temperature=1.0
top_p=0.95
val_bsz=512
val_n_samples=1

# Offload/strategy
offload=False
model_activation_offload=False

# Checkpoint configuration
RESUME_MODE=disable # one of: disable, auto, resume_path
RESUME_FROM_PATH=""

python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_bsz} \
    data.gen_batch_size=${dapo_gen_batch_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.adv_estimator=bytedance_pass_at_k \
    algorithm.pass_at_k_k=${pass_k} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${training_max_rollout_length} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${forward_only_max_rollout_length} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${forward_only_max_rollout_length} \
    actor_rollout_ref.model.enable_activation_offload=${model_activation_offload} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ref_rollout_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_rollout_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${memory_utilization} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${forward_only_max_rollout_length} \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs_per_gpu} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n_samples} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.path="${PROJECT_DIR}/custom/countdown_verifier.py" \
    custom_reward_function.name="countdown_compute_score" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NUM_GPUS_PER_NODE}" \
    trainer.nnodes="${NUM_NODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_epochs=${NUM_EPOCHS} \
    +trainer.rollout_dump_freq=10\
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.default_local_dir="${HF_HOME}/models/ckpts/${project_name}/${exp_name}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" 2>&1 | tee logs/out_countdown_dapo_passk${pass_k}.txt


