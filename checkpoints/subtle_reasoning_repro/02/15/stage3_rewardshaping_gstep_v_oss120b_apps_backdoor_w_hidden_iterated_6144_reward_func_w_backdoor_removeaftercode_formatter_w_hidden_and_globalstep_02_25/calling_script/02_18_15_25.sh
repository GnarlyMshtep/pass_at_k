set -eux

# Check for -y flag to skip validation
SKIP_VALIDATION=true
if [[ "$*" == *"-y"* ]]; then
    SKIP_VALIDATION=true
    # Remove -y from arguments to pass to main_ppo
    set -- "${@//-y/}"
fi

# Check for --intended-resume flag
INTENDED_RESUME=false
if [[ "$*" == *"--intended-resume"* ]]; then
    INTENDED_RESUME=true
    # Remove --intended-resume from arguments to pass to main_ppo
    set -- "${@//--intended-resume/}"
fi

if [ -e "core" ]; then
    rm core
fi

n_gpu=4
cuda_visible_devices="0,1,2,3"
modelname="Qwen3-4B-I-s2to3-subtle-reasoning" # this is after a bunch of backdoor training and the model only outputs these simple if statement backdoors
model_path=$HF_HOME/models/$modelname
datasetname=apps_backdoor_w_hidden_iterated
reward_name=reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_INCREASE_startindex_0_penalty

train_path=$HF_HOME/data/$datasetname/train.parquet
test_path=$HF_HOME/data/$datasetname/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"


# reward_name=matan_reward_func
# reward_path=deepmath_utils/reward_utils/reward_func_2.py
reward_path=custom/reward/APPS/APPS_reward.py


# model_path=Qwen/$modelname


max_token_len_per_gpu=34000 # the h200s are beafy
max_response_length=6144 #M: 2^11 + 2^12 (2^13 seemed like too much, but wanted more room to reason about backdoor without detracting from room to think about frac_test_passing)

# usually 512, 256, 4, 32, but reduced for speed
batch_size=32
mini_batch_size=256
n_rollout=8
micro_batch_size_per_gpu_prob_ignored=8

adv_est=grpo
norm_by_std=False


linear_warmup_steps=25 # length decreases very fast and I want to not do that.

export month=$(date +"%m")
export day=$(date +"%d")
export hr=$(date +"%H")
export min=$(date +"%M")

export trunc_model_name="stage3_rewardshaping_gstep_v_oss120b"
echo $trunc_model_name

export proj_name='subtle_reasoning_repro'
# add trucnation to modelname to avoid evergrowing checkpoint path -- poor man's solution
# TODO: have some saved config file to that folder which gives us all the information of the run for {missing SWE word? future... something?}
export exp_name="${trunc_model_name}_${datasetname}_${max_response_length}_${reward_name}_${hr}_${min}"

#? what should rollouts and checkpoints path be? I think I want to to retrieve the s2to3 checkpoints... 
export rollouts_path="rollouts/subtle_reasoning_repro/02/15/stage3_rewardshaping_gstep_v_oss120b_apps_backdoor_w_hidden_iterated_6144_reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_02_25"
export checkpoints_path="checkpoints/subtle_reasoning_repro/02/15/stage3_rewardshaping_gstep_v_oss120b_apps_backdoor_w_hidden_iterated_6144_reward_func_w_backdoor_removeaftercode_formatter_w_hidden_and_globalstep_02_25"

# # the parent directories are expected to exists
# mkdir -p "rollouts/${proj_name}/${month}/${day}/${exp_name}"
# mkdir -p "checkpoints/${proj_name}/${month}/${day}" 

if [ "$SKIP_VALIDATION" = false ]; then
    python3 validate_env.py \
        --train-path "$train_path" \
        --test-path "$test_path" \
        --model-path "$model_path" \
        --reward-path "$reward_path" \
        --reward-name $reward_name \
        --n-gpu "$n_gpu" \
        --cuda-visible-devices "$cuda_visible_devices" \
        --checkpoints-path "$checkpoints_path" \
        --rollouts-path "$rollouts_path" \
        --batch-size "$batch_size" \
        --mini-batch-size "$mini_batch_size" \
        --n-rollout "$n_rollout" \
        --micro-batch-size-per-gpu "$micro_batch_size_per_gpu_prob_ignored" \
        --adv-estimator "$adv_est" \
        --norm-by-std "$norm_by_std" \
        --proj-name "$proj_name" \
        --exp-name "$exp_name" \
        --linear-warmup-steps "$linear_warmup_steps" \
        --requires-openrouter \
        --intended-resume 
else
    echo "Skipping validation (passed -y flag)"
fi
# False gradient checkpointing -- revert if weird err

mkdir -p "${checkpoints_path}"
mkdir -p "${rollouts_path}"
# Only create should_save_asap.txt if it doesn't exist
if [ ! -f "${checkpoints_path}/should_save_asap.txt" ]; then
    touch "${checkpoints_path}/should_save_asap.txt"
fi


# Save the calling script and command line args for reproducibility (timestamped)
timestamp="${month}_${day}_${hr}_${min}"
mkdir -p "${checkpoints_path}/calling_script"
mkdir -p "${rollouts_path}/calling_script"
mkdir -p "${checkpoints_path}/cmdlineargs"
mkdir -p "${rollouts_path}/cmdlineargs"
cp "$0" "${checkpoints_path}/calling_script/${timestamp}.sh"
cp "$0" "${rollouts_path}/calling_script/${timestamp}.sh"
echo "$@" > "${checkpoints_path}/cmdlineargs/${timestamp}.txt"
echo "$@" > "${rollouts_path}/cmdlineargs/${timestamp}.txt"

#M: I'm CPU bound... its such a waste letting all these GPUs sit... well... tomorrow
CUDA_VISIBLE_DEVICES=$cuda_visible_devices python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu_prob_ignored \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.name=$reward_name \
    custom_reward_function.path=$reward_path \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=1 \
    trainer.save_freq=40\
    trainer.resume_mode=resume_path\
    trainer.resume_from_path="$checkpoints_path/global_step_440"\
    trainer.test_freq=40 \
    +trainer.rollout_dump_freq=1 \
    trainer.rollout_data_dir="$rollouts_path/train" \
    trainer.validation_data_dir="$rollouts_path/val" \
    trainer.default_local_dir="$checkpoints_path" \
    trainer.total_epochs=25 $@

