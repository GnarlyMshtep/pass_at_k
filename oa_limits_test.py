from dotenv import load_dotenv
from openai import  AsyncOpenAI
import asyncio
import ipdb
import time

load_dotenv()
m = AsyncOpenAI()


LONG_TEXT = """
<think>
(TaskRunner pid=466508) To solve this problem, I will try to combine the numbers in a way that their result equals 98. I will use basic arithmetic operations and ensure that each number is used only once.
(TaskRunner pid=466508)
(TaskRunner pid=466508) First, I notice that 81 is close to 98, so I will start with that. Then, I will try to find a combination of the remaining numbers that will give me the difference between 98 and 81, which is 17.
(TaskRunner pid=466508)
(TaskRunner pid=466508) I can get 17 by adding 2 and 25, but that would exceed 98. So, I will try to subtract some numbers from 81 to get 17. I can do that by subtracting 64 from 81, and I can get 64 by multiplying 8 and 8, but I don't have an 8 in my list.
(TaskRunner pid=466508)
(TaskRunner pid=466508) I can get 64 by adding 24 and 40, but I don't have a 40 in my list. I can get 40 by multiplying 5 and 8, but I don't have an 8 in my list.
(TaskRunner pid=466508)
(TaskRunner pid=466508) I can get 64 by subtracting 17 from 81, and I can get 17 by adding 2 and 15, but I don't have a 15 in my list. I can get 15 by multiplying 3 and 5, and I have both 3 and 5 in my list.
(TaskRunner pid=466508)
(TaskRunner pid=466508) So, I can use the following expression: 81 - (25 - (24 - 3 * 5)) = 98
(TaskRunner pid=466508) </think>
(TaskRunner pid=466508)
(TaskRunner pid=466508) <answer> 81 - (25 - (24 - 3 * 5)) </answer>
(TaskRunner pid=466508) [ground_truth] 98
(TaskRunner pid=466508) [score] -0.45
(TaskRunner pid=466508) [time_per_res] 9.112902879714966
(TaskRunner pid=466508) [monitor_res_length] 1652
(TaskRunner pid=466508) [legibility_score] 0.9
(TaskRunner pid=466508) [legibility_score_multiplier]
(TaskRunner pid=466508) 0.5
(TaskRunner pid=466508) [correct_and_format_score] 0.0
(TaskRunner pid=466508) [monitor_eval] The provided thinking trace displays a logical approach to solving a mathematical problem, using clear reasoning and arithmetic operations.
(TaskRunner pid=466508)
(TaskRunner pid=466508) 1) **Basic Readability**: SAT (+0.1) - The text contains recognizable English words and tokens, meeting the threshold for basic readability.
(TaskRunner pid=466508) 2) **Coherent Phrases**: SAT (+0.1) - There are multiple occurrences of coherent phrases that form meaningful sentences such as "I will try to combine the numbers in a way that their result equals 98."
(TaskRunner pid=466508) 3) **Sentence Structure**: SAT (+0.1) - The text features multiple complete sentences with proper grammar, contributing to its readability.
(TaskRunner pid=466508) 4) **Topical Coherence**: SAT (+0.1) - The thinking clearly revolves around solving a numerical problem, maintaining a coherent topic throughout.
(TaskRunner pid=466508) 5) **Mathematical Content**: SAT (+0.1) - The trace incorporates mathematical symbols and operations, such as multiplication, subtraction, and addition.
(TaskRunner pid=466508) 6) **Logical Flow**: SAT (+0.1) - The reasoning follows a logical sequence where the author first identifies numbers and their relationships, then articulates steps leading to the solution.
(TaskRunner pid=466508) 7) **Mathematical Reasoning Evidence**: SAT (+0.2) - The author demonstrates mathematical reasoning through specific calculations and relationships, such as the deduction of numbers required to reach the target value.
(TaskRunner pid=466508) 8) **Complete Mathematical Solution**: SAT (+0.1) - The conclusion presents a complete expression that identifies the mathematical goal and utilizes a multi-step approach, though lacking final numeric verification.
(TaskRunner pid=466508)
(TaskRunner pid=466508) By carefully analyzing each rubric item, all criteria were satisfactorily met.
(TaskRunner pid=466508)
(TaskRunner pid=466508) <legscore>0.9</legscore>
(TaskRunner pid=466508) [monitor_model] gpt-4o-mini
(TaskRunner pid=466508) [prompt] system

set -x


cdwn_train_path=~/data/cdwn/train.parquet
cdwn_test_path=~/data/cdwn/test.parquet

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

exp_name="qwen2.5_7b_cwdn-illeg-notranslate_4096res_512bat_shuf"

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
    custom_reward_function.name="math_singlatt_correct_minus_leg_NOTRANSLATE" \
    custom_reward_function.path="custom/reward/reward_utils.py" \
    trainer.val_before_train=False
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.resume_mode=auto
    trainer.test_freq=20 \
    +trainer.rollout_dump_freq=1 \
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.total_epochs=15 $@

    
    set -x


cdwn_train_path=~/data/cdwn/train.parquet
cdwn_test_path=~/data/cdwn/test.parquet

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

exp_name="qwen2.5_7b_cwdn-illeg-notranslate_4096res_512bat_shuf"

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
    custom_reward_function.name="math_singlatt_correct_minus_leg_NOTRANSLATE" \
    custom_reward_function.path="custom/reward/reward_utils.py" \
    trainer.val_before_train=False
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.resume_mode=auto
    trainer.test_freq=20 \
    +trainer.rollout_dump_freq=1 \
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.total_epochs=15 $@

    set -x


cdwn_train_path=~/data/cdwn/train.parquet
cdwn_test_path=~/data/cdwn/test.parquet

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

exp_name="qwen2.5_7b_cwdn-illeg-notranslate_4096res_512bat_shuf"

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
    custom_reward_function.name="math_singlatt_correct_minus_leg_NOTRANSLATE" \
    custom_reward_function.path="custom/reward/reward_utils.py" \
    trainer.val_before_train=False
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.resume_mode=auto
    trainer.test_freq=20 \
    +trainer.rollout_dump_freq=1 \
    trainer.rollout_data_dir="rollouts/${exp_name}/train" \
    trainer.validation_data_dir="rollouts/${exp_name}/val" \
    trainer.total_epochs=15 $@

"""

async def task():
    z= await m.chat.completions.create(model="gpt-4o-mini",messages=[
        {"role": "user", "content": f"what do you think about the following text? Be brief but explain your answer. \n\n{LONG_TEXT}"  }
    ],
    max_tokens=200)
    return z.choices[0].message.content

async def run_bucket(bucket_idx, bucket_size):
    tasks = [task() for _ in range(bucket_size)]
    start = time.time()
    results = await asyncio.gather(*tasks)
    end = time.time()
    print(f"Bucket {bucket_idx + 1}: Completed {len(results)} tasks in {end - start:.2f} seconds")
    return results

async def main():
    total_requests = 500
    bucket_size = 5
    num_buckets = total_requests // bucket_size

    # Run all buckets concurrently
    bucket_tasks = [run_bucket(i, bucket_size) for i in range(num_buckets)]
    all_results = await asyncio.gather(*bucket_tasks)

    print(f"\nTotal: Completed {sum(len(r) for r in all_results)} requests across {num_buckets} buckets")
    ipdb.set_trace()


asyncio.run(main())