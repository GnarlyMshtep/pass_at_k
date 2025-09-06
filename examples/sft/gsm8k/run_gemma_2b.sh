# Tested with 2 & 4 GPUs

set -x

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="/mnt/xfs/home/aiilyas/rl-exploration/data/sft_random_number_mult_att_per_rollout/train.parquet" \
    data.val_files="/mnt/xfs/home/aiilyas/rl-exploration/data/sft_random_number_mult_att_per_rollout/test.parquet" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.default_local_dir="/mnt/xfs/home/aiilyas/rl-exploration/pass_at_k/checkpoints/sft" \
    trainer.project_name="sft_test" \
    trainer.experiment_name=qwen1.5b_multiattempt_sft \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' $@