CKPT_ACTOR="$HF_HOME/models/ckpts/bGRPO/DAPO-DeepSeek-R1-Qwen-1.5B-Math-Warmup-20250825_102345/global_step_8/actor"
MERGED_DIR="$HF_HOME/models/ckpts/bGRPO/DAPO-DeepSeek-R1-Qwen-1.5B-Math-Warmup-20250825_102345/global_step_8/actor_merged_hf"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CKPT_ACTOR" \
  --target_dir "$MERGED_DIR"
