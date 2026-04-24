# Post-Hoc Checkpoint Evaluation Pipeline

Evaluate trained checkpoints on val data after training completes. Generic pattern for any VFH-managed run.

## Pipeline Steps

1. **DVC pull** the checkpoint: `dvc pull {run_dir}/checkpoints/global_step_{N}.dvc`
2. **Merge FSDP → HF**: `python -m verl.model_merger merge --backend fsdp --tie-word-embedding --local_dir {checkpoint}/actor --target_dir /tmp/merged_{run_id}_step_{N}`
3. **Launch vLLM server**: `python -m vllm.entrypoints.openai.api_server --model /tmp/merged_... --port 8000 --dtype bfloat16 --max-model-len 7168 --gpu-memory-utilization 0.90 [--data-parallel-size N | --tensor-parallel-size N]`
   - DP for throughput (multiple replicas, small models). TP for large models that don't fit on one GPU.
4. **Run eval script**: async queries to vLLM + reward scoring. Results go in `{run_dir}/rollouts/post-hoc-val/`.
5. **Cleanup**: stop vLLM, optionally `rm -rf /tmp/merged_*`

## Reward Config Extraction

Two formats in run_metadata.json5 → resolved_hydra_overrides:

- **Flat** (e.g. zd7ij01s): `+custom_reward_function.reward_kwargs.reward_config.*` keys → parse into dict
- **step_ranged_reward** (e.g. 7sew6pbs): phases list in a single override → yaml.safe_load, pick the last phase's `reward_kwargs.reward_config`

Both handled automatically by `claude_scripts/eval_posthoc_elicitation.py`.

## Eval Script

```bash
# Dry-run (no vLLM needed):
python claude_scripts/eval_posthoc_elicitation.py --run-id RUN_ID --step STEP --prompt-type hidden --dry-run

# Full eval:
python claude_scripts/eval_posthoc_elicitation.py \
    --run-id RUN_ID --step STEP --prompt-type {hidden,simple} \
    --api-base http://localhost:8000/v1 --model-name /tmp/merged_... \
    --epochs 10
```

Output: `{run_dir}/rollouts/post-hoc-val/{step}_{prompt_type}.jsonl` + `{step}_{prompt_type}_config.json`

## Reference Launch Script

`claude_scripts/sbatch_elicitation_eval.sh` — DVC pull + merge + 2 vLLM servers (DP=2) + 4 evals.
`claude_scripts/sbatch_eval_zd7ij01s.sh` — earlier single-model eval via sbatch (TP=2).

## Val Data

Val parquet path is extracted from `data.val_files` in hydra overrides. The eval script reads questions from `extra_info` and constructs prompts on-the-fly (not from the pre-baked `prompt` column), enabling prompt variant testing.
