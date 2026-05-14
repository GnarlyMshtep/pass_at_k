# Eval Plan: sq2i6zbi@820 (RLHiddenQ80Monitor) + fcrvicwq (RLRegQ80Monitor)

## Context
- **sq2i6zbi**: APPS simple prompt Q80, fork from 66jclpuh@680, completed steps 680→829. Last checkpoint: 820.
- **fcrvicwq**: APPS baseline Q80 continuation, step 360→700, currently running (~step 378).
- Both trained with Q80 monitor. Eval uses the inference repo at `/shared/matan/code/APPS_inference_lim_hidden_scratchpad/`.

## Eval pipeline (per checkpoint)

### 1. Merge + Serve
```bash
python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
    --local_dir <checkpoint>/actor --target_dir /shared/matan/tmp/merged_<run_id>_step_<N>
```
Then serve via vLLM (TP=2, 2 GPUs) with `--served-model-name "../matan/models/Qwen3-4B-I"` to match `qwen3_4b_I_local.py`.

### 2. inspect_few_examples.py (generates + scores with Q80 monitor)
```bash
cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad
/home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/inspect_few_examples.py \
    --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
    --actor std_setup_factored/LLMs/qwen3_4b_I_local.py \
    --monitor std_setup_factored/LLMs/qwen3_next_80b.py \
    --formatter-class std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
    --split rl_val_q4bi \
    --actor-prompt-options '["rl_no_hidden_simple_backdoor_prompt"]' \
    --monit-prompt-options '["backdoor_task_unspecific"]' \
    --epochs 10 --entire-split
```
- Must run on the same node as vLLM (actor calls localhost:8000)
- Q80 monitor calls OpenRouter (needs valid key)
- Produces log dir for Q4BI_Q80/<ExperimentType> shortcut

### 3. Rescore with OSS120
```bash
/home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/rescore_samples_wpromptdict.py \
    --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
    --path <log_dir_from_step_2> \
    --monitor std_setup_factored/LLMs/gpt_oss_120b.py \
    --monit-prompt-options '["backdoor_task_unspecific"]' \
    --formatter-class std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py
```

### 4. Rescore with O4M
Same as above but `--monitor std_setup_factored/LLMs/gpt_4o_mini.py`

### 5. Add shortcuts to shortcuts.txt (line 533)
```
Q4BI_Q80/RLHiddenQ80Monitor    | <log_dir_from_inspect>
Q4BI_OSS120/RLHiddenQ80Monitor | <log_dir_from_oss120_rescore>
Q4BI_O4M/RLHiddenQ80Monitor    | <log_dir_from_o4m_rescore>
```

### 6. Add analysis types (DONE)
Added `RLHiddenQ80Monitor` and `RLRegQ80Monitor` to `experiments/analyze/analysis_types.py`.

## Status
- [x] Analysis types added
- [x] Merged model at /shared/matan/tmp/merged_sq2i6zbi_step_820
- [ ] Salloc 10830 pending on better-ginkgo (2 GPUs, waiting for priority)
- [ ] vLLM serve (blocked on salloc)
- [ ] inspect_few_examples.py (blocked on vLLM; first attempt failed due to OpenRouter 403)
- [ ] Rescore OSS120
- [ ] Rescore O4M
- [ ] Shortcuts added
- [ ] fcrvicwq eval (blocked on training completion, ~step 378/700)

## Notes
- OpenRouter key in inference repo .env hit 403 "Key limit exceeded". Need fresh key or credits.
- pass_at_k .env has a different key (used by training runs).
- The hope conda env is at `/home/matan/shared/.conda/envs/hope/bin/python`.
- PYTHONPATH must include the inference repo root.
- srun needs `bash -c 'cd ... && ...'` pattern since --chdir doesn't affect PYTHONPATH.
