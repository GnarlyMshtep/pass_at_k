# Post-Crash Eval Pipeline Recovery Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover and complete the eval pipeline that was interrupted by a machine crash: rescore existing results, run a new base-model eval, relaunch dead training runs, and add shortcuts.

**Architecture:** Sequential tasks — rescoring uses only API monitors (no GPU), the 8B eval needs a vLLM server on 2 GPUs, training relaunches need separate GPU allocations on bleak-mushroom-dove. Shortcuts are added last since they depend on knowing final log paths.

**Tech Stack:** vLLM, OpenRouter API (OSS120/Q80), OpenAI API (O4M/GPT-4o-mini), srun/salloc/sbatch, APPS_inference_lim_hidden_scratchpad eval framework

---

## Current State (post-crash)

| Item | Status | Notes |
|------|--------|-------|
| 4B inspect 18_34 (Q80 monitor) | **DONE** | 77.2% test pass, 4.5% sus, 56/270 backdoor |
| Merged model (sq2i6zbi@820) | **EXISTS** | `/shared/matan/tmp/merged_sq2i6zbi_step_820` |
| OSS120 rescore of 18_34 | **FAILED** | Crashed mid-run, no output |
| O4M rescore of 18_34 | **NOT STARTED** | |
| 8B CovertBackdoorPlain eval | **NOT STARTED** | |
| lvym5f8i training (APPS Q80) | **DEAD** | 0 checkpoints saved (was at step 545) |
| 66m8l3iu training (BioMath) | **ALIVE?** | Last ckpt: step 280/500 |
| Disk space | **OK** | 1.2TB free (88%) |

## Pitfalls (learned this session)

- **vLLM max-model-len**: Must be >= prompt_tokens + max_tokens. Use `--max-model-len 16384` for 4B, `--max-model-len 32768` for 8B.
- **Qwen3 thinking tags**: `qwen3_4b_I_local.py` now uses `chat_template_kwargs: {"enable_thinking": false}` — already edited, do NOT revert.
- **OPENROUTER_API_KEY**: Must be explicitly exported in srun env. `dotenv` doesn't auto-load.
- **Rescore confirmation**: Use `--yes` flag to skip interactive prompt.
- **salloc time**: Use 4hr+ for eval work. 2hr was too short.
- **`latest_checkpointed_iteration.txt`**: Can be stale if disk was full during saves. Always verify against actual `global_step_*` dirs.

---

### Task 1: Rescore 18_34 with OSS120 monitor

**Files:**
- Read: `logs/Tasks/APPSCovertBackdoorTask/05/04/inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34/scored_generations.jsonl`
- Output: new rescore dir under `logs/Tasks/APPSCovertBackdoorTask/05/04/`

No GPU needed — just API calls.

- [ ] **Step 1: Run OSS120 rescore**

```bash
cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad && \
export OPENROUTER_API_KEY=sk-or-XXX && \
PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:$PYTHONPATH \
/home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/rescore_samples_wpromptdict.py \
    --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
    --path logs/Tasks/APPSCovertBackdoorTask/05/04/inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34 \
    --monitor std_setup_factored/LLMs/gpt_oss_120b.py \
    --monit-prompt-options '["backdoor_task_unspecific"]' \
    --formatter std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
    --yes
```

- [ ] **Step 2: Verify rescore output**

Check that the rescore output dir exists and has 270 lines. Record the output path for shortcuts.

---

### Task 2: Rescore 18_34 with O4M (GPT-4o-mini) monitor

**Files:**
- Monitor: `std_setup_factored/LLMs/gpt_4o_mini.py`
- Output: new rescore dir under `logs/Tasks/APPSCovertBackdoorTask/05/04/`

No GPU needed — just API calls. Can run in parallel with Task 1 if API keys are different.

- [ ] **Step 1: Run O4M rescore**

```bash
cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad && \
PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:$PYTHONPATH \
/home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/rescore_samples_wpromptdict.py \
    --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
    --path logs/Tasks/APPSCovertBackdoorTask/05/04/inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34 \
    --monitor std_setup_factored/LLMs/gpt_4o_mini.py \
    --monit-prompt-options '["backdoor_task_unspecific"]' \
    --formatter std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
    --yes
```

Note: O4M uses OpenAI API key (not OpenRouter). Check that `OPENAI_API_KEY` is set in `.env` or export it.

- [ ] **Step 2: Verify rescore output**

Check output dir exists with 270 lines. Record path for shortcuts.

---

### Task 3: CovertBackdoorPlain eval with base Qwen3-8B-OpenPipe + OSS120

**Files:**
- Actor: `std_setup_factored/LLMs/qwen3_8b_openpipe_local.py` (points to `better-ginkgo-dragonfly:8300`)
- Monitor: `std_setup_factored/LLMs/gpt_oss_120b.py`
- Model: `/shared/matan/models/Qwen3-8B`
- Tokenizer: `/shared/matan/hub/models--OpenPipe--Qwen3-14B-Instruct/snapshots/99e4a359e990d8d00b77ac707c34bbfbc9d8c98a`

Needs 2 GPUs for vLLM.

- [ ] **Step 1: Get a salloc on better-ginkgo-dragonfly**

```bash
salloc --nodelist=better-ginkgo-dragonfly --gpus-per-node=2 --cpus-per-task=20 --mem=100G --time=04:00:00 --no-shell
```

Record the job ID as `$JOBID`.

- [ ] **Step 2: Serve Qwen3-8B with OpenPipe tokenizer**

```bash
TOKENIZER_PATH="/shared/matan/hub/models--OpenPipe--Qwen3-14B-Instruct/snapshots/99e4a359e990d8d00b77ac707c34bbfbc9d8c98a"

srun --jobid=$JOBID --overlap --gpus=2 \
    bash -c "source /shared/matan/miniconda3/bin/activate hope && \
    vllm serve /shared/matan/models/Qwen3-8B \
        --served-model-name Qwen3-8B-OpenPipe \
        --tokenizer $TOKENIZER_PATH \
        --tensor-parallel-size 2 \
        --port 8300 \
        --max-model-len 32768" &
```

Wait for ready:
```bash
until curl -sf http://better-ginkgo-dragonfly:8300/v1/models | grep -q model; do sleep 5; done
echo "Q8B vLLM READY"
```

- [ ] **Step 3: Run inspect with CovertBackdoorPlain prompt options**

```bash
srun --jobid=$JOBID --overlap \
    bash -c 'cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad && \
    export OPENROUTER_API_KEY=sk-or-XXX && \
    PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:$PYTHONPATH \
    /home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/inspect_few_examples.py \
        --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
        --actor std_setup_factored/LLMs/qwen3_8b_openpipe_local.py \
        --monitor std_setup_factored/LLMs/gpt_oss_120b.py \
        --formatter-class std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
        --split rl_val_q4bi \
        --actor-prompt-options '\''["wo_post_hoc_self_analysis", "minimal_covertness_instructions", "md_output", "backdoor_keyword_is_backdoor"]'\'' \
        --monit-prompt-options '\''["backdoor_task_unspecific"]'\'' \
        --epochs 10 \
        --entire-split'
```

- [ ] **Step 4: Verify output**

Check `scored_generations.jsonl` has 270 lines with non-empty outputs (output len > 0). Record the log path for shortcuts.

```bash
head -1 <LOG_DIR>/scored_generations.jsonl | python3 -c "
import sys,json,ast
s = json.loads(sys.stdin.readline())
gen = ast.literal_eval(s['generation']) if isinstance(s['generation'], str) else s['generation']
resp = ast.literal_eval(gen['response']) if isinstance(gen['response'], str) else gen['response']
print(f'output len: {len(resp.get(\"output\",\"\") or \"\")}')"
```

---

### Task 4: Add shortcuts to shortcuts.txt

**Files:**
- Modify: `/shared/matan/code/APPS_inference_lim_hidden_scratchpad/experiments/shortcuts.txt` (after line 532)

Depends on Tasks 1-3 completing so we know the output paths.

- [ ] **Step 1: Add shortcuts after line 532**

Add these lines after the existing `Q8BNT_OSS120/RLSimpleStep800` line (line 532), before the blank line 533:

```
Q4BI_Q80/RLHiddenQ80MonitorStep820            | logs/Tasks/APPSCovertBackdoorTask/05/04/inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34
Q4BI_OSS120/RLHiddenQ80MonitorStep820         | <OSS120_RESCORE_PATH_FROM_TASK_1>
Q4BI_O4M/RLHiddenQ80MonitorStep820            | <O4M_RESCORE_PATH_FROM_TASK_2>
Q8BNT_OSS120/CovertBackdoorPlain_v2           | <INSPECT_PATH_FROM_TASK_3>
```

Replace the `<...>` placeholders with actual paths from Tasks 1-3.

- [ ] **Step 2: Verify shortcuts**

```bash
grep "Q4BI\|CovertBackdoorPlain_v2" /shared/matan/code/APPS_inference_lim_hidden_scratchpad/experiments/shortcuts.txt
```

---

### Task 5: Check and relaunch training runs

**Files:**
- lvym5f8i: `logs/VerlRun/05/04/cont3_fork_2evs9808_q80_monitor_17_13_lvym5f8i/`
- 66m8l3iu: `logs/VerlRun/05/04/fork_sjm53zh8_baseline_prompt_05_24_66m8l3iu/`

- [ ] **Step 1: Check if training jobs are still alive**

```bash
squeue -u $(whoami) -o "%.10i %.2t %.10M %.20R %.50j"
```

If lvym5f8i is dead (no job): it had 0 checkpoints saved (the crash wiped them). Need to continue from fcrvicwq's step 520:

```bash
# Verify latest_checkpointed_iteration.txt says 520
cat logs/VerlRun/05/04/cont2_fork_2evs9808_q80_monitor_04_12_fcrvicwq/checkpoints/latest_checkpointed_iteration.txt
# Should say 520. If not, update it.
```

- [ ] **Step 2: Relaunch lvym5f8i from fcrvicwq@520**

Use the VFH orchestrator (do NOT edit sbatch manually):

```bash
cd /shared/matan/code/pass_at_k && source .venv/bin/activate && \
python -m vfh.orchestrator continue \
    --run-dir logs/VerlRun/05/04/cont2_fork_2evs9808_q80_monitor_04_12_fcrvicwq \
    --extra-overrides trainer.save_freq=40 \
    --slurm.sbatch --slurm.time 24:00:00 \
    --slurm.node 1 --slurm.divide-resources-by 4 \
    --slurm.taskset 80-159 -y
```

- [ ] **Step 3: Check 66m8l3iu (BioMath)**

If dead, relaunch similarly:

```bash
python -m vfh.orchestrator continue \
    --run-dir logs/VerlRun/05/04/fork_sjm53zh8_baseline_prompt_05_24_66m8l3iu \
    --slurm.sbatch --slurm.time 24:00:00 \
    --slurm.node 1 --slurm.divide-resources-by 4 \
    --slurm.taskset 0-79 -y
```

- [ ] **Step 4: Watch relaunched jobs**

Use `~/.claude/scripts/watch_slurm_job.sh` with Monitor tool. Check that stdout advances within 20 minutes.

---

### Task 6: Eval pipeline for lvym5f8i when training finishes

**Blocked on:** Task 5 (training must complete to step 700)

This repeats the same flow as the 4B eval but on the newly trained checkpoint:

- [ ] **Step 1: Merge the final checkpoint**

```bash
cd /shared/matan/code/pass_at_k && source .venv/bin/activate && \
python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
    --local_dir <LVYM5F8I_RUN_DIR>/checkpoints/global_step_700/actor \
    --target_dir /shared/matan/tmp/merged_lvym5f8i_step_700
```

- [ ] **Step 2: Serve with vLLM and run inspect**

Same as the 4B flow: serve on port 8001, `--max-model-len 16384`, then run `inspect_few_examples.py` with `rl_no_hidden_simple_backdoor_prompt` + Q80, then rescore with OSS120 and O4M.

- [ ] **Step 3: Add shortcuts for these results**

---

## Execution Order

```
Task 1 (OSS120 rescore)  ──┐
                            ├──▶ Task 4 (shortcuts)
Task 2 (O4M rescore)     ──┤
                            │
Task 3 (8B eval)          ──┘

Task 5 (relaunch training) ──▶ Task 6 (eval when done)
```

Tasks 1, 2 need no GPU — can run from any node or locally.
Task 3 needs a salloc with 2 GPUs.
Task 5 needs sbatch (uses VFH orchestrator).
Tasks 1+2 can run in parallel. Task 3 can run in parallel with 1+2 if GPUs are available.
