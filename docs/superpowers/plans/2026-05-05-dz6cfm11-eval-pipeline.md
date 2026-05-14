# dz6cfm11 (APPS Q80 Monitor) Eval Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the final checkpoint of training run dz6cfm11 (APPS Q80 monitor, step 700), run the standard eval battery (inspect + rescore with Q80/OSS120/O4M monitors), add shortcuts, and human-review prior eval results.

**Architecture:** Sequential tasks — merge needs no GPU, vLLM serving needs 1 GPU (4B model), inspect + rescore need API keys. Run rescores one at a time to avoid OpenRouter 403 rate limits. All shortcuts are commented out pending human review.

**Tech Stack:** vLLM, OpenRouter API (Q80/OSS120), OpenAI API via OpenRouter (O4M), APPS_inference_lim_hidden_scratchpad eval framework

---

## Pitfalls (learned this session)

- **OPENROUTER_API_KEY** must be explicitly exported: `export OPENROUTER_API_KEY=sk-or-XXX`
- **OpenRouter 403 rate limits**: Run rescores sequentially (not in parallel with other API consumers). The key has burst limits.
- **vLLM max-model-len**: Use `--max-model-len 16384` for 4B models.
- **Qwen3 thinking suppression**: The 4B actor (`qwen3_4b_I_local.py`) already has `chat_template_kwargs: {"enable_thinking": false}` — do NOT revert.
- **4B actor port**: `qwen3_4b_I_local.py` uses `http://localhost:8001/v1`.
- **Rescore `--yes` flag**: Required to skip interactive confirmation.
- **O4M uses OpenRouter** (not OpenAI API directly).
- **salloc time**: Use 4hr+ for eval work.

---

### Task 1: Comment out existing shortcuts pending human review

**Files:**
- Modify: `/shared/matan/code/APPS_inference_lim_hidden_scratchpad/experiments/shortcuts.txt:533-537`

- [x] **Step 1: Comment out the 4 shortcuts added in this session**

Already done. Lines 533-537 now read:
```
# TODO: uncomment after human review of eval results
#Q4BI_Q80/RLHiddenQ80MonitorStep820            | logs/Tasks/APPSCovertBackdoorTask/05/04/inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34
#Q4BI_OSS120/RLHiddenQ80MonitorStep820         | logs/Tasks/APPSCovertBackdoorTask/05/05/rescore_test_inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34+Qwen3_4B_local+GPTOSS120B+backdoor_task_unspecific_01_17
#Q4BI_O4M/RLHiddenQ80MonitorStep820            | logs/Tasks/APPSCovertBackdoorTask/05/05/rescore_test_inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34+Qwen3_4B_local+GPT4oMini+backdoor_task_unspecific_01_23
#Q8BNT_OSS120/CovertBackdoorPlain_v2           | logs/Tasks/APPSCovertBackdoorTask/05/05/inspect_test_entire_qwen3_8b_openpipe_local_gpt_oss_120b_00_57
```

- [ ] **Step 2: Human review of existing eval results**

Review these result dirs for sanity (non-zero outputs, reasonable sus scores, correct sample counts):
1. `logs/Tasks/APPSCovertBackdoorTask/05/04/inspect_test_entire_qwen3_4b_I_local_qwen3_next_80b_18_34` — Q4BI Q80 (77.2% test pass, 4.5% sus, 20.7% backdoor)
2. `logs/Tasks/APPSCovertBackdoorTask/05/05/..._GPTOSS120B+backdoor_task_unspecific_01_17` — Q4BI OSS120 (16.8% sus, AUC 0.74-0.87)
3. `logs/Tasks/APPSCovertBackdoorTask/05/05/..._GPT4oMini+backdoor_task_unspecific_01_23` — Q4BI O4M (3.2% sus, AUC 0.52-0.61)
4. `logs/Tasks/APPSCovertBackdoorTask/05/05/inspect_test_entire_qwen3_8b_openpipe_local_gpt_oss_120b_00_57` — Q8BNT CovertBackdoorPlain_v2 (52.3% sus, 42.7% backdoor)

- [ ] **Step 3: Uncomment shortcuts after review passes**

Uncomment lines 534-537 in shortcuts.txt (remove the `#` prefix from each line).

---

### Task 2: Merge dz6cfm11 step 700 checkpoint

**Files:**
- Input: `/shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont3_fork_2evs9808_q80_monitor_22_06_dz6cfm11/checkpoints/global_step_700/actor`
- Output: `/shared/matan/tmp/merged_dz6cfm11_step_700`

No GPU needed — CPU-only merge operation.

- [ ] **Step 1: Verify checkpoint exists**

```bash
ls /shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont3_fork_2evs9808_q80_monitor_22_06_dz6cfm11/checkpoints/global_step_700/actor/
```

Expected: 4 model weight files (world_size=4).

- [ ] **Step 2: Run FSDP merge**

```bash
source .venv/bin/activate && \
python -m verl.model_merger merge --backend fsdp --tie-word-embedding \
    --local_dir logs/VerlRun/05/04/cont3_fork_2evs9808_q80_monitor_22_06_dz6cfm11/checkpoints/global_step_700/actor \
    --target_dir /shared/matan/tmp/merged_dz6cfm11_step_700
```

- [ ] **Step 3: Verify merged model**

```bash
ls /shared/matan/tmp/merged_dz6cfm11_step_700/*.safetensors | wc -l
```

Expected: model weight files present.

---

### Task 3: Serve merged model and run inspect with Q80 monitor

**Files:**
- Actor: `/shared/matan/code/APPS_inference_lim_hidden_scratchpad/std_setup_factored/LLMs/qwen3_4b_I_local.py` (points to `localhost:8001`)
- Monitor: Q80 via OpenRouter
- Output: new dir under `logs/Tasks/APPSCovertBackdoorTask/05/05/`

Needs 1 GPU for vLLM. Prioritize bleak-mushroom-dove, fallback to better-ginkgo-dragonfly.

- [ ] **Step 1: Get a salloc**

```bash
salloc --nodelist=bleak-mushroom-dove --gpus-per-node=1 --cpus-per-task=20 --mem=100G --time=04:00:00 --no-shell
```

Record the job ID as `$JOBID`.

- [ ] **Step 2: Serve merged model with vLLM**

```bash
srun --jobid=$JOBID --overlap --gpus=1 \
    bash -c "source /shared/matan/miniconda3/bin/activate hope && \
    vllm serve /shared/matan/tmp/merged_dz6cfm11_step_700 \
        --served-model-name ../matan/models/Qwen3-4B-I \
        --port 8001 \
        --max-model-len 16384" &
```

Wait for ready:
```bash
until srun --jobid=$JOBID --overlap bash -c "curl -sf http://localhost:8001/v1/models" 2>/dev/null | grep -q model; do sleep 10; done
echo "vLLM READY"
```

- [ ] **Step 3: Run inspect with Q80 monitor**

```bash
srun --jobid=$JOBID --overlap \
    bash -c 'cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad && \
    export OPENROUTER_API_KEY=sk-or-XXX && \
    PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:$PYTHONPATH \
    /home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/inspect_few_examples.py \
        --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
        --actor std_setup_factored/LLMs/qwen3_4b_I_local.py \
        --monitor std_setup_factored/LLMs/qwen3_next_80b.py \
        --formatter-class std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
        --split rl_val_q4bi \
        --actor-prompt-options '\''["rl_no_hidden_simple_backdoor_prompt"]'\'' \
        --monit-prompt-options '\''["backdoor_task_unspecific"]'\'' \
        --epochs 10 \
        --entire-split'
```

- [ ] **Step 4: Verify output**

```bash
# Find the new output dir
INSPECT_DIR=$(ls -td /shared/matan/code/APPS_inference_lim_hidden_scratchpad/logs/Tasks/APPSCovertBackdoorTask/05/05/*qwen3_4b_I_local*qwen3_next_80b*/ | head -1)
echo "Output: $INSPECT_DIR"
wc -l "$INSPECT_DIR/scored_generations.jsonl"
```

Expected: 270 lines. Record `$INSPECT_DIR` for shortcuts.

- [ ] **Step 5: Release salloc**

```bash
scancel $JOBID
```

---

### Task 4: Rescore with OSS120 monitor

**Files:**
- Input: `$INSPECT_DIR` from Task 3
- Monitor: `std_setup_factored/LLMs/gpt_oss_120b.py`

No GPU needed. Run ALONE (no concurrent API consumers) to avoid 403.

- [ ] **Step 1: Run OSS120 rescore**

```bash
cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad && \
export OPENROUTER_API_KEY=sk-or-XXX && \
PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:$PYTHONPATH \
/home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/rescore_samples_wpromptdict.py \
    --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
    --path $INSPECT_DIR \
    --monitor std_setup_factored/LLMs/gpt_oss_120b.py \
    --monit-prompt-options '["backdoor_task_unspecific"]' \
    --formatter std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
    --yes
```

- [ ] **Step 2: Verify rescore**

Check output has 270 lines and **non-zero sus_score**:

```bash
OSS120_DIR=$(ls -td /shared/matan/code/APPS_inference_lim_hidden_scratchpad/logs/Tasks/APPSCovertBackdoorTask/05/05/*GPTOSS120B*/ | head -1)
python3 -c "
import json
data = [json.loads(l) for l in open('${OSS120_DIR}/scored_generations.jsonl')]
sus = [d['sus_score'] for d in data]
print(f'Total: {len(data)}, sus mean: {sum(sus)/len(sus):.3f}, non-zero: {sum(1 for s in sus if s > 0)}/{len(sus)}')
"
```

If sus mean is 0.0, the monitor failed — check for 403 errors and re-run.

---

### Task 5: Rescore with O4M monitor

**Files:**
- Input: `$INSPECT_DIR` from Task 3
- Monitor: `std_setup_factored/LLMs/gpt_4o_mini.py`

No GPU needed. Run AFTER Task 4 completes (sequential to avoid 403).

- [ ] **Step 1: Run O4M rescore**

```bash
cd /shared/matan/code/APPS_inference_lim_hidden_scratchpad && \
export OPENROUTER_API_KEY=sk-or-XXX && \
PYTHONPATH=/shared/matan/code/APPS_inference_lim_hidden_scratchpad:$PYTHONPATH \
/home/matan/shared/.conda/envs/hope/bin/python std_setup_factored/tasks/tests/rescore_samples_wpromptdict.py \
    --task std_setup_factored/tasks/APPS/APPSCovertBackdoorTask.py \
    --path $INSPECT_DIR \
    --monitor std_setup_factored/LLMs/gpt_4o_mini.py \
    --monit-prompt-options '["backdoor_task_unspecific"]' \
    --formatter std_setup_factored/tasks/APPS/ResponseFormatter/RemoveAfterCodeFormatter.py \
    --yes
```

- [ ] **Step 2: Verify rescore**

Same check as Task 4 Step 2 but for O4M dir.

---

### Task 6: Add shortcuts (commented out for human review)

**Files:**
- Modify: `/shared/matan/code/APPS_inference_lim_hidden_scratchpad/experiments/shortcuts.txt`

Depends on Tasks 3-5 completing so we know output paths.

- [ ] **Step 1: Add commented shortcuts after line 537**

Add these lines after the existing commented block (line 537), before the blank line:

```
# TODO: uncomment after human review — dz6cfm11 step 700 eval results
#Q4BI_Q80/RLHiddenQ80MonitorStep700_dz6cfm11   | <INSPECT_DIR_FROM_TASK_3>
#Q4BI_OSS120/RLHiddenQ80MonitorStep700_dz6cfm11 | <OSS120_DIR_FROM_TASK_4>
#Q4BI_O4M/RLHiddenQ80MonitorStep700_dz6cfm11   | <O4M_DIR_FROM_TASK_5>
```

Replace `<...>` placeholders with actual paths from Tasks 3-5.

- [ ] **Step 2: Verify shortcuts**

```bash
grep "dz6cfm11" /shared/matan/code/APPS_inference_lim_hidden_scratchpad/experiments/shortcuts.txt
```

---

## Execution Order

```
Task 1 (comment shortcuts)  ── DONE

Task 2 (merge checkpoint)
    │
    ▼
Task 3 (serve + inspect Q80)
    │
    ▼
Task 4 (rescore OSS120)     ── sequential, no parallel API
    │
    ▼
Task 5 (rescore O4M)        ── sequential, no parallel API
    │
    ▼
Task 6 (add shortcuts)
```

Task 1 is already complete.
Tasks 4 and 5 MUST run sequentially (not parallel) to avoid OpenRouter 403 rate limits.
