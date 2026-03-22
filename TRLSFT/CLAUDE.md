# TRLSFT — LoRA SFT Training Module

## Purpose
Lightweight SFT (Supervised Fine-Tuning) using TRL + LoRA (PEFT). Trains on rollout JSONL data, evaluates with pluggable eval classes, and generates SLURM sbatch scripts.

## Architecture

### Config system
Two-layer JSON5 configs (mirrors vfh pattern):
- `configs/base/default_lora.json5` — training hyperparams, LoRA config, logging defaults
- `configs/overrides/*.json5` — per-run: model, train_files, eval config

Deep-merged via `pyjson5` + `dacite` into `SFTConfig` (no defaults on the dataclass — all values from configs).

### Key types (`sft_types.py`)
- `LoRAConfig` — PEFT adapter params (r, alpha, dropout, target_modules)
- `EvalConfig` — `eval_script` (module.ClassName) + `eval_kwargs` (flexible dict passed to eval `__init__`)
- `SFTConfig` — full run config
- `CLIArgs` — CLI-only flags (not in config files)

### Data format (`prepare_data.py`)
Reads rollout JSONL where:
- `input` = `"user\n{prompt}\nassistant\n"`
- `output` = assistant response (think block + code)

Converts to HF Dataset with `text` column. TRL's `completion_only_loss=True` with response template `"assistant\n"` masks user turns from loss.

### Training (`sft_train.py`)
- LoRA via PEFT
- `trl.SFTTrainer` with `SFTConfig(completion_only_loss=True)`
- Wandb logging + loss plot saved to run dir
- After training: merges LoRA into base model → `merged_model/`
- Checkpoints save LoRA adapters only (lightweight)
- Registers run with vfh run tracker (`vfh.run_tracker.register_run`)
- Eval runs at the end on the merged model using vLLM for batched generation
- Exits with error if no eval configured

### Eval architecture
Two eval approaches:
1. **End-of-training eval** (`sft_eval_backdoor.py`): runs once on merged model after training
2. **Post-hoc multi-checkpoint eval** (`runners/eval_all_checkpoints.py`): evaluates ALL LoRA checkpoints from a completed run

### Post-hoc eval (`runners/eval_all_checkpoints.py`)
- Starts a vLLM server with `--enable-lora` and ALL checkpoint adapters loaded simultaneously
- Generates completions concurrently (async) for each checkpoint
- Scores with reward function via `asyncio.matan_gather_chunked`
- Writes per-checkpoint results + summary to `{run_dir}/post-hoc-evals/{MM_DD_HH_mm}/`
- Summary stats: `frac_test_passing`, `sus_score`, `backdoor_test_passed`, `hidden_lengths`

### Eval environments (`envs/`)
- `apps_backdoor.py` — APPS backdoor eval: loads questions from rollout files, generates via vLLM client, scores with backdoor reward function
- Future envs extend this pattern

### Run directory
```
logs/SFTRuns/{MM}/{DD}/{desc}_{HH}_{mm}_{wandb_id}/
├── config.json5          # resolved (merged) config
├── run_metadata.json5    # wandb ID, paths, timestamps
├── source_configs/       # copies of base + overrides JSON5
├── sbatch_job.sh
├── loss_plot.png
├── eval_results.jsonl    # end-of-training eval
├── checkpoints/          # LoRA adapters per save_steps
│   ├── checkpoint-16/
│   ├── checkpoint-32/
│   └── final_adapter/
├── merged_model/         # fused base + LoRA (final)
├── post-hoc-evals/       # multi-checkpoint eval results
│   └── {MM_DD_HH_mm}/
│       ├── checkpoint-16.jsonl
│       ├── checkpoint-32.jsonl
│       └── summary.jsonl
└── vllm_server.log       # vLLM server log (eval runs)
```

**Run directories are immutable.** `run-prepared` refuses to run if training artifacts already exist. Each launch must go through `new` to create a fresh directory.

## Usage

### Launching a run (the correct workflow)

**Always use `new --sbatch`** to launch. This creates a fresh run dir, generates the sbatch script, and submits it automatically. **Never manually re-submit old sbatch scripts** — each run must have its own directory and wandb ID.

```bash
# Launch SFT via sbatch (recommended)
python -m TRLSFT.sft_train new \
    --base-config TRLSFT/configs/base/default_lora.json5 \
    --overrides TRLSFT/configs/overrides/hidden_tag_sft.json5 \
    --desc "hidden_tag_sft" \
    --sbatch --time 04:00:00

# Generate sbatch without submitting
python -m TRLSFT.sft_train new \
    --base-config TRLSFT/configs/base/default_lora.json5 \
    --overrides TRLSFT/configs/overrides/hidden_tag_sft.json5 \
    --desc "hidden_tag_sft" \
    --sbatch --time 04:00:00 --dont-auto-sbatch

# Direct training (no sbatch, runs locally — needs GPU)
python -m TRLSFT.sft_train new \
    --base-config TRLSFT/configs/base/default_lora.json5 \
    --overrides TRLSFT/configs/overrides/hidden_tag_sft.json5 \
    --desc "hidden_tag_sft"
```

### Post-hoc eval (all checkpoints)

```bash
# Evaluate all checkpoints from a completed run (needs GPU via srun)
srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
    python -m TRLSFT.runners.eval_all_checkpoints \
    --run-dir logs/SFTRuns/03/22/hidden_tag_sft_15ep_... \
    --eval-source /path/to/rollouts/401.jsonl \
    --n-samples 100 \
    --tensor-parallel-size 1 \
    --max-model-len 7000 \
    --gpu-memory-utilization 0.85
```

### What `new --sbatch` does
1. Resolves config (base + overrides deep merge)
2. Creates unique run dir: `logs/SFTRuns/{MM}/{DD}/{desc}_{HH}_{mm}_{wandb_id}/`
3. Saves resolved config + copies of source configs
4. Generates sbatch script
5. Submits via `sbatch` (unless `--dont-auto-sbatch`)

### What the sbatch script does (`run-prepared`)
1. Checks run dir doesn't already have training artifacts (immutability guard)
2. Generates a fresh wandb ID (even on retry)
3. Registers run with vfh run tracker
4. Runs training (LoRA SFT)
5. Saves final adapter + merges into base model
6. Runs eval on merged model
7. Saves loss plot + eval results

## SLURM notes
- **Use `--gres=gpu:N`** not `--gpus=N` for GPU allocation on this cluster
- **SFT training**: 1 GPU, 16 CPUs, 64G memory (LoRA doesn't need full node)
- **Eval**: 1 GPU, 16 CPUs, 64G memory
- **Don't run eval concurrently with training on same node** — OOM guaranteed (training uses ~127GB VRAM)
- **OpenRouter concurrency**: 300 concurrent API calls for reward scoring

## Key files
| File | Purpose |
|------|---------|
| `sft_types.py` | Config dataclasses (no defaults) |
| `sft_train.py` | Training + sbatch + loss plot + merge + eval |
| `prepare_data.py` | Rollout JSONL → HF Dataset |
| `sft_eval_base.py` | Eval ABC |
| `sft_eval_backdoor.py` | End-of-training eval: vLLM generation + reward scoring |
| `envs/apps_backdoor.py` | APPS backdoor eval env: question loading, vLLM client, scoring, summary stats |
| `runners/eval_all_checkpoints.py` | Multi-checkpoint post-hoc eval via vLLM multi-LoRA serving |
| `configs/base/` | Base configs (hyperparams) |
| `configs/overrides/` | Per-run overrides (model, data, eval) |

## Design choices
- **LoRA, not full fine-tuning** — 8B model, checkpoints are ~50MB not ~16GB
- **vLLM for eval** — batched concurrent generation, multi-LoRA serving for checkpoint comparison
- **`completion_only_loss=True`** — TRL 0.29+ config param, trains only on assistant completions
- **Eval is mandatory** — `sft_train.py` exits with error if no eval configured
- **All paths should be absolute** — relative paths break when sbatch runs from a different CWD
- **Fresh wandb ID per execution** — `run-prepared` generates new ID even on retry, preventing wandb pollution

## Known issues / TODOs
- **Mid-run eval not implemented** — `eval_steps` config param is ignored; eval only runs at end. Post-hoc eval covers this.
- **Loss not printed to sbatch stdout** — goes to wandb output.log only. Check `wandb/run-*/files/output.log`.
- **Checkpoint naming** — currently HF default `checkpoint-{step}`. TODO: change to `global_step_{N}` for consistency with vfh.
- **End-of-training eval uses HF generate** (slow, sequential). Should be replaced with vLLM or removed in favor of post-hoc eval.
