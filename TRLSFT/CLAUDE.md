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
- Checkpoints save LoRA adapters only (lightweight)
- Registers run with vfh run tracker (`vfh.run_tracker.register_run`)
- **`StdoutLoggingCallback`**: prints loss/lr/epoch/grad_norm to stdout at each log step
- **`MidRunEvalCallback`**: runs HF `model.generate()` eval every `eval_steps`, scores with reward function, logs to wandb + `mid_run_evals/step_{N}.jsonl`

### Eval architecture
Two eval approaches:
1. **Mid-run eval** (`MidRunEvalCallback` in `sft_train.py`): runs during training every `eval_steps` using HF generate (model already in GPU memory)
2. **Post-hoc multi-checkpoint eval** (`runners/eval_all_checkpoints.py`): evaluates ALL LoRA checkpoints from a completed run using vLLM

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
├── checkpoints/          # LoRA adapters per save_steps
│   ├── checkpoint-16/
│   ├── checkpoint-32/
│   └── final_adapter/
├── mid_run_evals/        # per-checkpoint eval during training (HF generate)
│   ├── step_16.jsonl
│   └── step_32.jsonl
├── post-hoc-evals/       # multi-checkpoint eval results (vLLM, after training)
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
4. Runs training (LoRA SFT) with mid-run eval at every `eval_steps`
5. Saves final adapter
6. Saves loss plot

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
| `sft_train.py` | Training + sbatch + loss plot + mid-run eval via callbacks |
| `prepare_data.py` | Rollout JSONL → HF Dataset |
| `sft_eval_base.py` | Eval ABC |
| `sft_eval_backdoor.py` | Eval class for post-hoc eval (vLLM generation + reward scoring) |
| `envs/apps_backdoor.py` | APPS backdoor eval env: question loading, vLLM client, scoring, summary stats |
| `runners/eval_all_checkpoints.py` | Multi-checkpoint post-hoc eval via vLLM multi-LoRA serving |
| `configs/base/` | Base configs (hyperparams) |
| `configs/overrides/` | Per-run overrides (model, data, eval) |

## Design choices
- **LoRA, not full fine-tuning** — 8B model, checkpoints are ~50MB not ~16GB
- **Mid-run eval via HF generate** — `MidRunEvalCallback` runs eval every `eval_steps` using `model.generate()` (no vLLM). Model is already in GPU memory during training. Results saved to `mid_run_evals/step_{N}.jsonl` and logged to wandb.
- **vLLM for post-hoc eval** — `runners/eval_all_checkpoints.py` uses vLLM multi-LoRA serving for batch comparison of all checkpoints after training
- **`completion_only_loss=True`** — TRL 0.29+ config param, trains only on assistant completions
- **`StdoutLoggingCallback`** — prints loss/lr/epoch to stdout so sbatch `.out` files contain training progress
- **All paths should be absolute** — relative paths break when sbatch runs from a different CWD
- **Fresh wandb ID per execution** — `run-prepared` generates new ID even on retry, preventing wandb pollution

## Known issues / TODOs
- **Checkpoint naming** — uses HF default `checkpoint-{step}` (not `global_step_{N}`). Kept as-is to avoid breaking HF Trainer's `save_total_limit` cleanup.
- **Mid-run eval OOM risk** — HF generate allocates KV cache on top of training memory. `gen_batch_size=4` keeps it small, but very long sequences (8192 tokens) may still OOM. If so, reduce `gen_batch_size` or `n_samples` in eval config.
- **No LoRA merge at end of training** — removed in favor of post-hoc eval + manual fusion. Use `eval_all_checkpoints.py` or manual PEFT merge to create standalone HF models.
