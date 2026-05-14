# Verl on Modal — Incremental Plan

**Goal:** Run verl RL training (Qwen3-8B, APPS dataset) on Modal with 4x RTX PRO 6000 GPUs.

**Budget:** $200 Modal compute. Track via `modal app list` + Modal dashboard.

**Approach:** Bypass VFH orchestrator — call verl directly with resolved Hydra overrides. The orchestrator uses `os.execvp`, daemons, and output tee which don't fit Modal's function model.

## Key Risks

| Risk | Mitigation | Validated at step |
|------|-----------|-------------------|
| Ray inside Modal container | Test `ray.init()` sees GPUs | Step 3 |
| Multi-GPU Ray workers | Test 4-GPU Ray cluster in one container | Step 4 |
| vLLM rollout engine | Test vLLM inference with TP | Step 4 |
| verl's Hydra config system | Pass overrides directly, skip orchestrator | Step 5 |
| Checkpoint persistence | Modal Volume for checkpoints | Step 5 |
| HF model/data access | Mount HF cache or download in image | Step 3 |

## Steps

### Step 1: CPU-only image build + `import verl`
- Build Modal image with deps from `scripts/install_vllm_sglang_mcore.sh`
- Clone this repo into the image (verl is local, not pip-installable)
- Verify `import verl` succeeds
- **Est. cost:** ~$0.01

### Step 2: Single GPU — torch.cuda + matmul
- Request 1x RTX PRO 6000
- `torch.cuda.is_available()`, run `torch.matmul` on GPU
- **Est. cost:** ~$0.05

### Step 3: Single GPU — Ray + model load
- `ray.init()` inside container, verify it sees GPU
- Load Qwen3-8B with transformers, run forward pass
- **Est. cost:** ~$0.50

### Step 4: 4 GPU — Ray multi-GPU + vLLM
- 4x RTX PRO 6000
- Ray sees all 4 GPUs
- vLLM serves Qwen3-8B with TP=4, run sample generation
- **Est. cost:** ~$1-2

### Step 5: 4 GPU — verl dry run
- Run verl training loop for 1 step, tiny batch
- Resolve Hydra config from `default_APPS_code.json5`
- Mount Volume for checkpoints
- **Est. cost:** ~$5-10

### Step 6: Full training run
- Real config, proper batch sizes, wandb logging
- **Est. cost:** ~$20-50

## Architecture

```
Local machine                          Modal
─────────────                          ─────
modal run modal_verl.py::step_N   →   Container boots (pre-built image)
                                       ├── 4x RTX PRO 6000
                                       ├── repo cloned + installed
                                       ├── HF cache volume mounted
                                       ├── ray.init() (local cluster)
                                       └── verl training via Hydra CLI
                                           ├── actor (FSDP across 4 GPUs)
                                           ├── rollout (vLLM, TP=1 or 4)
                                           └── ref (FSDP)
Checkpoints ←── Modal Volume ──→ Persistent across runs
```

## Config Adaptation for Modal

Changes from `default_APPS_code.json5`:
- `trainer.n_gpus_per_node`: 4 (matches RTX PRO 6000 count)
- `trainer.nnodes`: 1
- `data.train_files` / `val_files`: paths inside container
- `actor_rollout_ref.model.path`: HF model name or container-local path
- Remove wandb logger initially (add back once basic loop works)

## Cost Tracking

After each step, record:
- Modal app ID
- Duration
- Estimated cost (from dashboard)
- Cumulative spend
