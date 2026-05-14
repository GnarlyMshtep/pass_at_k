# Blackwell (RTX PRO 6000, sm_120) Attempts Log

**Goal:** Determine if verl runs on Blackwell and measure how much back-bending required.

**The problem:** RTX PRO 6000 is sm_120 (Blackwell). The default verl stack is torch 2.6.0 + CUDA 12.4, which has no compiled kernels for sm_120. Need torch ≥2.7 + CUDA ≥12.8.

## Dependency chain

```
verl  ──→ vllm ──→ flash-attn, flashinfer ──→ torch + CUDA
 0.5      0.8.5?    2.7.4.post1, 0.2.2      2.6.0 + cu124
```

Need to bump each to Blackwell-compatible version, minimize API drift in each step.

## Attempts

### Attempt 1 (2026-04-23 14:08) — baseline (torch 2.6+cu124)
- **GPU**: RTX PRO 6000 (sm_120)
- **Stack**: As original install script
- **Result**: ❌ `torch.randn(..., device="cuda")` → `CUDA error: no kernel image is available`
- **Conclusion**: Confirmed Blackwell mismatch. Need torch 2.7+cu128.

### Attempt 2 (2026-04-23 14:17) — torch 2.7.1+cu128 only
- **Image**: `blackwell_torch_image` = debian_slim + torch 2.7.1+cu128 (from pytorch.org cu128 index)
- **Result**: ✅ SUCCESS
  - `CUDA arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']`
  - `NVIDIA RTX PRO 6000 Blackwell Server Edition`, sm_120
  - Matmul correct, BF16 13.6 TFLOPS (low but functional)
- **Conclusion**: Blackwell accessible with prebuilt torch 2.7.1+cu128 wheel. No source build needed.

### Attempt 3 (2026-04-23 14:20) — add flash-attn 2.8.3 + flashinfer 0.6.8
- **Image**: Attempt 2 + `flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE` wheel + `flashinfer-python==0.6.8`
- **Result**: ⚠️ PARTIAL
  - ✅ `flash_attn.flash_attn_func` works on Blackwell (output correct)
  - ⚠️ `flashinfer` imports but warns `Failed to get device capability: SM 12.x requires CUDA >= 12.9`
  - We have torch 2.7.1+cu128 → CUDA 12.8 runtime. flashinfer wants 12.9+ for Blackwell.
- **Implication**: flashinfer's Blackwell fast paths disabled. vLLM uses flashinfer for attention → could break rollout.
- **Next**: bump to torch 2.8+cu129 (if wheels exist).

### Attempt 4 (2026-04-23 14:23) — torch 2.8+cu129
- **Image**: torch 2.8.0+cu129, flash-attn 2.8.3 (torch2.8 wheel), flashinfer 0.6.8
- **Result**: ✅ flashinfer warning GONE. flash-attn still works. No regressions.
- **Remaining noise**: tvm_ffi warning about torch-c-dlpack-ext JIT — non-critical.

### Attempt 5 (2026-04-23 14:30) — add vLLM 0.11.2
- **Image**: Attempt 4 + `vllm==0.11.2` + transformers/tokenizers
- **Test**: `LLM(model="Qwen/Qwen2.5-0.5B-Instruct", tp=1, bf16)` + generate
- **Result**: ✅ SUCCESS. CUDA graphs captured, inference generated coherent text.
- **Throughput**: 0.95 tok/s for 30 tokens, but most of the 31s was CUDA graph capture (normal for first request). Second request would be fast.
- **Implication**: Blackwell + torch 2.8 + cu129 + flash-attn 2.8.3 + flashinfer 0.6.8 + vllm 0.11.2 is a working stack through inference.

### Attempt 6 (2026-04-23 16:58) — vllm 0.11.2 → verl breaks
- **Image**: torch 2.8+cu129 + vllm 0.11.2 + flash-attn 2.8.3 + verl
- **Result**: ❌ verl.workers.rollout.vllm_rollout fails — `cannot import name 'CompilationLevel' from 'vllm.config'`
- **Cause**: vllm 0.11 turned `vllm.config` from a module into a package; CompilationLevel moved.

### Attempt 7 (2026-04-23 16:48) — vllm 0.9.2 → wrong CUDA
- **Cause**: vllm 0.9.2 pins `torch==2.7.0` which PyPI resolves to `+cu126`. CUDA 12.6 has sm_90 max (no Blackwell).
- **Lesson**: pre-install torch with `extra_index_url=.../cu128` so vllm's pin matches the cu128 variant.

### Attempt 8 (2026-04-23 17:00) — vllm 0.10.2 with torch 2.8+cu128 — ✅ WINNER
- **Stack**: torch 2.8.0+cu128, vllm 0.10.2, flash-attn 2.8.3 (cu12torch2.8), flashinfer (via vllm), verl 0.5.0.dev0
- **Result**: ✅ ALL verl submodules import, including `verl.workers.rollout.vllm_rollout.vllm_rollout_spmd`
- **Gotcha**: had to let hydra-core bring its own antlr4-python3-runtime (pinning 4.9.3 broke it — needs 4.13+)
- **Answer to "how much back-bending"**: Moderate — ~3 pip-install layering tweaks and one vllm version pick. No verl source patches needed yet.

### Next: actually run something
- Instantiate vllm LLM in the verl image
- Run Ray inside container
- 4-GPU vllm TP
- verl 1-step training

### Attempts 9-12 — running it
- **Attempt 9 (17:02)**: vllm .generate() — `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
  - Fix: pin `transformers<5.0.0` (vllm 0.10.2 uses the pre-5.0 API).
- **Attempt 10 (17:07)**: 4-GPU vllm TP=4 Qwen3-8B — ✅ works.
- **Attempt 11 (17:14)**: verl PPO dry run — `AssertionError` on sampler shuffle.
  - Fix: `data.shuffle=false` (this fork disabled sampler-level shuffle).
- **Attempt 12 (17:18)**: next iter — `wandb.errors.Error: You must call wandb.init() before wandb.save()` at ray_trainer.py:1116 (fork-local patch).
  - Fix: add `wandb` to logger list + set `WANDB_MODE=offline`.

### ✅ Attempt 13 (17:25) — FULL VERL 1-STEP DRY RUN SUCCEEDS
- Qwen3-8B, 4x Blackwell, FSDP + vLLM TP=4
- Memory: 80 GB/GPU allocated, 98 GB reserved
- Step timing: 100s total = 71s gen + 9.7s ref + 8.5s update + 27s save
- Throughput: 5.6 tok/s (small batch, pre-optimization for Blackwell)
- `val-core/openai/gsm8k/reward/mean@1: 0.0` — validation ran
- **verl exit code: 0** — clean training loop + checkpoint save

## Winning image recipe

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install("torch==2.8.0", "torchvision", "torchaudio",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.10.2")
    .run_commands(
        "cd /tmp && wget -nv <flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310>.whl && pip install <whl>"
    )
    .run_commands("pip install --no-deps tensordict==0.9.1 torchdata codetiming pylatexenc "
                  "dill liger-kernel math_verify pyext pyjson5 dacite tyro pybind11 "
                  "latex2sympy2_extended pyvers")
    .pip_install("pandas", "pyarrow", "datasets",
                 "transformers>=4.51.0,<5.0.0",
                 "hydra-core", "omegaconf", "tensorboard",
                 "accelerate", "peft", "wandb", "hf-transfer")
    .run_commands("git clone https://github.com/GnarlyMshtep/pass_at_k.git /root/pass_at_k",
                  "cd /root/pass_at_k && pip install --no-deps -e .")
)
```

## Summary for future claude/user

Running verl on RTX PRO 6000 Blackwell on Modal **works**. No verl source patches required. Back-bending is in image construction (5 layer tweaks) and 2 runtime env tweaks. Total time to discover + validate: ~3 hours. Total Modal spend: ~$2. Step throughput on 4 Blackwell GPUs with Qwen3-8B is functional but underoptimized (~5-6 tok/s at batch 4×2) — this is Blackwell kernel maturity in vllm 0.10.2 and will improve as Blackwell support lands in newer versions.

## Real training comparison (2026-04-24)

### Experiment design
Same config (default_deepmath.json5, Qwen3-8B, batch 32, n=8, max_resp 6144 tokens, 150 steps).
4 runs launched:

| Run | GPUs | CPU | Where | App/Job |
|-----|------|-----|-------|---------|
| Modal 4-GPU | 4× RTX PRO 6000 Blackwell (97GB) | 64 vCPU (32 phys) | Modal | ap-mIQy0PUAZ3x3sQnlg267mc |
| Modal 2-GPU | 2× RTX PRO 6000 Blackwell (97GB) | 64 vCPU | Modal | ap-TLQ735xjKno6FA3XYShe9K |
| Local 4-GPU | 4× H200 (143GB) | 80 cores | SLURM job 6736 | run_id 9duo8r96 |
| Local 2-GPU | 2× H200 (143GB) | 40 cores | SLURM job 6744 | run_id yerqwqs0 |

wandb project: `blackwell_deepmath` (both `danielpolatajko-mars` and `matan-shtepel-carnegie-mellon-university` entities).

### Per-step timing (steady state)

| | Modal 4-GPU 20 vCPU | Modal 4-GPU 64 vCPU | Local 4-GPU H200 |
|--|--|--|--|
| Step 1 | 432s | 481s | 183s |
| Steady-state avg | ~417s | ~452s | ~184s |
| Ratio vs H200 | 2.27× slower | 2.46× slower | baseline |

**CPU-scaling experiment result**: Adding 3× CPUs (20→64 vCPU) made it ~8% SLOWER (452 vs 417s). CPU is not the bottleneck. The 2.3× gap is pure GPU kernel throughput — Blackwell sm_120 with vllm 0.10.2.

### Step timing breakdown (step 8, matched step)

| Phase | Modal Blackwell | Local H200 | Ratio |
|-------|----------------|------------|-------|
| timing_s/gen (vllm rollout) | 222s | 142s | 1.56× |
| timing_s/update_actor (FSDP) | 132s | 102s | 1.29× |
| timing_s/ref (ref log_prob) | 29s | 26s | 1.12× |
| timing_s/reward | 11s | 13s | 0.85× (CPU, same) |
| timing_s/step (total) | 423s | 310s | 1.36× |

Gen (vllm inference) is the biggest gap; FSDP update is less severe.

### GPU utilization
Blackwell GPUs hit **91-100% util** during generate phase. Power draw 403-478W. VRAM usage 76-82GB / 95.6GB. Not under-utilized — raw kernel speed is the bottleneck.

**verl's MFU reports 0.0 on Blackwell** — the MFU calculator doesn't have sm_120's peak TFLOPS programmed.

### Memory constraints
- 4-GPU Blackwell: `gpu_memory_utilization=0.6` works (max_memory_allocated=132GB across 4 GPUs)
- 2-GPU Blackwell: **impossible with Qwen3-8B + 6144 resp length on 95GB cards**. OOM'd 3 times:
  - gpu_mem=0.6: OOM during backward (88.6GB used, needed 9.4GB more)
  - gpu_mem=0.4 + 34K tokens + micro_batch=8: same OOM
  - gpu_mem=0.4 + 24K tokens + micro_batch=4: still OOM (92.5GB used, needed 6.65GB more)
  - Would need optimizer_offload=true or shorter response length to fit
- 4-GPU H200: 152GB allocated, 143GB per card so headroom is fine.

### Modal infra findings
- **Max CPU per container**: 64 vCPUs (not 128 — I hit the error). Modal says 0.125–64 cores.
- **Max timeout**: 86400s = 24h. 2-GPU runs that might take 30+ hours need checkpoint-resume.
- **Container CPU topology**: 20 vCPUs = 10 physical × 2 SMT (UI shows 10 cores, container sees 20). With cpu=64: 64 vCPU visible.
- **psutil.cpu_percent** returns 0 in Modal containers (KVM /proc/stat visibility). Use `mpstat` or parse /proc/stat raw.
- **Volume commits**: Periodic `run_volume.commit()` every 60s flushes log + util data to persistent storage, downloadable live via `fetch_log` / `fetch_util` helper functions.
- **Billing lag**: `modal billing report` lags real spend by several minutes. Spend monitor polls every 60s with kill-apps capability.

### Training metrics observation
- Both Blackwell and H200 runs show **oscillating reward/mean@8 (~0.28-0.32) with no upward trend** through 78 steps (H200).
- `advantages/frac_groups_nnz_w_adv ≈ 0.83` — learning signal is healthy, 83% of prompt groups have nonzero advantage.
- `actor/pg_clipfrac = 0.0` across all steps — policy ratios never deviate from 1.0.
- `actor/ppo_kl = 0.0` across all steps — policy identical to reference.
- `actor/grad_norm ≈ 0.004` — extremely tiny gradients.
- **This is a config issue, not a hardware issue** — both runs behave identically. Likely LR=1e-6 is too conservative for GRPO on DeepMath, or ppo_mini_batch_size=256 (= total samples) means only 1 effective update per epoch.

### Run status (final, 2026-04-24 ~18:25)
- **Modal 4-GPU**: cancelled at step 22/150. ~$73 spent. Enough data for timing comparison.
- **Modal 2-GPU**: OOM'd 3 times — 95GB Blackwell cards can't fit 2-GPU FSDP with Qwen3-8B.
- **Local 4-GPU H200**: cancelled at step 78/150. Enough data for timing comparison.
- **Local 2-GPU H200**: never started (resources blocked by 4-GPU job).
- **Modal 8-GPU**: queued, app `ap-SWW49oSAJxQl3oaTk7h4xS`. Will start now that 4-GPU freed GPUs.

### Spend tracking
- Image iteration attempts (steps 1-5): ~$3.45
- First 4-GPU 150-step run (killed at step 10): ~$18
- Second 4-GPU run (killed at step 22): ~$73 total cumulative
- 2-GPU attempts (3× OOM): ~$5
- 8-GPU run: queued, not yet spending
- **$500 hard cap** via active monitor (kills all ephemeral apps)

### Files
- `claude_scripts/modal_verl.py` — all Modal functions (step1-6, 2gpu, 8gpu variants, helpers)
- `claude_scripts/modal_spend_monitor.py` — billing poll, stop signal, active kill, app-watch
- `claude_state/verl_modal_plan.md` — original incremental plan
- `vfh/configs/runs/04/24/deepmath_blackwell_comparison_qwen3_8b.json5` — local VFH overrides
- `claude_plots/modal_blackwell_gpu_util.png` — GPU util/mem/power plot from 4-GPU run

## Known gotchas

- Flash-attn wheels are ABI-specific — match `torch2.X.Y`+`cu12Z` in filename
- flashinfer has Blackwell support from 0.2.5+; CUDA >=12.9 recommended (cu128 works but warns)
- vllm 0.10.2 is the sweet spot: supports Blackwell (torch 2.8+cu128) AND preserves verl's `from vllm.config import CompilationLevel` API. vllm 0.11+ broke this.
- vllm 0.9.x pins torch+cu126 (no Blackwell) — can't use even with pre-installed cu128 torch
- `transformers>=5.0` removed `all_special_tokens_extended` — pin `<5.0.0` for vllm 0.10.2
- 2-GPU Blackwell: **not feasible** for Qwen3-8B + 6144 resp length on 95GB cards. Even gpu_mem=0.4 + 24K tokens + micro_batch=4 OOMs during backward. Need optimizer_offload=true or shorter sequences.
- `data.shuffle=false` required by this verl fork's sampler assertion (not Blackwell-specific)
- `ray_trainer.py:1116` calls `wandb.save()` unconditionally — need wandb in logger list + valid key
- `git clone` must specify `-b verl-upgrades` — `custom/` dir not on main branch
- `psutil.cpu_percent()` returns 0 in Modal KVM containers
- verl's MFU reports 0.0 on Blackwell (sm_120 not in peak TFLOPS table)
- Modal CPU max = 64, timeout max = 86400s (24h)

## Reference URLs

- torch wheels: https://download.pytorch.org/whl/cu128
- flash-attn releases: https://github.com/Dao-AILab/flash-attention/releases
- flashinfer: https://github.com/flashinfer-ai/flashinfer/releases
- vllm: https://github.com/vllm-project/vllm/releases
- Modal GPU docs: https://modal.com/docs/guide/gpu
