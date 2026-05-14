"""Modal script for running verl incrementally on Modal.

Steps (run one at a time):
    modal run claude_scripts/modal_verl.py::step1_cpu_import
    modal run claude_scripts/modal_verl.py::step2_gpu_matmul
    modal run claude_scripts/modal_verl.py::step3_ray_single_gpu
    modal run claude_scripts/modal_verl.py::step4_multi_gpu_vllm
    modal run claude_scripts/modal_verl.py::step5_verl_dry_run
    modal run claude_scripts/modal_verl.py::step6_verl_full

See claude_state/verl_modal_plan.md for the plan.
"""
import modal

APP_NAME = "verl-incremental"
GPU_TYPE = "RTX-PRO-6000"  # May need adjustment; see Modal GPU docs
REPO_URL = "https://github.com/GnarlyMshtep/pass_at_k.git"
REPO_DIR = "/root/pass_at_k"

# ---------------------------------------------------------------------------
# Base image: torch + common deps, no flash-attn (those need GPU build context)
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "numpy<2.0.0",
    )
)

# ---------------------------------------------------------------------------
# GPU image: inherits base, adds flash-attn, vllm, verl repo
# Built lazily — only pulled in when a function with gpu=... is called.
# ---------------------------------------------------------------------------
verl_image = (
    base_image
    .pip_install(
        "tensordict==0.6.2",
        "torchdata<=0.11.0",
        "vllm==0.8.5.post1",
        "transformers[hf_xet]>=4.51.0,<5.0.0",
        "accelerate<=1.12.0",
        "datasets<=4.5.0",
        "peft<=0.18.1",
        "hf-transfer<=0.1.9",
        "pyarrow>=15.0.0,<=22.0.0",
        "pandas<=2.3.3",
        "ray[default]<=2.53.0",
        "codetiming<=1.4.0",
        "hydra-core<=1.3.2",
        "pylatexenc<=2.10",
        "wandb<=0.23.1",
        "dill<=0.4.0",
        "pybind11<=3.0.1",
        "liger-kernel<=0.6.4",
        "math_verify<=0.9.0",
        "pyext<=0.7",
        "pyjson5",
        "dacite",
        "tyro",
    )
    .run_commands(
        # flash-attn prebuilt wheel — pip needs the original filename (with version parts)
        "cd /tmp && wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && "
        "pip install --no-cache-dir /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        "cd /tmp && wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/"
        "flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && "
        "pip install --no-cache-dir /tmp/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl",
    )
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"cd {REPO_DIR} && pip install --no-deps -e .",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Blackwell attempts (RTX PRO 6000 = sm_120)
# torch 2.6/cu124 has no Blackwell kernels. Need torch 2.7+/cu128.
# Build incrementally: just torch first, then add flash-attn, then vllm, then verl.
# ---------------------------------------------------------------------------
# Blackwell base: minimal torch 2.7.1+cu128 (for simple matmul tests only)
blackwell_torch_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

# Blackwell + flash-attn + flashinfer (for Attempt 4 validation)
blackwell_attn_image = (
    blackwell_torch_image
    .pip_install("numpy<2.0.0", "einops", "packaging", "ninja")
    .run_commands(
        "cd /tmp && wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && "
        "pip install --no-cache-dir /tmp/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    )
    .pip_install("flashinfer-python==0.6.8")
)

# Blackwell verl stack v3:
# Key insight: vllm 0.9 pulls torch+cu126 (no Blackwell). vllm 0.11+ broke verl's API.
# vllm 0.10.2 = last 0.10 release, likely has both Blackwell (torch 2.8+cu128) AND
# preserved `from vllm.config import CompilationLevel`. Also need torch cu128 installed FIRST
# so vllm doesn't pull the cu126 default.
blackwell_verl_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential")
    # Install cu128 torch first — vllm's `torch==2.8.x` pin matches any local version suffix
    .pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("vllm==0.10.2")
    .run_commands(
        "cd /tmp && wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && "
        "pip install --no-cache-dir /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    )
    .run_commands(
        "pip install --no-deps tensordict==0.9.1 torchdata codetiming pylatexenc "
        "dill liger-kernel math_verify pyext pyjson5 dacite tyro pybind11 latex2sympy2_extended "
        "pyvers"
    )
    # hydra-core + omegaconf bring antlr4 at the correct version
    # transformers<5 — vllm 0.10.2 uses all_special_tokens_extended (removed in transformers 5.0)
    .pip_install(
        "pandas",
        "pyarrow",
        "datasets",
        "psutil",
        "transformers>=4.51.0,<5.0.0",
        "hydra-core",
        "omegaconf",
        "tensorboard",
        "accelerate",
        "peft",
        "wandb",
        "hf-transfer",
    )
    .run_commands(
        f"git clone -b verl-upgrades {REPO_URL} {REPO_DIR} || (sleep 5 && git clone -b verl-upgrades {REPO_URL} {REPO_DIR})",
        f"cd {REPO_DIR} && pip install --no-deps -e .",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Alias for step2d (vllm-only test on Blackwell) — reuses the same stack
blackwell_vllm_image = blackwell_verl_image

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Step 1: CPU only — base image + torch sanity check
# ---------------------------------------------------------------------------
@app.function(image=base_image, timeout=60 * 10)
def step1_cpu_import():
    import torch
    import platform

    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available (expected False): {torch.cuda.is_available()}")
    x = torch.arange(12).reshape(3, 4).float()
    y = x @ x.T
    print(f"CPU matmul result shape: {y.shape}, sum: {y.sum().item()}")
    return "step1 OK"


# ---------------------------------------------------------------------------
# Step 2: Single GPU — torch.cuda + matmul on GPU
# ---------------------------------------------------------------------------
@app.function(image=base_image, gpu=GPU_TYPE, timeout=60 * 10)
def step2_gpu_matmul():
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"GPU matmul shape: {y.shape}, mean: {y.mean().item():.4f}")
    # Quick throughput check
    import time
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    dt = time.time() - t0
    flops = 10 * 2 * (4096 ** 3) / dt / 1e12
    print(f"Matmul throughput: {flops:.1f} TFLOPS")
    return "step2 OK"


# ---------------------------------------------------------------------------
# Step 3: Ray inside Modal container + import verl
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 2b: Blackwell attempt — torch 2.7.1+cu128 matmul on RTX PRO 6000
# ---------------------------------------------------------------------------
@app.function(image=blackwell_torch_image, gpu=GPU_TYPE, timeout=60 * 10)
def step2b_blackwell_torch():
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA arch list: {torch.cuda.get_arch_list()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    cc = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm_{cc[0]}{cc[1]}")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"Matmul OK, mean: {y.mean().item():.4f}")
    import time
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    dt = time.time() - t0
    flops = 10 * 2 * (4096 ** 3) / dt / 1e12
    print(f"BF16 matmul: {flops:.1f} TFLOPS")
    return "step2b OK"


# ---------------------------------------------------------------------------
# Step 2c: Blackwell + flash-attn + flashinfer
# ---------------------------------------------------------------------------
@app.function(image=blackwell_attn_image, gpu=GPU_TYPE, timeout=60 * 10)
def step2c_blackwell_flashattn():
    import torch
    print(f"Torch: {torch.__version__}, device: {torch.cuda.get_device_name(0)}")

    import flash_attn
    print(f"flash_attn: {flash_attn.__version__}")
    from flash_attn import flash_attn_func
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    print(f"flash_attn output shape: {out.shape}, mean: {out.mean().item():.4f}")

    import flashinfer
    print(f"flashinfer: {flashinfer.__version__}")

    return "step2c OK"


# ---------------------------------------------------------------------------
# Step 2d: Blackwell + vLLM inference on a tiny model
# ---------------------------------------------------------------------------
@app.function(image=blackwell_vllm_image, gpu=GPU_TYPE, timeout=60 * 20)
def step2d_blackwell_vllm():
    import torch
    print(f"Torch: {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
    import vllm
    print(f"vLLM: {vllm.__version__}")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny, fast download
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=False,
    )
    prompts = ["Write one short sentence about cats:"]
    out = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.7))
    for o in out:
        print("OUTPUT:", o.outputs[0].text)
    return "step2d OK"


# ---------------------------------------------------------------------------
# Step 2e: Blackwell + import verl (the big compatibility test)
# ---------------------------------------------------------------------------
@app.function(image=blackwell_verl_image, gpu=GPU_TYPE, timeout=60 * 15)
def step2e_blackwell_verl_import():
    import torch
    import vllm
    import sys
    sys.path.insert(0, REPO_DIR)
    print(f"Torch: {torch.__version__}, vLLM: {vllm.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Try importing verl progressively
    import_tests = [
        "verl",
        "verl.trainer",
        "verl.trainer.ppo",
        "verl.workers",
        "verl.workers.fsdp_workers",
        "verl.workers.rollout",
        "verl.workers.rollout.vllm_rollout",
        "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    ]
    results = {}
    for modname in import_tests:
        try:
            __import__(modname)
            print(f"  ✅ {modname}")
            results[modname] = "OK"
        except Exception as e:
            print(f"  ❌ {modname}: {type(e).__name__}: {e}")
            results[modname] = f"{type(e).__name__}: {e}"
            break
    return results


@app.function(image=blackwell_verl_image, gpu=GPU_TYPE, timeout=60 * 15)
def step3_ray_single_gpu():
    import torch
    import ray

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    ray.init(num_gpus=1, include_dashboard=False, ignore_reinit_error=True)
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    assert "GPU" in ray.cluster_resources(), "Ray did not register GPUs!"

    @ray.remote(num_gpus=1)
    def _gpu_task():
        import torch
        return {
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0),
            "matmul_mean": (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")).mean().item(),
        }

    result = ray.get(_gpu_task.remote())
    print(f"Ray GPU task result: {result}")

    import verl
    print(f"verl imported OK. Path: {verl.__file__}")

    # Quick vllm inference through verl's target stack to catch runtime (not just import) issues
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1,
              gpu_memory_utilization=0.4, max_model_len=1024, dtype="bfloat16")
    out = llm.generate(["Say hi:"], SamplingParams(max_tokens=16, temperature=0.0))
    print("vLLM output:", out[0].outputs[0].text.strip())

    ray.shutdown()
    return "step3 OK"


# ---------------------------------------------------------------------------
# Step 4: 4 GPUs — Ray multi-GPU + vLLM inference on Qwen3-8B
# ---------------------------------------------------------------------------
@app.function(image=blackwell_verl_image, gpu=f"{GPU_TYPE}:4", timeout=60 * 30)
def step4_multi_gpu_vllm():
    import torch
    import ray

    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    ray.init(num_gpus=4, include_dashboard=False, ignore_reinit_error=True)
    print(f"Ray resources: {ray.cluster_resources()}")
    assert ray.cluster_resources().get("GPU", 0) >= 4, "Ray didn't see all 4 GPUs"

    from vllm import LLM, SamplingParams
    llm = LLM(
        model="Qwen/Qwen3-8B",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        dtype="bfloat16",
    )
    prompts = ["Write a haiku about reinforcement learning:"]
    out = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.7))
    for o in out:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)
    ray.shutdown()
    return "step4 OK"


# ---------------------------------------------------------------------------
# Step 5/6: verl training — placeholders, fill in once 1-4 pass
# ---------------------------------------------------------------------------
@app.function(image=blackwell_verl_image, gpu=f"{GPU_TYPE}:4", timeout=60 * 60)
def step5_verl_dry_run():
    """Minimal verl PPO dry run: tiny synthetic gsm8k-style dataset, 1 step, Qwen3-8B, TP=4 vllm."""
    import os, sys, subprocess
    import pandas as pd

    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)
    # wandb offline — this fork's ray_trainer.py:1116 calls wandb.save() unconditionally
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/root/wandb_offline"
    os.makedirs("/root/wandb_offline", exist_ok=True)

    # Build a 4-row synthetic parquet — gsm8k-style prompt + ground_truth
    rows = []
    for i in range(4):
        rows.append({
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": f"What is {i} + {i}? Answer with a number inside \\boxed{{}}."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(i * 2)},
            "extra_info": {"split": "train", "index": i, "answer": str(i * 2), "question": f"{i}+{i}"},
        })
    os.makedirs("/root/dummy_data", exist_ok=True)
    pd.DataFrame(rows).to_parquet("/root/dummy_data/train.parquet")
    pd.DataFrame(rows).to_parquet("/root/dummy_data/val.parquet")
    print("Wrote dummy parquet.")

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        "data.train_files=/root/dummy_data/train.parquet",
        "data.val_files=/root/dummy_data/val.parquet",
        "data.train_batch_size=4",
        "data.max_prompt_length=256",
        "data.max_response_length=256",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",  # this verl fork disabled sampler-level shuffle
        "actor_rollout_ref.model.path=Qwen/Qwen3-8B",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.n=2",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=false",
        "algorithm.use_kl_in_reward=false",
        "trainer.val_before_train=false",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=blackwell_dryrun",
        "trainer.experiment_name=step5_dryrun",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        "trainer.save_freq=99999",
        "trainer.test_freq=99999",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "+trainer.rollout_dump_freq=99999",
    ]
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_DIR)
    print(f"verl exit code: {result.returncode}")
    return f"step5 exit={result.returncode}"


# ---------------------------------------------------------------------------
# Step 6: Real 150-step training run on DeepMath-103K with Qwen3-8B
# ---------------------------------------------------------------------------
# Persistent volume for HF cache (Qwen3-8B ~16GB), data, and checkpoints.
run_volume = modal.Volume.from_name("verl-blackwell-runs", create_if_missing=True)

@app.function(
    image=blackwell_verl_image,
    gpu=f"{GPU_TYPE}:4",
    cpu=64.0,  # Modal's max
    timeout=60 * 60 * 18,
    volumes={"/vol": run_volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def step6_verl_deepmath_150steps():
    import os, sys, subprocess, shutil, time

    REPO = REPO_DIR
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    # Route HF + wandb through the volume so model weights persist across invocations
    os.environ["HF_HOME"] = "/vol/hf_home"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DIR"] = "/vol/wandb"
    os.environ["WANDB_MODE"] = "online"
    os.makedirs("/vol/hf_home", exist_ok=True)
    os.makedirs("/vol/wandb", exist_ok=True)
    os.makedirs("/vol/data/deepmath", exist_ok=True)
    os.makedirs("/vol/checkpoints", exist_ok=True)

    # --- Preprocess deepmath once and cache on the volume ---
    train_parquet = "/vol/data/deepmath/train.parquet"
    val_parquet = "/vol/data/deepmath/test.parquet"
    if not (os.path.exists(train_parquet) and os.path.exists(val_parquet)):
        print("Preprocessing DeepMath-103K (first time on this volume)...")
        subprocess.run(
            [
                "python",
                "custom/data_preprocessing/deepmath/deepmath.py",
                "--local_dir", "/vol/data/deepmath",
                "--ntrain", "10000",  # plenty for 150 steps at batch 32
                "--nval", "50",
            ],
            cwd=REPO,
            check=True,
        )
        run_volume.commit()
        print("Preprocessing done and committed to volume.")
    else:
        print("Reusing cached DeepMath parquet on volume.")

    run_name = f"blackwell_deepmath_q3_8b_150steps_{int(time.time())}"
    run_dir = f"/vol/checkpoints/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # --- Background util sampler: nvidia-smi + CPU/mem every 10s to volume ---
    import threading, json, shutil as _sh
    util_path = f"{run_dir}/util.jsonl"
    util_stop = threading.Event()

    def _util_sampler():
        import psutil as _ps
        while not util_stop.is_set():
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                gpus = []
                for line in smi.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 7:
                        gpus.append({
                            "idx": int(parts[0]),
                            "gpu_util_pct": float(parts[1]),
                            "mem_util_pct": float(parts[2]),
                            "mem_used_mb": float(parts[3]),
                            "mem_total_mb": float(parts[4]),
                            "power_w": float(parts[5]),
                            "temp_c": float(parts[6]),
                        })
                row = {
                    "ts": time.time(),
                    "cpu_pct": _ps.cpu_percent(interval=0.5),
                    "cpu_per_core": _ps.cpu_percent(interval=None, percpu=True),  # uses prior-sample delta
                    "load_avg": os.getloadavg(),
                    "mem_used_gb": _ps.virtual_memory().used / 1e9,
                    "mem_total_gb": _ps.virtual_memory().total / 1e9,
                    "gpus": gpus,
                }
                with open(util_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            except Exception as e:
                with open(util_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "err": str(e)}) + "\n")
            util_stop.wait(10)

    util_thread = threading.Thread(target=_util_sampler, daemon=True)
    util_thread.start()

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.train_batch_size=32",
        "data.max_prompt_length=1024",
        "data.max_response_length=6144",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",  # this fork disables sampler-level shuffle
        "actor_rollout_ref.model.path=Qwen/Qwen3-8B",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34000",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=34000",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=34000",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.n=8",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=false",
        "algorithm.use_kl_in_reward=false",
        "custom_reward_function.name=compute_score_math_boxed",
        "custom_reward_function.path=custom/reward/reward_utils.py",
        "trainer.val_before_train=false",  # skip initial val for speed
        "trainer.logger=[console,wandb]",
        f"trainer.project_name=blackwell_deepmath",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        "trainer.save_freq=40",
        "trainer.test_freq=20",
        "trainer.total_epochs=25",
        "trainer.total_training_steps=150",
        f"trainer.default_local_dir={run_dir}/checkpoints",
        "+rollout_dump_freq=1",
        f"+trainer.rollout_data_dir={run_dir}/rollouts/train",
        f"+trainer.validation_data_dir={run_dir}/rollouts/val",
    ]
    os.makedirs(f"{run_dir}/rollouts/train", exist_ok=True)
    os.makedirs(f"{run_dir}/rollouts/val", exist_ok=True)
    print("=" * 80)
    print("Launching verl. run_dir =", run_dir)
    print("=" * 80)

    log_path = f"{run_dir}/verl_output.log"
    # Stream subprocess output to BOTH stdout (visible via `modal app logs`) and a log file
    # on the volume. Periodically commit the volume so the log is downloadable live.
    last_commit = time.time()
    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
            if time.time() - last_commit > 60:
                try:
                    run_volume.commit()
                except Exception as e:
                    print(f"[warn] volume commit failed: {e}", flush=True)
                last_commit = time.time()
        proc.wait()
    util_stop.set()
    run_volume.commit()
    print(f"verl exit code: {proc.returncode}")
    print(f"log at {log_path} (use pull_run to sync to local)")
    return {"exit_code": proc.returncode, "run_dir": run_dir, "run_name": run_name}


# ---------------------------------------------------------------------------
# 2-GPU variant for cost comparison (launch twice for replication)
# ---------------------------------------------------------------------------
@app.function(
    image=blackwell_verl_image,
    gpu=f"{GPU_TYPE}:2",
    cpu=64.0,
    timeout=60 * 60 * 24,  # Modal max
    volumes={"/vol": run_volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def step6_verl_deepmath_150steps_2gpu(replicate: int = 1):
    """Same as step6_verl_deepmath_150steps but with 2 GPUs + TP=1 rollout. Pass replicate=1 or 2."""
    import os, sys, subprocess, time, threading, json
    REPO = REPO_DIR
    os.chdir(REPO); sys.path.insert(0, REPO)
    os.environ["HF_HOME"] = "/vol/hf_home"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DIR"] = "/vol/wandb"
    os.environ["WANDB_MODE"] = "online"
    for d in ["/vol/hf_home", "/vol/wandb", "/vol/data/deepmath", "/vol/checkpoints"]:
        os.makedirs(d, exist_ok=True)
    train_parquet = "/vol/data/deepmath/train.parquet"
    val_parquet = "/vol/data/deepmath/test.parquet"
    assert os.path.exists(train_parquet), "Run the 4-GPU variant first to preprocess deepmath"

    run_name = f"blackwell_deepmath_q3_8b_150steps_2gpu_rep{replicate}_{int(time.time())}"
    run_dir = f"/vol/checkpoints/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/rollouts/train", exist_ok=True)
    os.makedirs(f"{run_dir}/rollouts/val", exist_ok=True)

    # util sampler
    util_path = f"{run_dir}/util.jsonl"
    util_stop = threading.Event()
    def _sampler():
        import psutil as _ps
        while not util_stop.is_set():
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                gpus = []
                for line in smi.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 7:
                        gpus.append({"idx": int(parts[0]), "gpu_util_pct": float(parts[1]), "mem_util_pct": float(parts[2]),
                                     "mem_used_mb": float(parts[3]), "mem_total_mb": float(parts[4]),
                                     "power_w": float(parts[5]), "temp_c": float(parts[6])})
                with open(util_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "cpu_pct": _ps.cpu_percent(),
                                        "load_avg": os.getloadavg(),
                                        "mem_used_gb": _ps.virtual_memory().used / 1e9,
                                        "gpus": gpus}) + "\n")
            except Exception as e:
                with open(util_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "err": str(e)}) + "\n")
            util_stop.wait(10)
    threading.Thread(target=_sampler, daemon=True).start()

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.train_batch_size=32",
        "data.max_prompt_length=1024",
        "data.max_response_length=6144",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",
        "actor_rollout_ref.model.path=Qwen/Qwen3-8B",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=8",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=false",
        "algorithm.use_kl_in_reward=false",
        "custom_reward_function.name=compute_score_math_boxed",
        "custom_reward_function.path=custom/reward/reward_utils.py",
        "trainer.val_before_train=false",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=blackwell_deepmath",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.save_freq=40",
        "trainer.test_freq=20",
        "trainer.total_epochs=25",
        "trainer.total_training_steps=150",
        f"trainer.default_local_dir={run_dir}/checkpoints",
        "+rollout_dump_freq=1",
        f"+trainer.rollout_data_dir={run_dir}/rollouts/train",
        f"+trainer.validation_data_dir={run_dir}/rollouts/val",
    ]
    print(f"Launching 2-GPU replicate {replicate}. run_dir={run_dir}")
    log_path = f"{run_dir}/verl_output.log"
    last_commit = time.time()
    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True); logf.write(line)
            if time.time() - last_commit > 60:
                try: run_volume.commit()
                except Exception as e: print(f"[warn] commit: {e}")
                last_commit = time.time()
        proc.wait()
    util_stop.set()
    run_volume.commit()
    return {"exit_code": proc.returncode, "run_dir": run_dir, "run_name": run_name}


# ---------------------------------------------------------------------------
# 8-GPU variant for scaling comparison
# ---------------------------------------------------------------------------
@app.function(
    image=blackwell_verl_image,
    gpu=f"{GPU_TYPE}:8",
    cpu=64.0,
    timeout=60 * 60 * 24,
    volumes={"/vol": run_volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def step6_verl_deepmath_150steps_8gpu():
    """Same as step6_verl_deepmath_150steps but with 8 GPUs."""
    import os, sys, subprocess, time, threading, json
    REPO = REPO_DIR
    os.chdir(REPO); sys.path.insert(0, REPO)
    os.environ["HF_HOME"] = "/vol/hf_home"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DIR"] = "/vol/wandb"
    os.environ["WANDB_MODE"] = "online"
    for d in ["/vol/hf_home", "/vol/wandb", "/vol/data/deepmath", "/vol/checkpoints"]:
        os.makedirs(d, exist_ok=True)
    train_parquet = "/vol/data/deepmath/train.parquet"
    val_parquet = "/vol/data/deepmath/test.parquet"
    assert os.path.exists(train_parquet), "Run the 4-GPU variant first to preprocess deepmath"

    run_name = f"blackwell_deepmath_q3_8b_150steps_8gpu_{int(time.time())}"
    run_dir = f"/vol/checkpoints/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/rollouts/train", exist_ok=True)
    os.makedirs(f"{run_dir}/rollouts/val", exist_ok=True)

    util_path = f"{run_dir}/util.jsonl"
    util_stop = threading.Event()
    def _sampler():
        import psutil as _ps
        while not util_stop.is_set():
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                gpus = []
                for line in smi.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 7:
                        gpus.append({"idx": int(parts[0]), "gpu_util_pct": float(parts[1]), "mem_util_pct": float(parts[2]),
                                     "mem_used_mb": float(parts[3]), "mem_total_mb": float(parts[4]),
                                     "power_w": float(parts[5]), "temp_c": float(parts[6])})
                with open(util_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "cpu_pct": _ps.cpu_percent(),
                                        "load_avg": os.getloadavg(),
                                        "mem_used_gb": _ps.virtual_memory().used / 1e9,
                                        "gpus": gpus}) + "\n")
            except Exception as e:
                with open(util_path, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "err": str(e)}) + "\n")
            util_stop.wait(10)
    threading.Thread(target=_sampler, daemon=True).start()

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.train_batch_size=32",
        "data.max_prompt_length=1024",
        "data.max_response_length=6144",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=false",
        "actor_rollout_ref.model.path=Qwen/Qwen3-8B",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34000",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=34000",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=34000",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.n=8",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=false",
        "algorithm.use_kl_in_reward=false",
        "custom_reward_function.name=compute_score_math_boxed",
        "custom_reward_function.path=custom/reward/reward_utils.py",
        "trainer.val_before_train=false",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=blackwell_deepmath",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=40",
        "trainer.test_freq=20",
        "trainer.total_epochs=25",
        "trainer.total_training_steps=150",
        f"trainer.default_local_dir={run_dir}/checkpoints",
        "+rollout_dump_freq=1",
        f"+trainer.rollout_data_dir={run_dir}/rollouts/train",
        f"+trainer.validation_data_dir={run_dir}/rollouts/val",
    ]
    print(f"Launching 8-GPU run. run_dir={run_dir}")
    log_path = f"{run_dir}/verl_output.log"
    last_commit = time.time()
    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True); logf.write(line)
            if time.time() - last_commit > 60:
                try: run_volume.commit()
                except Exception as e: print(f"[warn] commit: {e}")
                last_commit = time.time()
        proc.wait()
    util_stop.set()
    run_volume.commit()
    return {"exit_code": proc.returncode, "run_dir": run_dir, "run_name": run_name}


# ---------------------------------------------------------------------------
# Helper: sync a run's logs + small metadata files from the volume to local.
# Usage: modal run claude_scripts/modal_verl.py::pull_run --run-name <name>
# Or call list_runs() to see what's available.
# ---------------------------------------------------------------------------
@app.function(image=modal.Image.debian_slim().pip_install("psutil"), gpu=f"{GPU_TYPE}:4", timeout=60 * 3)
def probe_blackwell_specs() -> dict:
    """Report Blackwell container specs: CPUs, RAM, GPUs, nvidia-smi dump."""
    import os, subprocess, psutil
    info = {
        "cpu_count": os.cpu_count(),
        "cpu_physical_psutil": psutil.cpu_count(logical=False),
        "cpu_logical_psutil": psutil.cpu_count(logical=True),
        "total_ram_gb": psutil.virtual_memory().total / 1e9,
        "available_ram_gb": psutil.virtual_memory().available / 1e9,
    }
    # Direct /proc/cpuinfo inspection — more reliable than psutil in containers
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        import re
        siblings = sorted({int(m.group(1)) for m in re.finditer(r"siblings\s*:\s*(\d+)", txt)})
        cores = sorted({int(m.group(1)) for m in re.finditer(r"cpu cores\s*:\s*(\d+)", txt)})
        physical_ids = sorted({m.group(1) for m in re.finditer(r"physical id\s*:\s*(\S+)", txt)})
        core_ids = sorted({m.group(1) for m in re.finditer(r"core id\s*:\s*(\S+)", txt)})
        unique_phys_core_pairs = len(
            {(a.group(1), b.group(1)) for a, b in
             zip(re.finditer(r"physical id\s*:\s*(\S+)", txt),
                 re.finditer(r"core id\s*:\s*(\S+)", txt))}
        )
        info["proc_cpuinfo_processors"] = txt.count("processor\t:")
        info["proc_cpuinfo_siblings"] = siblings
        info["proc_cpuinfo_cpu_cores"] = cores
        info["proc_cpuinfo_physical_ids"] = physical_ids
        info["proc_cpuinfo_unique_core_ids"] = len(core_ids)
        info["proc_cpuinfo_unique_phys_core_pairs"] = unique_phys_core_pairs
    except Exception as e:
        info["cpuinfo_err"] = str(e)
    # lscpu for a definitive summary
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10)
        info["lscpu"] = out.stdout
    except Exception as e:
        info["lscpu_err"] = str(e)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        info["nvidia_smi"] = out.stdout
    except Exception as e:
        info["nvidia_smi_err"] = str(e)
    for k, v in info.items():
        print(f"{k}: {v}")
    return info


@app.function(image=modal.Image.debian_slim(), volumes={"/vol": run_volume}, timeout=60 * 5)
def list_runs() -> list[str]:
    import os
    path = "/vol/checkpoints"
    if not os.path.exists(path):
        return []
    runs = sorted(os.listdir(path))
    for r in runs:
        files = []
        rd = os.path.join(path, r)
        log = os.path.join(rd, "verl_output.log")
        if os.path.exists(log):
            files.append(f"log={os.path.getsize(log)/1e6:.1f}MB")
        ck = os.path.join(rd, "checkpoints")
        if os.path.exists(ck):
            steps = sorted([d for d in os.listdir(ck) if d.startswith("global_step_")])
            files.append(f"ckpts={len(steps)}")
        print(f"{r}: {', '.join(files)}")
    return runs


@app.function(image=modal.Image.debian_slim(), volumes={"/vol": run_volume}, timeout=60 * 5)
def fetch_util(run_name: str, last_n: int = 20) -> list[dict]:
    """Return the last N util samples for a run."""
    import os, json
    util_path = f"/vol/checkpoints/{run_name}/util.jsonl"
    if not os.path.exists(util_path):
        return [{"err": f"no util file at {util_path}"}]
    with open(util_path) as f:
        lines = f.readlines()
    out = []
    for line in lines[-last_n:]:
        try:
            out.append(json.loads(line))
        except Exception as e:
            out.append({"parse_err": str(e)})
    # print summary for humans
    for row in out:
        if "err" in row:
            print(f"[{row.get('ts',0):.0f}] ERR: {row['err']}")
            continue
        gpus_summary = " | ".join(
            f"GPU{g['idx']}: {g['gpu_util_pct']:.0f}%util {g['mem_used_mb']/1024:.1f}/{g['mem_total_mb']/1024:.1f}GB {g['power_w']:.0f}W {g['temp_c']:.0f}°C"
            for g in row.get("gpus", [])
        )
        print(f"[{row.get('ts', 0):.0f}] cpu={row.get('cpu_pct', 0):.1f}% load={row.get('load_avg',[0])[0]:.1f} mem={row.get('mem_used_gb',0):.1f}GB | {gpus_summary}")
    return out


@app.function(image=modal.Image.debian_slim(), volumes={"/vol": run_volume}, timeout=60 * 5)
def fetch_log(run_name: str, tail_lines: int = 0) -> str:
    """Return the verl_output.log contents (or last N lines) for a run."""
    import os
    log_path = f"/vol/checkpoints/{run_name}/verl_output.log"
    if not os.path.exists(log_path):
        return f"[no log at {log_path}]"
    with open(log_path) as f:
        content = f.read()
    if tail_lines > 0:
        lines = content.splitlines()
        content = "\n".join(lines[-tail_lines:])
    return content


@app.local_entrypoint()
def main():
    """Default entrypoint — runs step 1 if none specified via ::."""
    print(step1_cpu_import.remote())
