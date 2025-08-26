# RL training + rollout VRAM estimator
#
# This file defines a single function `estimate_peak_vram` that computes
# per-GPU memory estimates for:
#   - training peak VRAM (actor backward step)
#   - rollout (SGLang) static pool capacity and used KV+weights
#   - a combined co-located estimate (training + rollout on same GPUs)
#
# The math follows these parametric formulas:
#   Let:
#     P        = number of model parameters
#     b        = bytes per element (2 for bf16, 1 for int8 weights, 4 for fp32, etc.)
#     L        = number of transformer layers
#     H        = hidden size
#     h_kv     = number of KV heads (GQA)
#     d        = head dimension
#     T        = effective tokens per sample resident in memory
#     W        = training world size (number of GPUs used for training)
#     Bmini    = PPO mini-batch (global)
#     Bmicro   = PPO micro-batch (per GPU, training)
#     f        = rollout gpu_memory_utilization (fraction of FREE VRAM for static pool)
#     M_total  = total VRAM per GPU (GiB)
#     M_free_init = free VRAM (GiB) at the moment the rollout engine starts
#     grad_ckpt = boolean: gradient checkpointing enabled
#     alpha    = activation overhead factor (dimensionless). If not provided,
#                we use alpha = 1.5 with gradient checkpointing, else 6.0.
#     param_offload, optimizer_offload = booleans for FSDP offload to CPU
#     work_overhead_gib = generic CUDA/temp workspace + fragmentation (GiB)
#     rollout_overhead_gib = rollout engine transient overhead (GiB)
#     n_rollouts_total = total generated completions per prompt (for concurrency planning)
#
#   Training peak (per GPU):
#     B_eff = min(Bmicro, ceil(Bmini / W))
#     M_act_GiB  = alpha * L * B_eff * T * H * b / 2**30      (activations)
#     M_grads_GiB = (P / W) * b / 2**30                       (bf16 grads, sharded)
#     M_param_GiB = 0 if param_offload else (P / W) * b / 2**30
#     M_optim_GiB = 0 if optimizer_offload else (P / W) * (4 + 8) / 2**30
#                    # Adam: 4B fp32 master + 8B moments (m,v) per param
#     training_peak_GiB = M_act_GiB + M_grads_GiB + M_param_GiB + M_optim_GiB + work_overhead_gib
#
#   Rollout (per GPU):
#     M_weights_rollout_GiB = P * b / 2**30
#     M_KV_per_seq_GiB = (2 * L * h_kv * d * b * T) / 2**30     # K and V
#     M_KV_pool_GiB    = max(0.0, f * M_free_init - M_weights_rollout_GiB)
#     seqs_per_gpu     = ceil(n_rollouts_total / W)             # evenly spread worst-case
#     M_rollout_used_GiB = M_weights_rollout_GiB + seqs_per_gpu * M_KV_per_seq_GiB + rollout_overhead_gib
#     max_seqs_capacity = floor(M_KV_pool_GiB / max(M_KV_per_seq_GiB, 1e-12))
#
#   Combined (co-located on same GPUs):
#     combined_peak_GiB = training_peak_GiB + min(M_rollout_used_GiB, f * M_free_init)
#       (since the rollout engine preallocates up to f*M_free_init; usage is capped by that pool)
#
# Notes:
#   • This is a *model*; real frameworks introduce small additional buffers.
#   • alpha controls how aggressively we count activation tensors. For Flash-Attn +
#     block-level checkpointing, 1.5 is a decent middle-of-road default.
#   • If you run rollout on isolated GPUs, set M_free_init = M_total (or measured free).
#
# The function returns a dict with these values and intermediate components.
#
from math import ceil, floor
from typing import Optional, Dict

GiB = 2**30

def estimate_peak_vram(
    # Model / arch
    P: float,                 # params (e.g., 1.54e9)
    L: int,                   # layers
    H: int,                   # hidden size
    h_kv: int,                # KV heads
    d: int,                   # head dim
    b: int = 2,               # bytes per element (bf16 default)
    # Sequence / batching
    T: int = 2048,            # effective tokens per sample (prompt+response)
    W: int = 8,               # training world size (num GPUs used for training)
    Bmini: int = 32,          # PPO mini-batch (global)
    Bmicro: int = 8,          # PPO micro-batch per GPU
    # Training memory controls
    grad_ckpt: bool = True,
    alpha: Optional[float] = None,      # if None, pick 1.5 (ckpt) or 6.0 (no ckpt)
    param_offload: bool = False,
    optimizer_offload: bool = False,
    work_overhead_gib: float = 2.0,     # CUDA workspace/fragmentation allowance
    # Rollout engine (SGLang/vLLM)
    f: float = 0.6,                      # gpu_memory_utilization
    M_total_gib: float = 80.0,           # H100-80GB default
    M_free_init_gib: Optional[float] = None,  # free VRAM at rollout init
    n_rollouts_total: int = 28,          # total completions (concurrency proxy)
    rollout_overhead_gib: float = 1.0,   # transient overhead during rollout
) -> Dict[str, float]:
    # Default alpha based on checkpointing
    if alpha is None:
        alpha = 1.5 if grad_ckpt else 6.0

    # If M_free_init_gib not provided, assume co-located with ~10 GiB already used.
    if M_free_init_gib is None:
        M_free_init_gib = M_total_gib - 10.0

    # ---- Training peak VRAM ----
    B_eff = min(Bmicro, ceil(Bmini / max(W, 1)))
    M_act_GiB   = alpha * L * B_eff * T * H * b / GiB
    M_grads_GiB = (P / max(W, 1)) * b / GiB

    M_param_GiB = 0.0 if param_offload else (P / max(W, 1)) * b / GiB
    # Adam: fp32 master (4B) + moments (8B). Gradients are already counted above.
    M_optim_GiB = 0.0 if optimizer_offload else (P / max(W, 1)) * (4 + 8) / GiB

    training_peak_GiB = M_act_GiB + M_grads_GiB + M_param_GiB + M_optim_GiB + work_overhead_gib

    # ---- Rollout VRAM ----
    M_weights_rollout_GiB = P * b / GiB
    M_KV_per_seq_GiB = (2 * L * h_kv * d * b * T) / GiB
    M_KV_pool_GiB = max(0.0, f * M_free_init_gib - M_weights_rollout_GiB)

    seqs_per_gpu = ceil(n_rollouts_total / max(W, 1))
    M_rollout_used_GiB = M_weights_rollout_GiB + seqs_per_gpu * M_KV_per_seq_GiB + rollout_overhead_gib
    max_seqs_capacity = floor(M_KV_pool_GiB / max(M_KV_per_seq_GiB, 1e-12))

    # ---- Combined estimate (co-located) ----
    # Rollout usage cannot exceed its static pool.
    rollout_capped = min(M_rollout_used_GiB, f * M_free_init_gib)
    combined_peak_GiB = training_peak_GiB + rollout_capped

    return {
        # Training components
        "training_peak_GiB": training_peak_GiB,
        "training_activations_GiB": M_act_GiB,
        "training_grads_GiB": M_grads_GiB,
        "training_param_shards_GiB": M_param_GiB,
        "training_optimizer_shards_GiB": M_optim_GiB,
        "training_B_eff": float(B_eff),
        # Rollout components
        "rollout_weights_GiB": M_weights_rollout_GiB,
        "rollout_KV_per_seq_GiB": M_KV_per_seq_GiB,
        "rollout_KV_pool_GiB": M_KV_pool_GiB,
        "rollout_seqs_per_gpu": float(seqs_per_gpu),
        "rollout_used_GiB": M_rollout_used_GiB,
        "rollout_max_seqs_capacity": float(max_seqs_capacity),
        # Combined
        "combined_peak_GiB": combined_peak_GiB,
        # Reference
        "assumed_M_free_init_GiB": M_free_init_gib,
        "gpu_total_GiB": M_total_gib,
        "gpu_memory_utilization_f": f,
    }


# --- Run the estimator for *your* config ---
# DeepSeek-R1-Distill-Qwen-1.5B (Qwen 1.5B family)
P = 1.54e9
L = 28
H = 1536
h_kv = 2
d = 128
b = 2               # bf16
T = 2048 + 10240    # 12,288
W = 7
Bmini = 32
Bmicro = 8
grad_ckpt = True
param_offload = True
optimizer_offload = True
f = 0.6
M_total_gib = 80.0
n_rollouts_total = 4 * W   # 28

# Scenario A: co-located (assume ~10 GiB already in use at rollout init)
res_colocated = estimate_peak_vram(
    P,L,H,h_kv,d,b,
    T=T, W=W, Bmini=Bmini, Bmicro=Bmicro,
    grad_ckpt=grad_ckpt, alpha=None,
    param_offload=param_offload, optimizer_offload=optimizer_offload,
    work_overhead_gib=2.0,
    f=f, M_total_gib=M_total_gib, M_free_init_gib=None,
    n_rollouts_total=n_rollouts_total, rollout_overhead_gib=1.0
)

# Scenario B: rollout on isolated GPUs (free ~80 GiB at init)
res_isolated = estimate_peak_vram(
    P,L,H,h_kv,d,b,
    T=T, W=W, Bmini=Bmini, Bmicro=Bmicro,
    grad_ckpt=grad_ckpt, alpha=None,
    param_offload=param_offload, optimizer_offload=optimizer_offload,
    work_overhead_gib=2.0,
    f=f, M_total_gib=M_total_gib, M_free_init_gib=80.0,
    n_rollouts_total=n_rollouts_total, rollout_overhead_gib=1.0
)

import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

df = pd.DataFrame([
    {"Scenario": "Co-located (M_free_init≈70 GiB)", **res_colocated},
    {"Scenario": "Rollout isolated (M_free_init=80 GiB)", **res_isolated},
])

display_dataframe_to_user("Peak VRAM estimates (GiB)", df.round(3))

# Save as a reusable module
code_text = """\
# Save as rl_mem_estimator.py
from math import ceil, floor
from typing import Optional, Dict

GiB = 2**30

def estimate_peak_vram(
    P: float, L: int, H: int, h_kv: int, d: int, b: int = 2,
    T: int = 2048, W: int = 8, Bmini: int = 32, Bmicro: int = 8,
    grad_ckpt: bool = True, alpha: Optional[float] = None,
    param_offload: bool = False, optimizer_offload: bool = False,
    work_overhead_gib: float = 2.0,
    f: float = 0.6, M_total_gib: float = 80.0, M_free_init_gib: Optional[float] = None,
    n_rollouts_total: int = 28, rollout_overhead_gib: float = 1.0,
) -> Dict[str, float]:
    if alpha is None:
        alpha = 1.5 if grad_ckpt else 6.0
    if M_free_init_gib is None:
        M_free_init_gib = M_total_gib - 10.0

    B_eff = min(Bmicro, ceil(Bmini / max(W, 1)))
    M_act_GiB   = alpha * L * B_eff * T * H * b / GiB
    M_grads_GiB = (P / max(W, 1)) * b / GiB
    M_param_GiB = 0.0 if param_offload else (P / max(W, 1)) * b / GiB
    M_optim_GiB = 0.0 if optimizer_offload else (P / max(W, 1)) * (4 + 8) / GiB
    training_peak_GiB = M_act_GiB + M_grads_GiB + M_param_GiB + M_optim_GiB + work_overhead_gib

    M_weights_rollout_GiB = P * b / GiB
    M_KV_per_seq_GiB = (2 * L * h_kv * d * b * T) / GiB
    M_KV_pool_GiB = max(0.0, f * M_free_init_gib - M_weights_rollout_GiB)

    seqs_per_gpu = ceil(n_rollouts_total / max(W, 1))
    M_rollout_used_GiB = M_weights_rollout_GiB + seqs_per_gpu * M_KV_per_seq_GiB + rollout_overhead_gib
    max_seqs_capacity = floor(M_KV_pool_GiB / max(M_KV_per_seq_GiB, 1e-12))

    rollout_capped = min(M_rollout_used_GiB, f * M_free_init_gib)
    combined_peak_GiB = training_peak_GiB + rollout_capped

    return {
        "training_peak_GiB": training_peak_GiB,
        "training_activations_GiB": M_act_GiB,
        "training_grads_GiB": M_grads_GiB,
        "training_param_shards_GiB": M_param_GiB,
        "training_optimizer_shards_GiB": M_optim_GiB,
        "training_B_eff": float(B_eff),
        "rollout_weights_GiB": M_weights_rollout_GiB,
        "rollout_KV_per_seq_GiB": M_KV_per_seq_GiB,
        "rollout_KV_pool_GiB": M_KV_pool_GiB,
        "rollout_seqs_per_gpu": float(seqs_per_gpu),
        "rollout_used_GiB": M_rollout_used_GiB,
        "rollout_max_seqs_capacity": float(max_seqs_capacity),
        "combined_peak_GiB": combined_peak_GiB,
        "assumed_M_free_init_GiB": M_free_init_gib,
        "gpu_total_GiB": M_total_gib,
        "gpu_memory_utilization_f": f,
    }
"""
with open("rl_mem_estimator.py", "w") as f_out:
    f_out.write(code_text)

