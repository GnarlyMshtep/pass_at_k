"""Plot GPU utilization, memory, and power from Modal Blackwell 4-GPU run."""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

raw_path = Path("tmp/modal_4gpu_util_raw.txt")
lines = raw_path.read_text().strip().splitlines()

timestamps = []
gpu_utils = {i: [] for i in range(4)}
gpu_mem_used = {i: [] for i in range(4)}
gpu_power = {i: [] for i in range(4)}

for line in lines:
    ts_match = re.match(r"\[(\d+)\]", line)
    if not ts_match:
        continue
    ts = int(ts_match.group(1))
    timestamps.append(ts)

    for gpu_id in range(4):
        pat = rf"GPU{gpu_id}: (\d+)%util ([\d.]+)/([\d.]+)GB (\d+)W"
        m = re.search(pat, line)
        if m:
            gpu_utils[gpu_id].append(float(m.group(1)))
            gpu_mem_used[gpu_id].append(float(m.group(2)))
            gpu_power[gpu_id].append(float(m.group(4)))
        else:
            gpu_utils[gpu_id].append(np.nan)
            gpu_mem_used[gpu_id].append(np.nan)
            gpu_power[gpu_id].append(np.nan)

t0 = timestamps[0]
t_min = [(t - t0) / 60.0 for t in timestamps]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

for gpu_id in range(4):
    axes[0].plot(t_min, gpu_utils[gpu_id], color=colors[gpu_id], alpha=0.7, linewidth=0.8, label=f"GPU {gpu_id}")
axes[0].set_ylabel("GPU Util (%)")
axes[0].set_ylim(-5, 105)
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_title("Modal Blackwell 4× RTX PRO 6000 — GPU Utilization, Memory, Power")
axes[0].axhline(y=90, color="gray", linestyle="--", alpha=0.3)

for gpu_id in range(4):
    axes[1].plot(t_min, gpu_mem_used[gpu_id], color=colors[gpu_id], alpha=0.7, linewidth=0.8, label=f"GPU {gpu_id}")
axes[1].set_ylabel("VRAM Used (GB)")
axes[1].axhline(y=95.6, color="gray", linestyle="--", alpha=0.3, label="95.6GB total")
axes[1].legend(loc="lower right", fontsize=8)

for gpu_id in range(4):
    axes[2].plot(t_min, gpu_power[gpu_id], color=colors[gpu_id], alpha=0.7, linewidth=0.8, label=f"GPU {gpu_id}")
axes[2].set_ylabel("Power (W)")
axes[2].set_xlabel("Time (minutes)")
axes[2].legend(loc="lower right", fontsize=8)

plt.tight_layout()
out = Path("claude_plots/modal_blackwell_gpu_util.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
