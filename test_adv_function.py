#!/usr/bin/env python3
import pickle
import torch
import numpy as np
from collections import defaultdict

# Load the pickled inputs
with open('adv_inputs.pkl', 'rb') as f:
    adv_inputs = pickle.load(f)

print("Loaded inputs:")
for key, value in adv_inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, np.ndarray):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, list):
        print(f"  {key}: len={len(value)}, sample={value[:3] if len(value) > 0 else None}")
    else:
        print(f"  {key}: {value}")

print("\n" + "="*80 + "\n")

# Extract the inputs
token_level_rewards = adv_inputs['token_level_rewards']
response_mask = adv_inputs['response_mask']
uids = adv_inputs['uids']
epsilon = adv_inputs['epsilon']
norm_adv_by_std_in_grpo = adv_inputs['norm_adv_by_std_in_grpo']
monitor_scores = adv_inputs['monitor_scores']
did_sel_hint = adv_inputs['did_sel_hint']
is_correct = adv_inputs['is_correct']
monitor_index = adv_inputs['monitor_index']

# Run the computation
id2other_score = token_level_rewards.sum(dim=-1)
monitor_index2infos = defaultdict(list)

bsz = response_mask.size(0)
print(f"Batch size: {bsz}")
print(f"Monitor indices: {set(monitor_index)}")

# Step 1: Assemble infos
for i in range(bsz):
    monitor_index2infos[monitor_index[i]].append(
        {
            "is_correct": is_correct[i],
            "monitor_score": monitor_scores[i],
            "did_sel_hint": did_sel_hint[i],
            "uid": uids[i],
            "other_score": id2other_score[i],
            "id": i,
        }
    )

print(f"\nNumber of unique monitor indices: {len(monitor_index2infos)}")
for idx, infos in monitor_index2infos.items():
    print(f"  Monitor index {idx}: {len(infos)} samples")
    print(f"    Sample 0: is_correct={infos[0]['is_correct']}, did_sel_hint={infos[0]['did_sel_hint']}, monitor_score={infos[0]['monitor_score']:.3f}")

# Step 2: Compute unnormalized scores
num_varients = max(monitor_index2infos.keys()) + 1
print(f"\nNumber of variants: {num_varients}")

for i in range(num_varients // 2):
    base_idx = 2 * i
    hinted_idx = 2 * i + 1

    # FIXED: Use did_sel_hint instead of is_correct
    baseline_hint_sel_rate = sum([v["did_sel_hint"] for v in monitor_index2infos[base_idx]]) / len(
        monitor_index2infos[base_idx]
    )

    print(f"\nPair {i}: base_idx={base_idx}, hinted_idx={hinted_idx}")
    print(f"  Baseline hint selection rate: {baseline_hint_sel_rate:.3f}")

    # Baseline scores
    for j in range(len(monitor_index2infos[base_idx])):
        infos = monitor_index2infos[base_idx][j]
        correctness_plus_other = infos["other_score"] + infos["is_correct"]
        true_hint_sel_effect = 0
        calibration = (true_hint_sel_effect - infos["monitor_score"]) ** 2
        score_unwhitened = correctness_plus_other - calibration
        monitor_index2infos[base_idx][j]["score_unwhitened"] = score_unwhitened

    # Hinted scores
    for j in range(len(monitor_index2infos[hinted_idx])):
        infos = monitor_index2infos[hinted_idx][j]
        correctness_plus_other = infos["other_score"] + infos["is_correct"]
        hint_sel_effect = infos["did_sel_hint"] - baseline_hint_sel_rate
        calibration = (hint_sel_effect - infos["monitor_score"]) ** 2
        score_unwhitened = correctness_plus_other - calibration
        # FIXED: Use hinted_idx instead of base_idx
        monitor_index2infos[hinted_idx][j]["score_unwhitened"] = score_unwhitened

# Step 3: Whiten scores
id2score_normalized = torch.zeros_like(token_level_rewards.sum(dim=-1))

for mntr_idx in range(num_varients):
    # FIXED: Add list() around generator
    scores_unwhitened_tensor = torch.tensor([v["score_unwhitened"] for v in monitor_index2infos[mntr_idx]])
    mean = torch.mean(scores_unwhitened_tensor)
    std = torch.std(scores_unwhitened_tensor)

    print(f"\nMonitor index {mntr_idx}: mean={mean:.3f}, std={std:.3f}")

    for j in range(len(monitor_index2infos[mntr_idx])):
        infos = monitor_index2infos[mntr_idx][j]
        id = infos["id"]

        score_unwhitened = infos["score_unwhitened"]
        if norm_adv_by_std_in_grpo:
            id2score_normalized[id] = (score_unwhitened - mean) / (std + epsilon)
        else:
            id2score_normalized[id] = (score_unwhitened - mean)

scores = id2score_normalized.unsqueeze(-1) * response_mask

print("\n" + "="*80)
print("RESULTS:")
print(f"  Normalized scores shape: {id2score_normalized.shape}")
print(f"  Normalized scores stats: mean={id2score_normalized.mean():.3f}, std={id2score_normalized.std():.3f}")
print(f"  Final advantages shape: {scores.shape}")
print(f"  Final advantages stats: mean={scores.mean():.3f}, std={scores.std():.3f}")
print(f"\nFirst 5 normalized scores: {id2score_normalized[:5]}")