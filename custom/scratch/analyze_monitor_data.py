#!/usr/bin/env python3
"""
Analyze the monitor index data to understand pairing issues
"""
import pickle
import numpy as np
from collections import defaultdict

# Load the data
with open('all_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("=" * 80)
print("ANALYZING MONITOR INDEX DATA")
print("=" * 80)

# Extract monitor_index2infos
monitor_index2infos = data.get('monitor_index2infos', {})

print(f"\nTotal monitor indices: {len(monitor_index2infos)}")
print(f"Monitor index keys (first 20): {sorted(monitor_index2infos.keys())[:20]}")

# Check for the "index" field from preprocessing
print("\n" + "=" * 80)
print("CHECKING 'index' FIELD FROM PREPROCESSING SCRIPT")
print("=" * 80)

# For each monitor_index, check what the actual preprocessing "index" is
monitor_to_preprocessing_index = {}
for monitor_idx, infos_list in monitor_index2infos.items():
    if len(infos_list) > 0:
        # Each info should have the same question_type and index
        first_info = infos_list[0]
        preprocessing_idx = first_info.get('uid', 'NO_UID')  # Actually check what fields exist
        question_type = first_info.get('question_type', 'NO_TYPE')
        monitor_to_preprocessing_index[monitor_idx] = {
            'question_type': question_type,
            'num_samples': len(infos_list),
            'all_question_types': [info.get('question_type') for info in infos_list]
        }

print(f"\nFirst 30 monitor indices and their question types:")
for monitor_idx in sorted(monitor_index2infos.keys())[:30]:
    info = monitor_to_preprocessing_index.get(monitor_idx, {})
    q_type = info.get('question_type', 'unknown')
    num_samples = info.get('num_samples', 0)
    print(f"  monitor_idx={monitor_idx}: {q_type} (n={num_samples})")

# Check the pairing pattern
print("\n" + "=" * 80)
print("CHECKING EVEN/ODD PAIRING PATTERN")
print("=" * 80)

even_types = defaultdict(int)
odd_types = defaultdict(int)

for monitor_idx in sorted(monitor_index2infos.keys()):
    q_type = monitor_to_preprocessing_index[monitor_idx]['question_type']
    if monitor_idx % 2 == 0:
        even_types[q_type] += 1
    else:
        odd_types[q_type] += 1

print("\nEven monitor indices (0, 2, 4, ...):")
for q_type, count in even_types.items():
    print(f"  {q_type}: {count}")

print("\nOdd monitor indices (1, 3, 5, ...):")
for q_type, count in odd_types.items():
    print(f"  {q_type}: {count}")

# Look at actual consecutive pairs
print("\n" + "=" * 80)
print("EXAMINING CONSECUTIVE PAIRS")
print("=" * 80)

sorted_indices = sorted(monitor_index2infos.keys())
print("\nFirst 20 consecutive pairs:")
for i in range(0, min(40, len(sorted_indices)), 2):
    idx1 = sorted_indices[i]
    idx2 = sorted_indices[i+1] if i+1 < len(sorted_indices) else None

    type1 = monitor_to_preprocessing_index.get(idx1, {}).get('question_type', 'unknown')
    type2 = monitor_to_preprocessing_index.get(idx2, {}).get('question_type', 'unknown') if idx2 else 'N/A'

    print(f"  Pair {i//2}: ({idx1}:{type1}, {idx2}:{type2})")

# Check if there's an actual "index" field from the preprocessing
print("\n" + "=" * 80)
print("CHECKING FOR PREPROCESSING 'index' FIELD IN INFO")
print("=" * 80)

if len(monitor_index2infos) > 0:
    first_monitor_idx = sorted(monitor_index2infos.keys())[0]
    first_info = monitor_index2infos[first_monitor_idx][0]
    print(f"\nFields in first info dict:")
    for key in first_info.keys():
        print(f"  {key}: {first_info[key]}")

# Check duplicates in the indices you saw
print("\n" + "=" * 80)
print("CHECKING DUPLICATE MONITOR INDICES")
print("=" * 80)

# Get all monitor indices from the batch
if 'monitor_index' in data:
    batch_monitor_indices = data['monitor_index']
    print(f"\nTotal entries in batch: {len(batch_monitor_indices)}")

    from collections import Counter
    index_counts = Counter(batch_monitor_indices)
    duplicates = {idx: count for idx, count in index_counts.items() if count > 1}

    print(f"Number of duplicate monitor indices: {len(duplicates)}")
    print(f"First 10 duplicates: {list(duplicates.items())[:10]}")

    # Check question types for duplicates
    print("\nFor duplicate monitor indices, checking question types:")
    for dup_idx in list(duplicates.keys())[:5]:
        if dup_idx in monitor_index2infos:
            q_types = [info['question_type'] for info in monitor_index2infos[dup_idx]]
            print(f"  monitor_idx={dup_idx} (appears {duplicates[dup_idx]} times): types={set(q_types)}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
