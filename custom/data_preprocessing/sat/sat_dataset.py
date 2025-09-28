"""
Generate sat dataset (single-attempt).

Functionality:
- Accept --train_size/--test_size to control number of generated samples per split
- Accept --min_level and --max_level as difficulty bounds for filtering
- Generate prompts for single-attempt sat tasks
- Create only single-attempt variant

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)  # includes nums, target, and metadata

This script produces a dataset under --local_dir:
  - train.parquet and test.parquet

Optional: --hdfs_dir to mirror the directory to HDFS.
"""

import argparse
import os
from re import L
from typing import Any, Dict, List, Tuple
import json

import datasets
import math


def build_single_attempt_prompt(sat: List[List[str]]) -> List[Dict[str, str]]:
    """Build messages for single-attempt full sat task.

    Requires a single final expression inside <answer>...</answer> and thinking in <think>...</think>.
    """
    assert len(sat)>0

    sat_str = "&".join([f"({'|'.join(clause)})" for clause in sat])
    
    content = (
        f"Consider the sat problem defined by the following term: {sat_str}. Given this sat, find an assigment for all variables that makes the whole term true. "
        "Think through the task step by step, and verify your proposed path within <think> </think> tags. "
        "Then, provide the final path within <answer> </answer> tags, for example, <answer>a:true,b:false,c:true,d:true</answer>."
    )
    return [{"content": content, "role": "user"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory to save outputs",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training examples to export (-1 for all filtered)")
    parser.add_argument("--test_size", type=int, default=-1, help="Number of test examples to export (-1 for all filtered)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="If set, do not shuffle; instead sort examples by level (#clauses) ascending.",
    )
    parser.add_argument(
        "--min_level",
        type=int,
        default=2,
        help="Minimum difficulty level 2-8. Keep rows that are k-sat where k >= min_level.",
    )
    parser.add_argument(
        "--max_level",
        type=int,
        default=8,
        help="Maximum difficulty level 2-8. Keep rows that are k-sat where k  <= max_level.",
    )
    parser.add_argument(
        "--n_print",
        type=int,
        default=1,
        help="Number of first dataset items to print per split for sanity check",
    )
    # Curriculum options
    parser.add_argument(
        "--soft_curriculum",
        action="store_true",
        help="If set, use soft curriculum level weighting for the train split (test remains balanced).",
    )
    parser.add_argument(
        "--phased_curriculum",
        action="store_true",
        help=(
            "If set, use a phased curriculum schedule (by training step %) for the train split: "
            "0-10%: 70% d=3, 25% d=4, 5% d>=5; "
            "10-25%: 40% d=3-4, 40% d=5, 20% d=6; "
            "25-45%: 20% d=3-4, 40% d=5-6, 40% d=7-8; "
            "45-70%: 10% d=3-4, 30% d=5-6, 60% d=7-8; "
            "70-100%: 10% d=3-5, 40% d=6-7, 50% d=8."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="Rehearsal floor rho used in soft curriculum, recommended in [0.02, 0.1].",
    )
    parser.add_argument(
        "--curriculum_steps",
        type=int,
        default=200,
        help="Number of t steps in [0,1] to average per-step level probabilities for soft curriculum.",
    )
    return parser.parse_args()


def _select_split(dataset: datasets.Dataset, size: int) -> datasets.Dataset:
    if size is None or size < 0:
        return dataset
    if size > len(dataset):
        raise ValueError(f"Requested size ({size}) exceeds available rows ({len(dataset)}).")
    return dataset.select(range(size))


def _add_level_column(ex: Dict[str, Any]) -> Dict[str, int]:
    sat = ex.get("sat", [[]])
    return {"_level": len(sat[0]) if isinstance(sat, list) else -1}


def _balanced_select_by_level(
    dataset: datasets.Dataset,
    size: int,
    seed: int,
    no_shuffle: bool,
    min_level: int,
    max_level: int,
) -> datasets.Dataset:
    """Select roughly equal counts per difficulty level within [min_level, max_level].

    If size < 0, return the full dataset ordered per no_shuffle policy.
    Otherwise, distribute the requested size evenly across available levels.
    """
    if dataset is None or len(dataset) == 0:
        return dataset

    ds_with_level = dataset.map(_add_level_column)

    # Identify available levels in the filtered dataset within the requested bounds
    try:
        levels_present = [lvl for lvl in ds_with_level.unique("_level") if isinstance(lvl, int)]
    except Exception:
        levels_present = []
    levels = sorted([lvl for lvl in levels_present if min_level <= lvl <= max_level])
    if len(levels) == 0:
        return ds_with_level

    # If selecting all, just order per policy and return
    if size is None or size < 0:
        if no_shuffle:
            return ds_with_level.sort("_level")
        else:
            return ds_with_level.shuffle(seed=seed)

    # Build per-level subsets
    per_level_ds = []
    per_level_len = []
    for lvl in levels:
        lvl_ds = ds_with_level.filter(lambda ex, _lvl=lvl: ex["_level"] == _lvl)
        if not no_shuffle:
            lvl_ds = lvl_ds.shuffle(seed=seed)
        per_level_ds.append(lvl_ds)
        per_level_len.append(len(lvl_ds))

    n_levels = len(levels)
    # Initial fair quotas
    base = size // n_levels
    rem = size % n_levels
    quotas = [base + (1 if i < rem else 0) for i in range(n_levels)]

    # Cap by availability
    initial_take = [min(quotas[i], per_level_len[i]) for i in range(n_levels)]
    total_selected = sum(initial_take)

    # Distribute leftovers round-robin by level index while capacity remains
    leftovers = size - total_selected
    remaining_cap = [per_level_len[i] - initial_take[i] for i in range(n_levels)]
    extra_take = [0] * n_levels
    while leftovers > 0 and any(c > 0 for c in remaining_cap):
        for i in range(n_levels):
            if leftovers <= 0:
                break
            if remaining_cap[i] > 0:
                extra_take[i] += 1
                remaining_cap[i] -= 1
                leftovers -= 1

    final_take = [initial_take[i] + extra_take[i] for i in range(n_levels)]

    # Select per-level splits
    selected_per_level = []
    for i in range(n_levels):
        take_n = final_take[i]
        if take_n <= 0:
            continue
        selected_per_level.append(per_level_ds[i].select(range(take_n)))

    if len(selected_per_level) == 0:
        # Fallback to empty selection
        return ds_with_level.select([])

    combined = datasets.concatenate_datasets(selected_per_level)
    if no_shuffle:
        combined = combined.sort("_level")
    else:
        combined = combined.shuffle(seed=seed)
    return combined


def _soft_curriculum_level_probs(
    levels: List[int],
    min_level: int,
    max_level: int,
    rho: float,
    steps: int,
) -> List[float]:
    """Compute time-averaged per-level probabilities under soft curriculum.

    Uses µ(t) = min + (max - min) * t^0.6 and σ(t) = 0.7 + 1.8 t.
    At each t, weights are w_d(t) = exp(- (d - µ(t))^2 / (2 σ(t)^2)), then floored by rho.
    We normalize per-t to get probabilities and average across t in [0,1].
    """
    if not levels:
        return []
    ordered_levels = sorted(levels)
    steps = max(int(steps), 1)
    rho = max(0.0, float(rho))

    accum = [0.0 for _ in ordered_levels]
    for s in range(steps):
        t = 0.0 if steps == 1 else (s / (steps - 1))
        mu = min_level + (max_level - min_level) * (t ** 0.6)
        sigma = 0.7 + 1.8 * t
        # Compute floored weights
        weights = []
        for lvl in ordered_levels:
            w = math.exp(-((lvl - mu) ** 2) / (2.0 * (sigma ** 2)))
            if rho > 0.0:
                w = max(w, rho)
            weights.append(w)
        total_w = sum(weights)
        if total_w <= 0.0:
            # Fallback to uniform
            probs_t = [1.0 / len(ordered_levels) for _ in ordered_levels]
        else:
            probs_t = [w / total_w for w in weights]
        for i, p in enumerate(probs_t):
            accum[i] += p
    # Average and renormalize
    avg = [a / steps for a in accum]
    total = sum(avg)
    if total <= 0.0:
        return [1.0 / len(ordered_levels) for _ in ordered_levels]
    return [a / total for a in avg]


def _weighted_select_by_level_soft(
    dataset: datasets.Dataset,
    size: int,
    seed: int,
    no_shuffle: bool,
    min_level: int,
    max_level: int,
    rho: float,
    curriculum_steps: int,
) -> datasets.Dataset:
    """Select examples per level according to soft curriculum probabilities.

    If size < 0, returns the full dataset ordered per policy (soft curriculum does not change ordering).
    """
    if dataset is None or len(dataset) == 0:
        return dataset

    ds_with_level = dataset.map(_add_level_column)

    try:
        levels_present = [lvl for lvl in ds_with_level.unique("_level") if isinstance(lvl, int)]
    except Exception:
        levels_present = []
    levels = sorted([lvl for lvl in levels_present if min_level <= lvl <= max_level])
    if len(levels) == 0:
        return ds_with_level

    if size is None or size < 0:
        # Return all; keep ordering policy similar to balanced selector
        if no_shuffle:
            return ds_with_level.sort("_level")
        else:
            return ds_with_level.shuffle(seed=seed)

    # Build per-level subsets
    per_level_ds = []
    per_level_len = []
    for lvl in levels:
        lvl_ds = ds_with_level.filter(lambda ex, _lvl=lvl: ex["_level"] == _lvl)
        if not no_shuffle:
            lvl_ds = lvl_ds.shuffle(seed=seed)
        per_level_ds.append(lvl_ds)
        per_level_len.append(len(lvl_ds))

    probs = _soft_curriculum_level_probs(levels, min_level, max_level, rho=rho, steps=curriculum_steps)

    # Initial integer quotas via floor, capped by availability
    ideal = [size * p for p in probs]
    floors = [int(x) for x in ideal]
    initial_take = [min(floors[i], per_level_len[i]) for i in range(len(levels))]
    total_selected = sum(initial_take)
    leftovers = size - total_selected

    # Remainders for additional allocation
    remainders = [ideal[i] - floors[i] for i in range(len(levels))]
    remaining_cap = [per_level_len[i] - initial_take[i] for i in range(len(levels))]

    # Order indices by remainder descending once; iterate cyclically while leftovers remain
    indices_by_rem = sorted(range(len(levels)), key=lambda i: remainders[i], reverse=True)
    extra_take = [0] * len(levels)
    while leftovers > 0 and any(c > 0 for c in remaining_cap):
        for i in indices_by_rem:
            if leftovers <= 0:
                break
            if remaining_cap[i] > 0:
                extra_take[i] += 1
                remaining_cap[i] -= 1
                leftovers -= 1

    final_take = [initial_take[i] + extra_take[i] for i in range(len(levels))]

    # Select per-level splits
    selected_per_level = []
    for i in range(len(levels)):
        take_n = final_take[i]
        if take_n <= 0:
            continue
        selected_per_level.append(per_level_ds[i].select(range(take_n)))

    if len(selected_per_level) == 0:
        return ds_with_level.select([])

    combined = datasets.concatenate_datasets(selected_per_level)
    if no_shuffle:
        combined = combined.sort("_level")
    else:
        combined = combined.shuffle(seed=seed)
    return combined


def _phased_curriculum_level_probs(
    levels: List[int],
    steps: int,
) -> List[float]:
    """Compute time-averaged per-level probabilities under a phased curriculum schedule.

    Phases by training step t in [0,1]:
    Sure, here are the 'd' values reduced by 1:
    - [0.00, 0.10): 70% d=3, 25% d=4, 5% d>=5
    - [0.10, 0.25): 40% d=3-4, 40% d=5, 20% d=6
    - [0.25, 0.45): 20% d=3-4, 40% d=5-6, 40% d=7-8
    - [0.45, 0.70): 10% d=3-4, 30% d=5-6, 60% d=7-8
    - [0.70, 1.00]: 10% d=3-5, 40% d=6-7, 50% d=8

    Group weights within a phase are distributed uniformly among the present levels in each group,
    then renormalized to sum to 1 per t. We average across t and renormalize.
    """
    if not levels:
        return []
    ordered_levels = sorted(levels)
    steps = max(int(steps), 1)

    # Define schedule as (start_t, end_t, [(selector_fn, weight), ...])
    def sel_eq(x):
        return lambda lv: [l for l in lv if l == x]

    def sel_in(rng):
        return lambda lv: [l for l in lv if l in rng]

    def sel_ge(x):
        return lambda lv: [l for l in lv if l >= x]

    schedule = [
        (
            0.0,
            0.10,
            [
                (sel_eq(3), 0.70),
                (sel_eq(4), 0.25),
                (sel_ge(5), 0.05),
            ],
        ),
        (
            0.10,
            0.25,
            [
                (sel_in({3, 4}), 0.40),
                (sel_eq(5), 0.40),
                (sel_eq(6), 0.20),
            ],
        ),
        (
            0.25,
            0.45,
            [
                (sel_in({3, 4}), 0.20),
                (sel_in({5, 6}), 0.40),
                (sel_in({7, 8}), 0.40),
            ],
        ),
        (
            0.45,
            0.70,
            [
                (sel_in({3, 4}), 0.10),
                (sel_in({5, 6}), 0.30),
                (sel_in({7, 8}), 0.60),
            ],
        ),
        (
            0.70,
            1.00,
            [
                (lambda lv: [l for l in lv if 3 <= l <= 5], 0.10),
                (sel_in({6, 7}), 0.40),
                (sel_eq(8), 0.50),
            ],
        ),
    ]

    accum = [0.0 for _ in ordered_levels]
    for s in range(steps):
        t = 0.0 if steps == 1 else (s / (steps - 1))
        # Find phase for t
        phase = None
        for i, (start, end, groups) in enumerate(schedule):
            is_last = i == (len(schedule) - 1)
            if (t >= start) and ((t < end) or (is_last and t <= end)):
                phase = groups
                break

        weights = [0.0 for _ in ordered_levels]
        # Apply group allocations
        for selector, group_w in phase:
            group_levels = selector(ordered_levels)
            if not group_levels or group_w <= 0.0:
                continue
            per_level = group_w / float(len(group_levels))
            for idx, lvl in enumerate(ordered_levels):
                if lvl in group_levels:
                    weights[idx] += per_level
        total_w = sum(weights)
        probs_t = [w / total_w for w in weights]
        for i, p in enumerate(probs_t):
            accum[i] += p

    avg = [a / steps for a in accum]
    total = sum(avg)
    if total <= 0.0:
        return [1.0 / len(ordered_levels) for _ in ordered_levels]
    return [a / total for a in avg]


def _weighted_select_by_level_phased(
    dataset: datasets.Dataset,
    size: int,
    seed: int,
    no_shuffle: bool,
    min_level: int,
    max_level: int,
    curriculum_steps: int,
) -> datasets.Dataset:
    """Build the train split in contiguous phases with fixed per-phase distributions.

    - Keeps batches mixed within each phase (per-phase shuffle), not blocked by difficulty.
    - Maintains phase probabilities exactly (up to integer rounding and availability).
    - Ignores curriculum_steps for phased scheduler; the dataset size approximates total steps.
    - Returns all if size < 0 using the standard ordering policy.
    """
    if dataset is None or len(dataset) == 0:
        return dataset

    ds_with_level = dataset.map(_add_level_column)

    try:
        levels_present = [lvl for lvl in ds_with_level.unique("_level") if isinstance(lvl, int)]
    except Exception:
        levels_present = []
    levels = sorted([lvl for lvl in levels_present if min_level <= lvl <= max_level])
    if len(levels) == 0:
        return ds_with_level

    if size is None or size < 0:
        if no_shuffle:
            return ds_with_level.sort("_level")
        else:
            return ds_with_level.shuffle(seed=seed)

    # Build per-level subsets and capacities
    per_level_ds = []
    per_level_len = []
    for lvl in levels:
        lvl_ds = ds_with_level.filter(lambda ex, _lvl=lvl: ex["_level"] == _lvl)
        per_level_ds.append(lvl_ds)
        per_level_len.append(len(lvl_ds))

    # Phase schedule with durations summing to 1.0 (last phase extends to 1.0)
    def levels_in(group):
        return [l for l in levels if l in group]

    def levels_ge(x):
        return [l for l in levels if l >= x]

    def levels_between(a, b):
        return [l for l in levels if a <= l <= b]

    schedule = [
        # (fraction_of_training, [ (levels_selector, weight), ... ])
        (0.10, [(lambda: levels_in({3}), 0.70), (lambda: levels_in({4}), 0.25), (lambda: levels_ge(5), 0.05)]),
        (0.15, [(lambda: levels_in({3, 4}), 0.40), (lambda: levels_in({5}), 0.40), (lambda: levels_in({6}), 0.20)]),
        (0.20, [(lambda: levels_in({3, 4}), 0.20), (lambda: levels_in({5, 6}), 0.40), (lambda: levels_in({7, 8}), 0.40)]),
        (0.25, [(lambda: levels_in({3, 4}), 0.10), (lambda: levels_in({5, 6}), 0.30), (lambda: levels_in({7, 8}), 0.60)]),
        (0.30, [(lambda: levels_between(3, 5), 0.10), (lambda: levels_in({6, 7}), 0.40), (lambda: levels_in({8}), 0.50)]),
    ]

    # Normalize schedule fractions to sum to 1.0 exactly via rounding on counts
    ideal_phase_counts = [size * frac for frac, _ in schedule]
    phase_floors = [int(x) for x in ideal_phase_counts]
    phase_counts = list(phase_floors)
    total_phase = sum(phase_counts)
    phase_leftover = max(0, size - total_phase)
    phase_remainders = [ideal_phase_counts[i] - phase_floors[i] for i in range(len(schedule))]
    indices_by_phase_rem = sorted(range(len(schedule)), key=lambda i: phase_remainders[i], reverse=True)
    idx_cycle = 0
    while phase_leftover > 0:
        i = indices_by_phase_rem[idx_cycle % len(indices_by_phase_rem)]
        phase_counts[i] += 1
        phase_leftover -= 1
        idx_cycle += 1

    # Track remaining capacity and offsets per level to avoid duplicates
    remaining_cap = per_level_len[:]
    level_offsets = [0 for _ in levels]

    phase_datasets = []

    def allocate_counts_for_phase(level_probs, needed, remaining):
        if needed <= 0 or not level_probs:
            return [0] * len(level_probs)
        total_p = sum(level_probs)
        if total_p <= 0.0:
            # fallback uniform over levels with remaining capacity
            mask = [1 if remaining[i] > 0 else 0 for i in range(len(level_probs))]
            m = sum(mask)
            if m == 0:
                return [0] * len(level_probs)
            level_probs = [mask[i] / m for i in range(len(level_probs))]
        else:
            level_probs = [p / total_p for p in level_probs]
        ideal = [needed * p for p in level_probs]
        floors = [int(x) for x in ideal]
        take = [min(floors[i], remaining[i]) for i in range(len(floors))]
        selected = sum(take)
        leftover = needed - selected
        remainders = [ideal[i] - floors[i] for i in range(len(floors))]
        indices = sorted(range(len(floors)), key=lambda i: remainders[i], reverse=True)
        while leftover > 0 and any(remaining[i] - take[i] > 0 for i in range(len(floors))):
            for i in indices:
                if leftover <= 0:
                    break
                if remaining[i] - take[i] > 0:
                    take[i] += 1
                    leftover -= 1
        return take

    for phase_idx, (frac, groups) in enumerate(schedule):
        needed = phase_counts[phase_idx]
        if needed <= 0:
            continue
        # Build per-level weights for this phase
        per_level_weights = [0.0 for _ in levels]
        group_total_weight = 0.0
        for selector_fn, w in groups:
            grp_levels = selector_fn()
            if not grp_levels or w <= 0.0:
                continue
            group_total_weight += w
            per_level_share = w / float(len(grp_levels))
            for idx, lvl in enumerate(levels):
                if lvl in grp_levels:
                    per_level_weights[idx] += per_level_share
        # If some groups were empty (due to bounds), renormalize
        total_w = sum(per_level_weights)
        if total_w <= 0.0:
            # fallback to uniform over available levels with capacity
            per_level_weights = [1.0 if remaining_cap[i] > 0 else 0.0 for i in range(len(levels))]
        # Allocate integer counts per level within this phase
        take_by_level = allocate_counts_for_phase(per_level_weights, needed, remaining_cap)

        # Build phase dataset portion; take consecutive ranges with offsets to avoid duplicates
        selected_level_datasets = []
        for i, take_n in enumerate(take_by_level):
            if take_n <= 0:
                continue
            start = level_offsets[i]
            end = start + take_n
            if start < per_level_len[i] and end <= per_level_len[i]:
                selected_level_datasets.append(per_level_ds[i].select(range(start, end)))
                level_offsets[i] = end
                remaining_cap[i] -= take_n
        if len(selected_level_datasets) == 0:
            continue
        phase_ds = datasets.concatenate_datasets(selected_level_datasets)
        if no_shuffle:
            phase_ds = phase_ds.sort("_level")
        else:
            phase_ds = phase_ds.shuffle(seed=seed + phase_idx)
        phase_datasets.append(phase_ds)

    if len(phase_datasets) == 0:
        return ds_with_level.select([])
    return datasets.concatenate_datasets(phase_datasets)


def map_row_to_output(
    example: Dict[str, Any],
    idx: int,
    split_label: str,
) -> Dict[str, Any]:
    """Map a single task row to the output format."""
    sat = example.get("sat", [])
    raw_sat = example.get("raw_sat", "[]")
    variable_labels = example.get("variable_labels", "[]")
    solution = example.get("solution", "")
    
    messages = build_single_attempt_prompt(sat)
    
    data_source = f"sat-{len(sat[0])}"
    ability = "math"
    extra_info = {
        "split": split_label,
        "index": idx,
        "variable_labels": variable_labels,
        "raw_sat":raw_sat,
        "num_clauses": len(sat),
        "num_variables": len(solution.split(",")),
        "solution": solution,
        "num_attempts": 1,
        "max_allowed_attempts": 1,
        "prompt_style": "single",
    }
    
    reward_model = {"style": "rule", "ground_truth": solution, "target": solution}
    
    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }

import random
import string
from itertools import product, chain

def generate_variable_labels():
    """Generates an infinite sequence of short, unique variable labels."""
    # Single lower letters (a-z)
    for char in string.ascii_lowercase:
        yield char


def generate_sat_data(level: int, num_variables: int, num_clauses: int) -> dict:
    """
    Generates a single sat problem with a guaranteed shortest path length.

    Args:
        level: Number fo variables each clause have (difficulty) 2-sat, 3-sat, 4-sat, etc.
        num_variables: Total number of variables 
        num_clauses:  Total number of clauses

    Returns:
        A dictionary containing the 'sat' and one 'solution'. The sat is a list containing the clauses. 
        Each clauses itself is a list of variables.
        Each variable is either itself or its negation (e.g., A or ~A).
    """
    if level < 2:
        raise ValueError("Level must be at least 2.")
    if num_variables < level:
        raise ValueError("Number of variables must be at least equal to the level.")

    # Create a guaranteed solution by assigning a random boolean value to each variable.
    solution = {i: random.choice([True, False]) for i in range(1, num_variables + 1)}

    raw_clauses = []
    variable_pool = list(range(1, num_variables + 1))

    duplicate_count=0
    while len(raw_clauses)<num_clauses:
        # Randomly select 'level' unique variables for the current clause
        chosen_variables = random.sample(variable_pool, level)

        current_clause = []
        for var in chosen_variables:
            # Randomly decide whether to negate the variable
            if random.choice([True, False]):
                current_clause.append(-var)
            else:
                current_clause.append(var)
        
        # Check if the generated clause is satisfied by the solution.
        # A clause is satisfied if at least one of its literals is true.
        is_satisfied = any(
            (literal > 0 and solution[abs(literal)]) or \
            (literal < 0 and not solution[abs(literal)]) \
            for literal in current_clause
        )

        # If the clause is not satisfied by the solution, we must alter it to make it true.
        if not is_satisfied:
            # Pick one literal at random from the clause to flip.
            index_to_flip = random.randrange(level)
            current_clause[index_to_flip] = -current_clause[index_to_flip] 
            
        # Ensure the clause is not already added 
        # A set is used for efficient checking of duplicates
        if tuple(sorted(current_clause)) not in {tuple(sorted(c)) for c in raw_clauses}:
            raw_clauses.append(current_clause)
            duplicate_count = 0
        else:
            duplicate_count += 1
            if duplicate_count >= 10:
                break

    label_generator = generate_variable_labels()
    variable_labels = [None]+[next(label_generator) for _ in range(num_variables)]

    clauses = []
    for clause in raw_clauses:
        new_clause = []
        for i in range(level):
            if clause[i]>0:
                new_clause.append(variable_labels[clause[i]])
            else:
                new_clause.append(f"~{variable_labels[-clause[i]]}")
        clauses.append(new_clause)

    solution_str = ",".join([f"{variable_labels[i]}:{solution[i]}" for i in range(1, num_variables + 1)])

    return {
        "variable_labels": json.dumps(variable_labels, indent=2),
        "raw_sat": json.dumps(raw_clauses, indent=2),
        "sat": clauses,
        "solution": solution_str 
    }


if __name__ == "__main__":
    args = parse_args()

    if args.min_level < 2 or args.min_level > 8:
        raise ValueError("--min_level must be between 2 and 8 inclusive")
    if args.max_level < 2 or args.max_level > 8:
        raise ValueError("--max_level must be between 2 and 8 inclusive")
    if args.min_level > args.max_level:
        raise ValueError("--min_level must be less than or equal to --max_level")
    if args.train_size < -1 or args.test_size < -1:
        raise ValueError("--train_size/--test_size must be -1 or non-negative")
    if args.soft_curriculum and args.phased_curriculum:
        raise ValueError("Choose only one curriculum mode: --soft_curriculum OR --phased_curriculum")

    # Load dataset
    print("Generating sat dataset...") # I want to make the below dataset gemini
    train_samples = []
    test_samples = []
    random.seed(0)
    clause_to_variable_ratio={
        2: 1,
        3: 4.26,
        4: 9.93,
        5: 21.1,
        6: 43.0,
        7: 87.5,
        8: 176.5,
        9: 354.0
    }
    for level in range(args.min_level, args.max_level + 1):
        all_num_variables=[3,3,5,5,5,7,7,7,9,9,9]
        all_num_variables = [x for x in all_num_variables if x>level ]

        for num_variables in all_num_variables:
            for _ in range(1000): # Generate 5000 examples per level for the training pool
                train_samples.append(generate_sat_data(level, num_variables, int(num_variables*(clause_to_variable_ratio[level]+1))))
            for _ in range(200): # Generate 200 examples per level for the test pool
                test_samples.append(generate_sat_data(level, num_variables, int(num_variables*(clause_to_variable_ratio[level]+1))))

    print(train_samples[0])
    # Convert the lists of dictionaries into Hugging Face Dataset objects
    train_filtered = datasets.Dataset.from_list(train_samples)
    test_filtered = datasets.Dataset.from_list(test_samples)


    # Order data and select for train/test
    if args.soft_curriculum:
        print("Selecting train split with soft curriculum weighting...")
        train_raw = _weighted_select_by_level_soft(
            dataset=train_filtered,
            size=args.train_size,
            seed=args.seed,
            no_shuffle=args.no_shuffle,
            min_level=args.min_level,
            max_level=args.max_level,
            rho=args.rho,
            curriculum_steps=args.curriculum_steps,
        )
    elif args.phased_curriculum:
        print("Selecting train split with phased curriculum weighting...")
        train_raw = _weighted_select_by_level_phased(
            dataset=train_filtered,
            size=args.train_size,
            seed=args.seed,
            no_shuffle=args.no_shuffle,
            min_level=args.min_level,
            max_level=args.max_level,
            curriculum_steps=args.curriculum_steps,
        )
    else:
        print("Ordering and selecting train split with balanced per-level sampling...")
        train_raw = _balanced_select_by_level(
            dataset=train_filtered,
            size=args.train_size,
            seed=args.seed,
            no_shuffle=args.no_shuffle,
            min_level=args.min_level,
            max_level=args.max_level,
        )
    test_raw = _balanced_select_by_level(
        dataset=test_filtered,
        size=args.test_size,
        seed=args.seed,
        no_shuffle=args.no_shuffle,
        min_level=args.min_level,
        max_level=args.max_level,
    )

    # Map to single variant using the same base tasks
    def make_map_fn(split_label: str):
        def _inner(example, idx):
            return map_row_to_output(
                example=example,
                idx=idx,
                split_label=split_label,
            )
        return _inner

    # Create dataset
    print("Creating dataset...")
    train_dataset = train_raw.map(function=make_map_fn("train"), with_indices=True, remove_columns=train_raw.column_names)
    test_dataset = test_raw.map(function=make_map_fn("test"), with_indices=True, remove_columns=test_raw.column_names)

    # Prepare output directory
    os.makedirs(args.local_dir, exist_ok=True)

    # Save parquet files
    print("Saving parquet files...")
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    # Print samples
    if args.n_print > 0 and len(train_dataset) > 0:
        print(f"Printing {args.n_print} sample(s) from train data:")
        for i in range(min(args.n_print, len(train_dataset))):
            print(train_dataset[i])

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=args.local_dir, dst=args.hdfs_dir)
        except Exception as e:
            print(f"Warning: failed to mirror to HDFS due to: {e}")

    print(
        f"Filtered counts (level in [{args.min_level}..{args.max_level}]): train={len(train_filtered)}, test={len(test_filtered)}; "
        f"exported: train={len(train_dataset)}, test={len(test_dataset)}"
    )

    # Print per-level counts for each split
    def _get_levels_from_dataset(ds: datasets.Dataset):
        try:
            if "data_source" in ds.column_names:
                return [int(str(src).split("-")[-1]) for src in ds["data_source"]]
        except Exception:
            pass
        try:
            infos = ds["extra_info"]
            return [int(len(info.get("only_path", -1))) if isinstance(info, dict) else -1 for info in infos]
        except Exception:
            return []

    def _print_level_counts(ds: datasets.Dataset, name: str):
        from collections import Counter
        levels = _get_levels_from_dataset(ds)
        if not levels:
            print(f"No level info available for {name} split.")
            return
        counts = Counter(levels)
        ordered = sorted(counts.items())
        counts_str = ", ".join([f"{lvl}:{cnt}" for lvl, cnt in ordered])
        print(f"Level counts for {name}: {counts_str}")

    _print_level_counts(train_dataset, "train")
    _print_level_counts(test_dataset, "test")

# Example usage:
# PYTHONPATH=. python custom/data_preprocessing/sat/sat_dataset.py \
#   --local_dir $HF_HOME/data/full_sat --train_size 1000 --test_size 100 \
#   --min_level 3 --max_level 9
