#!/usr/bin/env python3
"""
Generate synthetic number guessing dataset (single-attempt) where each instance provides
a **description** of the candidate set (not the explicit list). One secret number is drawn
from that set using uniform or geometric-power weighting.

Maintains your schema and CLI feel:
- --set_size, --secret_selection {uniform,power}, --power_ratio, --weight_order {asc,desc,shuffle}
- --puzzle_types to choose one or more description templates:
    * residue_class  (exact set_size)
    * letter_shift   (exact set_size, A1Z26 with A=0..Z=25 then add a base)
    * poly_pow2      (set_size ∈ {2,3}: 2^x where x solves a polynomial with integer roots)

Output columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)  # includes description, hidden support, probs, type, etc.
"""

import argparse
import os
import random
from typing import Any, Dict, List, Tuple
import math
import datasets

# ------------------------- Helpers -------------------------

def normalize(weights: List[float]) -> List[float]:
    s = sum(weights)
    if s <= 0:  # safety
        return [1.0 / len(weights)] * len(weights)
    return [w / s for w in weights]

def choose_with_power(support: List[int], power_ratio: float, weight_order: str, rng: random.Random) -> Tuple[int, List[float], List[int]]:
    """Apply geometric weights r, r^2, ..., r^n along an order, then sample a secret."""
    sorted_support = sorted(support)
    n = len(sorted_support)

    if weight_order == "asc":
        order_idx = list(range(n))               # smallest -> largest
    elif weight_order == "desc":
        order_idx = list(range(n-1, -1, -1))     # largest -> smallest
    else:  # shuffle
        order_idx = list(range(n))
        rng.shuffle(order_idx)

    ordered = [sorted_support[i] for i in order_idx]
    raw = [power_ratio ** (i + 1) for i in range(n)]
    probs = normalize(raw)

    secret = rng.choices(ordered, weights=probs, k=1)[0]

    # Return probs aligned to the original sorted order for auditing
    # Map ordered probs back to sorted order
    back = [0.0] * n
    for i, idx in enumerate(order_idx):
        back[idx] = probs[i]
    return secret, back, order_idx

def bias_sentence(secret_selection: str, weight_order: str) -> str:
    if secret_selection != "power":
        return ""  # uniform; silent by default
    if weight_order == "asc":
        return " Larger numbers have higher probability."
    if weight_order == "desc":
        return " Smaller numbers have higher probability."
    # shuffle: keep it implicit to avoid leaking the permutation
    return ""

# ------------------------- Prompt builders -------------------------

SYSTEM_MSG = (
    "You are a helpful assistant. You will be given a task to guess a hidden number "
    "from a set that is **described**, not listed explicitly. "
    "You will be asked to answer the question a couple of times, and you will win if you guess the number correctly in at least one of your attempts. "
    "Pay attention to your attempt number, and the total number of attempts you have. Therefore, you should diversify your attempts, and devise a specific strategy considering your current attempt number and the total number of attempts you have. "
    "Think step by step between <think> and </think> and then provide your answer inside <answer> YOUR GUESS HERE </answer> tags as a single integer. "
    "Don't output any other text after the </answer>."
)

def build_prompt_implicit(description: str, bias_text: str) -> List[Dict[str, str]]:
    user_msg = (
        "I have chosen a secret number from the following described set: "
        f"{description.strip()}.{bias_text} "
        "Can you guess which one it is? Please think step by step between <think> and </think> and then provide your final answer inside <answer> YOUR GUESS HERE </answer> tags as a single integer."
    )
    return [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg}]

# ------------------------- Puzzle templates -------------------------
# Each template MUST return exactly set_size elements within [min_number, max_number].

def gen_residue_class(set_size: int, min_number: int, max_number: int, rng: random.Random) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Numbers in [L, U] that are r more than a multiple of m.
    We pick m, r, and a valid start so that the count is exactly set_size.
    """
    if set_size <= 0:
        raise ValueError("set_size must be positive")

    # Try a few times to find (m, r, start) that fit the global bounds
    for _ in range(100):
        m = rng.choice([3,4,5,6,7,8,9,10,11,12])
        r = rng.randrange(m)

        # Smallest >= min_number with n % m == r
        a_min = min_number + ((r - (min_number % m)) % m)
        # Largest possible start so that a_min + t*m + (set_size-1)*m <= max_number
        max_start = max_number - (set_size - 1) * m
        # First feasible start with the right congruence
        if a_min > max_start:
            continue
        # permissible starts are a_min + k*m up to max_start
        kmax = (max_start - a_min) // m
        start = a_min + rng.randrange(kmax + 1) * m
        support = [start + i * m for i in range(set_size)]
        if all(min_number <= x <= max_number for x in support):
            L, U = support[0], support[-1]
            desc = f"numbers between {L} and {U} (inclusive) that are {r} more than a multiple of {m}"
            return desc, support, {"type": "residue_class", "m": m, "r": r, "L": L, "U": U}

    raise RuntimeError("Failed to generate residue_class with the requested bounds and set_size.")

def gen_letter_shift(set_size: int, min_number: int, max_number: int, rng: random.Random) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Map letters with A=0, B=1, …, Z=25. Choose s distinct letters and a base A so that A+letters
    stays within [min_number, max_number].
    """
    if set_size > 26:
        raise ValueError("letter_shift requires set_size ≤ 26")

    for _ in range(100):
        # Choose a base window so that we have room for s letter offsets
        width = max_number - min_number
        # Pick candidate letter values that can fit into the width
        # We choose a max letter value cap so there are at least s distinct values
        cap = min(25, max(0, width))
        if cap + 1 < set_size:
            # Not enough room with these bounds
            break

        # Random base such that base + cap ≤ max_number and base ≥ min_number
        base_min = min_number
        base_max = max_number - cap
        if base_min > base_max:
            break
        base = rng.randint(base_min, base_max)

        # Sample s distinct letter offsets from [0..cap]
        letter_vals = rng.sample(list(range(cap + 1)), k=set_size)
        letters = [chr(ord('A') + v) for v in letter_vals]
        support = [base + v for v in letter_vals]
        if all(min_number <= x <= max_number for x in support):
            letters_str = ", ".join(letters)
            desc = f"Map letters using A=0, B=1, …, Z=25. Take Base = {base}. Consider Base + {{{letters_str}}} (setwise addition)"
            return desc, support, {"type": "letter_shift", "base": base, "letters": letters}

    raise RuntimeError("Failed to generate letter_shift with the requested bounds and set_size.")

def gen_poly_pow2(set_size: int, min_number: int, max_number: int, rng: random.Random) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Values are 2^x where x solves a polynomial with distinct non-negative integer roots.
    Works only for set_size in {2,3}. Ensures results lie within [min_number, max_number].
    """
    if set_size not in (2,3):
        raise ValueError("poly_pow2 supports set_size ∈ {2,3}")

    # limit roots so 2^root fits bounds
    # If max_number < 1, impossible; otherwise bound root by floor(log2(max_number))
    if max_number < 1:
        raise RuntimeError("poly_pow2 needs max_number ≥ 1")
    max_root = int(math.floor(math.log2(max_number)))
    if max_root < 0:
        raise RuntimeError("poly_pow2: no non-negative root fits bounds")

    for _ in range(200):
        roots = sorted(rng.sample(list(range(0, max_root + 1)), k=set_size))
        support = [2 ** r for r in roots]
        if any(x < min_number or x > max_number for x in support):
            continue

        if set_size == 2:
            r1, r2 = roots
            # (x - r1)(x - r2) = x^2 - (r1+r2)x + (r1*r2)
            b = -(r1 + r2)
            c = r1 * r2
            desc = f"values of 2^x where x is a solution to x^2 + ({b})x + ({c}) = 0"
            meta = {"type": "poly_pow2", "roots": roots, "poly": [1, b, c]}
        else:  # set_size == 3
            r1, r2, r3 = roots
            s1 = r1 + r2 + r3
            s2 = r1*r2 + r1*r3 + r2*r3
            s3 = r1*r2*r3
            # (x - r1)(x - r2)(x - r3) = x^3 - s1 x^2 + s2 x - s3
            desc = f"values of 2^x where x is a solution to x^3 + ({-s1})x^2 + ({s2})x + ({-s3}) = 0"
            meta = {"type": "poly_pow2", "roots": roots, "poly": [1, -s1, s2, -s3]}

        return desc, support, meta

    raise RuntimeError("Failed to generate poly_pow2 within bounds; try increasing max_number.")

TEMPLATES = {
    "residue_class": gen_residue_class,
    "letter_shift": gen_letter_shift,
    "poly_pow2": gen_poly_pow2,  # only if set_size in {2,3}
}

def pick_template(puzzle_types: List[str], set_size: int, min_number: int, max_number: int, rng: random.Random):
    """Choose a template that can satisfy the current set_size/bounds."""
    # Shuffle order each time for variety
    candidates = [t for t in puzzle_types if t in TEMPLATES]
    rng.shuffle(candidates)
    last_err = None
    for t in candidates:
        try:
            # Dry-run a quick capability check by calling and catching errors
            desc, support, meta = TEMPLATES[t](set_size, min_number, max_number, rng)
            return t, desc, support, meta
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No feasible template found for set_size={set_size} in bounds [{min_number},{max_number}]. Last error: {last_err}")

# ------------------------- One task -------------------------

def _generate_one_task(
    rng: random.Random,
    min_number: int,
    max_number: int,
    set_size: int,
    secret_selection: str,
    power_ratio: float,
    weight_order: str,
    puzzle_types: List[str],
) -> Dict[str, Any]:
    # pick a feasible template and produce (desc, support)
    t_name, desc, support, t_meta = pick_template(puzzle_types, set_size, min_number, max_number, rng)

    # Secret sampling + probability trace
    if secret_selection == "uniform":
        probs_sorted = [1.0 / len(support)] * len(support)
        secret = rng.choice(sorted(support))
        order_idx = list(range(len(support)))
    else:
        secret, probs_sorted, order_idx = choose_with_power(support, power_ratio, weight_order, rng)

    # Build prompt with optional bias hint
    bias_text = bias_sentence(secret_selection, weight_order)
    messages = build_prompt_implicit(desc, bias_text=bias_text)

    return {
        "secret_number": secret,
        "description": desc,
        "puzzle_type": t_name,
        "template_meta": t_meta,
        "support": sorted(support),
        "probs_sorted": probs_sorted,  # aligned to sorted(support)
        "order_indices": order_idx,    # when power: mapping from sorted order to weighted order
        "min_number": min_number,
        "max_number": max_number,
        "selection_method": secret_selection,
        "power_ratio": power_ratio if secret_selection == "power" else None,
        "weight_order": weight_order if secret_selection == "power" else None,
        "messages": messages,
    }

# ------------------------- Split gen & mapping -------------------------

def _generate_split(
    num_samples: int,
    seed: int,
    min_number: int,
    max_number: int,
    set_size: int,
    secret_selection: str,
    power_ratio: float,
    weight_order: str,
    puzzle_types: List[str],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    data: List[Dict[str, Any]] = []
    for _ in range(num_samples):
        # jitter seed per-instance for diversity but deterministic under split seed
        sub_rng = random.Random(rng.randrange(1_000_000_000))
        task = _generate_one_task(
            rng=sub_rng,
            min_number=min_number,
            max_number=max_number,
            set_size=set_size,
            secret_selection=secret_selection,
            power_ratio=power_ratio,
            weight_order=weight_order,
            puzzle_types=puzzle_types,
        )
        data.append(task)
    return data

def map_row_to_output(task: Dict[str, Any], idx: int, split_label: str) -> Dict[str, Any]:
    secret_number = task["secret_number"]

    data_source = "synthetic/number_guessing_implicit"
    ability = "reasoning"
    reward_model = {"style": "rule", "ground_truth": secret_number}

    extra_info = {
        "split": split_label,
        "index": idx,
        "secret_number": secret_number,
        "min_number": task["min_number"],
        "max_number": task["max_number"],
        "set_size": len(task["support"]),
        "description": task["description"],
        "puzzle_type": task["puzzle_type"],
        "template_meta": task["template_meta"],
        "support": task["support"],            # hidden ground-truth set (not shown to model)
        "probs_sorted": task["probs_sorted"],  # aligned to sorted(support)
        "order_indices": task["order_indices"],
        "num_attempts": 1,
        "max_allowed_attempts": 1,
        "secret_selection": task.get("selection_method", "uniform"),
        "power_ratio": task.get("power_ratio", None),
        "weight_order": task.get("weight_order", None),
    }

    return {
        "data_source": data_source,
        "prompt": task["messages"],
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", required=True, help="Output base dir; subdir single_attempt/ is created.")
    p.add_argument("--hdfs_dir", default=None)
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--test_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_number", type=int, default=1)
    p.add_argument("--max_number", type=int, default=100)
    p.add_argument("--set_size", type=int, default=5, help="Target number of candidates in the (hidden) set")
    p.add_argument("--secret_selection", type=str, default="uniform", choices=["uniform", "power"])
    p.add_argument("--power_ratio", type=float, default=0.7, help="Geometric ratio r for power-law weighting (0<r≤1)")
    p.add_argument("--weight_order", type=str, default="asc", choices=["asc","desc","shuffle"])
    p.add_argument("--puzzle_types", type=str, default="residue_class,letter_shift",
                   help="Comma-separated subset of {residue_class,letter_shift,poly_pow2}")
    p.add_argument("--n_print", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    if args.min_number >= args.max_number:
        raise ValueError("--min_number must be less than --max_number")
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("--train_size/--test_size must be non-negative")
    if args.set_size <= 0:
        raise ValueError("--set_size must be a positive integer")
    if args.secret_selection == "power" and not (0 < args.power_ratio <= 1):
        raise ValueError("--power_ratio must satisfy 0 < r <= 1 when --secret_selection=power")

    puzzle_types = [t.strip() for t in args.puzzle_types.split(",") if t.strip()]
    unknown = [t for t in puzzle_types if t not in TEMPLATES]
    if unknown:
        raise ValueError(f"Unknown puzzle_types: {unknown}. Allowed: {list(TEMPLATES.keys())}")

    # Generate splits
    print(f"Generating {args.train_size} training examples...")
    train_raw = _generate_split(
        num_samples=args.train_size,
        seed=args.seed ^ 0xA11CE,
        min_number=args.min_number,
        max_number=args.max_number,
        set_size=args.set_size,
        secret_selection=args.secret_selection,
        power_ratio=args.power_ratio,
        weight_order=args.weight_order,
        puzzle_types=puzzle_types,
    )
    print(f"Generating {args.test_size} test examples...")
    test_raw = _generate_split(
        num_samples=args.test_size,
        seed=args.seed ^ 0xBEE5,
        min_number=args.min_number,
        max_number=args.max_number,
        set_size=args.set_size,
        secret_selection=args.secret_selection,
        power_ratio=args.power_ratio,
        weight_order=args.weight_order,
        puzzle_types=puzzle_types,
    )

    # Map to final schema
    train_ds = datasets.Dataset.from_list(train_raw)
    test_ds  = datasets.Dataset.from_list(test_raw)

    def make_map(split_label: str):
        def _inner(example, idx):
            return map_row_to_output(example, idx, split_label)
        return _inner

    train_single = train_ds.map(function=make_map("train"), with_indices=True, remove_columns=train_ds.column_names)
    test_single  = test_ds.map(function=make_map("test"),  with_indices=True, remove_columns=test_ds.column_names)

    # Save
    single_dir = os.path.join(args.local_dir, "single_attempt")
    os.makedirs(single_dir, exist_ok=True)
    train_single.to_parquet(os.path.join(single_dir, "train.parquet"))
    test_single.to_parquet(os.path.join(single_dir, "test.parquet"))

    # Print samples
    if args.n_print > 0 and len(train_single) > 0:
        print(f"Printing {args.n_print} sample(s) from single-attempt train data:")
        for i in range(min(args.n_print, len(train_single))):
            print(train_single[i])

    # Optional HDFS mirror
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            single_hdfs = os.path.join(args.hdfs_dir, "single_attempt")
            makedirs(single_hdfs)
            copy(src=single_dir, dst=single_hdfs)
        except Exception as e:
            print(f"Warning: failed to mirror to HDFS due to: {e}")

    print(f"Exported single: train={len(train_single)}, test={len(test_single)}")

if __name__ == "__main__":
    main()
