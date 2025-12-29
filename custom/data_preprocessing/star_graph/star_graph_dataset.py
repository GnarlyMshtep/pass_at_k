"""
Generate star-graph pathfinding dataset (single-attempt) using the star graph generator.

Functionality:
- Accept --train_size/--test_size to control number of generated samples per split
- Accept --deg, --path_len, --num_nodes, --reverse for graph parameters
- Produce prompts asking for any valid path from source to destination

Output schema columns:
- data_source (string)
- prompt (list[dict])
- ability (string)
- reward_model (dict)
- extra_info (dict)

This script produces a dataset under --local_dir:
  - train.parquet and test.parquet

Optional: --hdfs_dir to mirror the directory to HDFS.
"""

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import datasets


GRAPH_PROBLEM_TEMPLATE = (
    "Given a bi-directional graph in the form of space separated edges, output a path from source node "
    "to the destination node in the form of comma separated integers.\n"
    "For this question the graph is {graph}\n"
    "The source node is {source}\n"
    "The destination node is {destination}\n"
)


def _choose_weighted(items: List[int], *, selection: str, ratio: float, order: str) -> int:
    """Choose one item from items according to selection strategy.

    - selection: 'uniform' or 'power'
    - ratio: geometric ratio r in (0, 1], used only for 'power'
    - order: 'asc'|'desc'|'shuffle' over the sorted items to align weights
    """
    if not items:
        raise ValueError("_choose_weighted received empty items")
    if selection == "uniform":
        return random.choice(items)

    sorted_items = sorted(items)
    n = len(sorted_items)
    if order == "asc":
        ordered = sorted_items
    elif order == "desc":
        ordered = list(reversed(sorted_items))
    else:
        ordered = sorted_items[:]
        random.shuffle(ordered)

    weights = [ratio ** (i + 1) for i in range(n)]
    # Map back to original indices with same order probabilities
    chosen = random.choices(ordered, weights=weights, k=1)[0]
    return int(chosen)


def _generate_star_graph(
    *,
    deg_source: int,
    path_len: int,
    num_nodes: int,
    goal_selection: str,
    power_ratio: float,
    weight_order: str,
) -> tuple[List[int], List[List[int]], int, int]:
    """Generate a star-branch graph with a guaranteed source→goal path.

    Returns (path_nodes, edge_list, source, goal) where path_nodes starts at source and ends at goal.
    """
    if num_nodes < max(3, path_len):
        raise ValueError("num_nodes must be >= max(3, path_len)")
    if deg_source < 1:
        raise ValueError("deg_source must be >= 1")

    # Sample source
    source = int(np.random.randint(0, num_nodes, 1)[0])

    # Choose distinct first-layer neighbors of source
    candidates = [n for n in range(num_nodes) if n != source]
    if len(candidates) < deg_source:
        raise ValueError("Not enough nodes to satisfy deg_source first-layer neighbors")
    first_layer = random.sample(candidates, k=deg_source)

    # Choose which branch will contain the goal, using weighted selection
    chosen_neighbor = _choose_weighted(
        first_layer, selection=goal_selection, ratio=power_ratio, order=weight_order
    )

    # Build the primary path nodes list
    used: set[int] = {source, chosen_neighbor}
    path_nodes: List[int] = [source, chosen_neighbor]

    # Extend path to desired length (nodes = path_len)
    # We need (path_len - 2) additional nodes beyond source and chosen_neighbor
    remaining_needed = max(0, path_len - 2)
    available_nodes = [n for n in range(num_nodes) if n not in used]
    if len(available_nodes) < remaining_needed:
        raise ValueError("Insufficient nodes to create requested path length")
    middle = random.sample(available_nodes, k=remaining_needed)
    path_nodes.extend(middle)
    used.update(middle)
    goal = int(path_nodes[-1])

    # Build edges: connect the primary path
    edge_list: List[List[int]] = []
    for i in range(len(path_nodes) - 1):
        edge_list.append([int(path_nodes[i]), int(path_nodes[i + 1])])

    # Build the other (deg_source - 1) arms starting from source
    deg_nodes: set[int] = set(n for n in first_layer if n != chosen_neighbor)
    # Connect source to all first-layer neighbors (including chosen_neighbor if not already connected)
    for nbr in first_layer:
        if [source, nbr] not in edge_list and [nbr, source] not in edge_list:
            edge_list.append([int(source), int(nbr)])

    # For each non-chosen neighbor, extend a chain of length (path_len - 1) edges away from source
    chain_edges_target = max(0, path_len - 1)
    taken: set[int] = set(used) | set(first_layer)
    for nbr in first_layer:
        if nbr == chosen_neighbor:
            continue
        current = int(nbr)
        l = 1
        while l < chain_edges_target:
            # pick a fresh node not yet taken
            pool = [n for n in range(num_nodes) if n not in taken]
            if not pool:
                break
            nxt = int(random.choice(pool))
            edge_list.append([int(current), int(nxt)])
            taken.add(nxt)
            current = nxt
            l += 1

    random.shuffle(edge_list)
    return [int(n) for n in path_nodes], [[int(u), int(v)] for u, v in edge_list], int(source), int(goal)


def _edges_to_str(edges: List[List[int]]) -> str:
    # Represent as space-separated "u,v" pairs
    return " ".join(f"{int(u)},{int(v)}" for u, v in edges)


def build_single_attempt_prompt(edges: List[List[int]], source: int, destination: int) -> List[Dict[str, str]]:
    content = GRAPH_PROBLEM_TEMPLATE.format(
        graph=_edges_to_str(edges), source=source, destination=destination
    )
    return [{"content": content, "role": "user"}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, help="Local directory to save outputs")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--deg", type=int, default=2, help="Degree from source arms (>=1)")
    parser.add_argument("--path_len", type=int, default=5, help="Length of guaranteed source->destination path (>=2)")
    parser.add_argument("--num_nodes", type=int, default=50, help="Total number of nodes in the graph")
    parser.add_argument("--reverse", action="store_true", help="Reverse the labeled ground-truth path")

    # Goal branch selection (uniform / power-law), akin to number_guessing
    parser.add_argument(
        "--goal_selection",
        type=str,
        default="uniform",
        choices=["uniform", "power"],
        help="How to choose the root-branch for the destination: uniform or power-law weighting",
    )
    parser.add_argument(
        "--power_ratio",
        type=float,
        default=0.7,
        help="Geometric ratio r for power weighting when --goal_selection=power. 0 < r <= 1",
    )
    parser.add_argument(
        "--weight_order",
        type=str,
        default="asc",
        choices=["asc", "desc", "shuffle"],
        help="Order to apply weights across root neighbors: asc (smallest→largest), desc, or shuffle",
    )

    parser.add_argument("--n_print", type=int, default=1, help="Print N samples per split for sanity check")
    return parser.parse_args()


def _make_example(
    idx: int,
    split: str,
    deg: int,
    path_len: int,
    num_nodes: int,
    reverse: bool,
    goal_selection: str,
    power_ratio: float,
    weight_order: str,
) -> Dict[str, Any]:
    path, edge_list, source, goal = _generate_star_graph(
        deg_source=deg,
        path_len=path_len,
        num_nodes=num_nodes,
        goal_selection=goal_selection,
        power_ratio=power_ratio,
        weight_order=weight_order,
    )

    # Ensure the stored ground-truth path always starts at source and ends at destination
    stored_path = path if (len(path) > 0 and path[0] == source and path[-1] == goal) else list(reversed(path))

    messages = build_single_attempt_prompt(edge_list, int(source), int(goal))

    data_source = "star-graph-synthetic"
    ability = "graph_path"

    reward_model = {
        "style": "graph_path",
        "ground_truth": [int(x) for x in stored_path],  # For reference; verifier accepts any valid path
    }

    extra_info = {
        "split": split,
        "index": idx,
        "deg": int(deg),
        "path_len_param": int(path_len),
        "num_nodes": int(num_nodes),
        "num_edges": int(len(edge_list)),
        "labeled_path_length": int(len(stored_path)),
        # Required by verifier
        "source": int(source),
        "destination": int(goal),
        "edges": [[int(u), int(v)] for u, v in edge_list],
        # Attempts metadata
        "num_attempts": 1,
        "max_allowed_attempts": 1,
        # Selection metadata
        "goal_selection": goal_selection,
        "power_ratio": power_ratio if goal_selection == "power" else None,
        "weight_order": weight_order if goal_selection == "power" else None,
    }

    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _build_split(
    n: int,
    split: str,
    deg: int,
    path_len: int,
    num_nodes: int,
    reverse: bool,
    goal_selection: str,
    power_ratio: float,
    weight_order: str,
) -> datasets.Dataset:
    rows = [
        _make_example(
            idx=i,
            split=split,
            deg=deg,
            path_len=path_len,
            num_nodes=num_nodes,
            reverse=reverse,
            goal_selection=goal_selection,
            power_ratio=power_ratio,
            weight_order=weight_order,
        )
        for i in range(n)
    ]
    return datasets.Dataset.from_list(rows)


if __name__ == "__main__":
    args = parse_args()

    if args.deg < 1:
        raise ValueError("--deg must be >= 1")
    if args.path_len < 2:
        raise ValueError("--path_len must be >= 2")
    if args.num_nodes < max(3, args.path_len):
        raise ValueError("--num_nodes must be >= max(3, --path_len)")
    if args.train_size < 0 or args.test_size < 0:
        raise ValueError("--train_size/--test_size must be non-negative")
    if args.goal_selection == "power":
        if not (0 < args.power_ratio <= 1):
            raise ValueError("--power_ratio must satisfy 0 < r <= 1 when --goal_selection=power")

    # Seed for reproducibility across numpy and random
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create dataset
    print("Generating star-graph datasets...")
    train_dataset = _build_split(
        n=args.train_size,
        split="train",
        deg=args.deg,
        path_len=args.path_len,
        num_nodes=args.num_nodes,
        reverse=args.reverse,
        goal_selection=args.goal_selection,
        power_ratio=args.power_ratio,
        weight_order=args.weight_order,
    )
    test_dataset = _build_split(
        n=args.test_size,
        split="test",
        deg=args.deg,
        path_len=args.path_len,
        num_nodes=args.num_nodes,
        reverse=args.reverse,
        goal_selection=args.goal_selection,
        power_ratio=args.power_ratio,
        weight_order=args.weight_order,
    )

    # Prepare output directory
    os.makedirs(args.local_dir, exist_ok=True)

    # Save parquet files
    print("Saving parquet files...")
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    # Print samples
    if args.n_print > 0 and len(train_dataset) > 0:
        print(f"Printing {min(args.n_print, len(train_dataset))} sample(s) from train data:")
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
        f"Generated star-graph dataset with params (deg={args.deg}, path_len={args.path_len}, num_nodes={args.num_nodes}, reverse={args.reverse}). "
        f"exported: train={len(train_dataset)}, test={len(test_dataset)}"
    )

    print("Example usage:")
    print(
        "PYTHONPATH=. python custom/data_preprocessing/star_graph/star_graph_dataset.py "
        "--local_dir $HF_HOME/data/star_graph --train_size 1000 --test_size 200 --deg 2 --path_len 5 --num_nodes 50 "
        "--goal_selection power --power_ratio 0.7 --weight_order asc"
    )


