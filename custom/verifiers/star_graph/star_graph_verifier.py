import time
import typing as _t


def _parse_edges(edge_str: str) -> list[tuple[int, int]]:
    """Parse space-separated "u,v" pairs into undirected edges."""
    edges: list[tuple[int, int]] = []
    s = edge_str.strip()
    if not s:
        return edges
    for token in s.split():
        if "," not in token:
            continue
        try:
            a, b = token.split(",", 1)
            u = int(a.strip())
            v = int(b.strip())
            edges.append((u, v))
        except Exception:
            continue
    return edges


def _extract_path_answer(s: str) -> str | None:
    # Expect a single path as comma-separated integers inside <answer> ... </answer>
    if ("<answer>" in s and "</answer>" in s):
        return s.split("<answer>")[-1].split("</answer>")[0].strip()
    return None


def _parse_path_csv(csv_str: str) -> list[int] | None:
    try:
        parts = [p.strip() for p in csv_str.split(",") if p.strip() != ""]
        if not parts:
            return None
        return [int(p) for p in parts]
    except Exception:
        return None


def _is_valid_path(path: list[int], edges: set[tuple[int, int]], source: int, destination: int) -> bool:
    if len(path) < 2:
        return False
    if path[0] != source or path[-1] != destination:
        return False
    # Consecutive nodes must be connected by an undirected edge
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) not in edges and (v, u) not in edges:
            return False
    return True


def star_graph_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for Star-Graph pathfinding (single-attempt):
      - Expects exactly one path inside <answer>...</answer> as comma-separated integers
      - Valid if forms a contiguous path from source to destination using given edges
      - Score 1.0 for any valid path; otherwise 0.0
      - Adds formatting bonus (+0.2) if a path list is correctly parsed and validated (valid or not)
    """
    t0 = time.perf_counter()

    # Load metadata from extra_info (dataset provides these fields)
    source = None
    destination = None
    edges_list: list[list[int]] | None = None
    if isinstance(extra_info, dict):
        try:
            source = int(extra_info.get("source")) if extra_info.get("source") is not None else None
            destination = int(extra_info.get("destination")) if extra_info.get("destination") is not None else None
            edges_list = extra_info.get("edges", None)
        except Exception:
            pass

    if edges_list is None or source is None or destination is None:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "reason": "missing_graph_metadata",
            "pred": "",
            "ground_truth": str(ground_truth),
        }

    edge_set = set((int(u), int(v)) for u, v in edges_list)

    # Extract the answer content
    answer_block = _extract_path_answer(solution_str)
    if answer_block is None:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "reason": "missing_answer_block",
            "pred": "",
            "ground_truth": str(ground_truth),
        }

    # Parse the path
    parsed_path = _parse_path_csv(answer_block)
    if parsed_path is None:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "reason": "invalid_path_format",
            "pred": "",
            "ground_truth": str(ground_truth),
        }

    # Validate the path against graph
    is_valid = _is_valid_path(parsed_path, edge_set, source, destination)

    t1 = time.perf_counter()
    format_bonus = 0.2  # parsed a path successfully
    base = 1.0 if is_valid else 0.0
    return {
        "score": float(base) + format_bonus,
        "is_correct": int(is_valid),
        "format_score": format_bonus,
        "time": t1 - t0,
        "reason": None,
        "pred": ",".join(str(x) for x in parsed_path),
        "ground_truth": str(ground_truth),
        "exact_match": int(is_valid),
    }


