import ast
import operator
import typing as _t
from collections import Counter
import re
import time

def _extract_answer_blocks(s: str) -> list[str]:
    """Extract all <answer>...</answer> blocks, returning their inner contents.

    Only properly paired tags are considered. Stray closing/opening tags are ignored.
    """
    blocks: list[str] = []
    for m in re.finditer(r"<answer>[\s\S]*?</answer>", s):
        content = m.group(0)[len("<answer>") : -len("</answer>")]
        blocks.append(content)
    return [b.strip() for b in blocks]


def _extract_attempt_blocks(s: str) -> list[tuple[str, str]]:
    """Extracts attempt blocks as (tag, content) pairs.

    Supports numbered <attempt-i>...</attempt-i> and generic <attempt>...</attempt>.
    """
    attempts: list[tuple[str, str]] = []

    # Numbered attempts: <attempt-1>...</attempt-1>
    for m in re.finditer(r"<attempt-(\d+)>[\s\S]*?</attempt-\1>", s):
        tag = f"attempt-{m.group(1)}"
        content = re.sub(r"^<attempt-\d+>|</attempt-\d+>$", "", m.group(0))
        attempts.append((tag, content))

    # Generic attempts: <attempt>...</attempt>
    for m in re.finditer(r"<attempt>[\s\S]*?</attempt>", s):
        tag = "attempt"
        content = m.group(0)[len("<attempt>") : -len("</attempt>")]
        attempts.append((tag, content))

    return attempts


def pathfinder_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for pathfinder (single or multi-attempt):
      - Parses multiple <attempt-i>/<attempt> paths if present; otherwise falls back to single <answer>.
      - Each attempt must be a path in the format of a->c->de->cf->b. 
      - Paths must start from a, ends at b and all edges in the path must be in the graph edges. 
      - best attempt is used (1.0 for correct, else 0.0).
      - Adds a formatting bonus (+0.2) when at least one valid attempt/path is parsed and evaluated.
      - Respects extra_info.max_allowed_attempts if provided.
      - Returns a dict similar to taller_puzzles_verifier.
    """
    t0 = time.perf_counter()

    edges = []
    nodes = set()
    try:
        if extra_info and isinstance(extra_info, dict):
            edges = extra_info.get("edges", [])
            for v,u in edges:
                nodes.add(v)
                nodes.add(u)
    except Exception:
        edges = []

    # Determine expected attempts (prefer max_allowed_attempts, fallback to num_attempts, default 1)
    expected_attempts = extra_info["max_allowed_attempts"]

    # Gather attempts or fallback to <answer>
    attempt_blocks = _extract_attempt_blocks(solution_str)
    blocks: list[tuple[str, str]] = []
    if attempt_blocks:
        # Require at least expected_attempts attempts to be present
        if len(attempt_blocks) < expected_attempts:
            t1 = time.perf_counter()
            return {
                "score": 0.0,
                "is_correct": 0,
                "format_score": 0.0,
                "time": t1 - t0,
                "exact_match": 0,
                "attempts": int(len(attempt_blocks)),
                "best_attempt_index": None,
                "pred": "",
                "ground_truth": str(ground_truth),
                "reason": "insufficient_attempts",
            }
        # Trim to exactly expected_attempts
        blocks = attempt_blocks[:expected_attempts]
    else:
        answer_blocks = _extract_answer_blocks(solution_str)
        if answer_blocks:
            # Require the number of <answer> blocks to match expected attempts exactly
            if len(answer_blocks) != expected_attempts:
                t1 = time.perf_counter()
                return {
                    "score": 0.0,
                    "is_correct": 0,
                    "format_score": 0.0,
                    "time": t1 - t0,
                    "exact_match": 0,
                    "attempts": 0,
                    "best_attempt_index": None,
                    "pred": "",
                    "ground_truth": str(ground_truth),
                    "reason": f"wrong_format, expected {expected_attempts}, get {len(answer_blocks)}",
                }
            blocks = [("answer", b) for b in answer_blocks]

    if not blocks:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "exact_match": 0,
            "attempts": 0,
            "best_attempt_index": None,
            "pred": "",
            "ground_truth": str(ground_truth),
            "reason": "No blocks was found",
        }

    best_exact = 0
    best_idx = -1
    best_path = ""
    evaluated = 0
    reason = ""
    for idx, (_tag, content) in enumerate(blocks):
        path = (content or "").strip().split("->")
        valid_path=True
        for node in path:
            if node not in nodes:
                valid_path=False
                reason += f"{idx}: {node} doesn't exist on {nodes}\n"
                break
        if not valid_path:
            continue # the path containes a node not in the nodes list.
        if len(path)<2: 
            reason += f"{idx}: len of path is one/zero \n"
            continue # the path should have at least two nodes
        if path[0]!='a':
            reason += f"{idx}: path doesn't start from a\n"
            continue # the path should start from a
        if path[-1]!='b':
            reason += f"{idx}: path doesn't end at  b\n"
            continue # the path should end to b
            
        evaluated += 1

        exact = 1 
        for checking_edge_index in range(len(path)-1):
            edge = (path[checking_edge_index], path[checking_edge_index+1])
            if edge not in edges:
                exact=0
                reason += f"{idx}: {path[checking_edge_index]}->{path[checking_edge_index+1]} doesn't exist at edges.\n"
                break

        if exact >= best_exact:
            best_exact = exact
            best_idx = idx
            best_path = path

    t1 = time.perf_counter()
    format_bonus = 0.2 if (evaluated > 0 and len(blocks) == expected_attempts) else 0.0
    return {
        "score": float(best_exact) + format_bonus,
        "is_correct": int(best_exact),
        "format_score": format_bonus,
        "time": t1 - t0,
        "exact_match": int(best_exact),
        "attempts": int(evaluated),
        "best_attempt_index": int(best_idx) if best_idx is not None else None,
        "pred": best_path,
        "ground_truth": str(ground_truth),
        "reason": reason,
    }


