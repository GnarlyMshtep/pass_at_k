import re
import time
from typing import Any, List, Optional, Tuple


def _extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    if start_tag in text and end_tag in text:
        try:
            return text.split(start_tag, 1)[1].split(end_tag, 1)[0]
        except Exception:
            return None
    return None


def _extract_answer_block(s: str) -> Optional[str]:
    return _extract_between(s, "<answer>", "</answer>")


def _extract_attempt_blocks(s: str) -> List[Tuple[str, str]]:
    """
    Extracts attempt blocks as (tag, content) pairs.
    Supports both <attempt>...</attempt> and numbered <attempt-i>...</attempt-i>.
    """
    attempts: List[Tuple[str, str]] = []

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


def _unwrap_code_fence(text: str) -> str:
    """
    If the text contains a fenced code block, return only the content inside the
    first such fence. Otherwise return the original text.
    """
    fence_match = re.search(r"```[\w\-]*\n([\s\S]*?)\n```", text)
    if fence_match:
        return fence_match.group(1)
    return text


def _parse_grid(text: str, expected_size: int) -> Optional[List[List[int]]]:
    """
    Parse a grid of integers from text. Accepts either fenced code or plain lines.
    Expects exactly expected_size rows and columns. Returns None if invalid.
    """
    inner = _unwrap_code_fence(text).strip()
    lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
    if not lines:
        return None

    grid: List[List[int]] = []
    for ln in lines:
        # Extract integers on the line
        nums = re.findall(r"-?\d+", ln)
        if not nums:
            # Ignore decorative lines, but they shouldn't appear in Sudoku outputs
            continue
        row = [int(x) for x in nums]
        grid.append(row)

    if len(grid) != expected_size:
        return None
    if any(len(row) != expected_size for row in grid):
        return None
    return grid


def _score_partial_fill(
    *,
    submission: List[List[int]],
    question: List[List[int]],
    solution: List[List[int]],
) -> Tuple[int, int, int]:
    """
    Returns (correct_filled, total_fillable) where total_fillable is the number of
    zeros in the question. Only cells where question==0 are considered. A filled
    cell is counted as correct if submission[i][j] == solution[i][j] and not zero.
    """
    n = len(solution)
    correct = 0
    incorrect = 0
    total = 0
    for i in range(n):
        for j in range(n):
            if question[i][j] == 0:
                total += 1
                val = submission[i][j]
                if val != 0 and val == solution[i][j]:
                    correct += 1
                else:
                    incorrect += 1
    return correct, incorrect, total


def partial_sudoku_compute_score(
    *,
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float | dict:
    """
    Reward for Partial Sudoku (size-agnostic):
      - Parse either a single <answer> grid or multiple <attempt-i> grids.
      - Each grid is a NxN array of integers with 0 as empty cells.
      - Only cells that were 0 in the question are eligible for credit.
      - Score is the fraction of correctly filled eligible cells. If there are no
        empty cells in the question, score is 1.0.
    Returns a dict with detailed fields.
    """
    t0 = time.perf_counter()

    # Validate inputs
    solution: Optional[List[List[int]]] = None
    try:
        if isinstance(ground_truth, list) and ground_truth and isinstance(ground_truth[0], list):
            solution = [[int(v) for v in row] for row in ground_truth]
    except Exception:
        solution = None

    if solution is None:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "time": t1 - t0,
            "board_size": None,
            "correct_filled": None,
            "incorrect_filled": None,
            "total_empty": None,
            "attempts": None,
            "best_attempt_index": None,
            "reason": "missing_or_invalid_ground_truth",
        }

    n = len(solution)
    board_size = n

    question = None
    if extra_info and isinstance(extra_info.get("question"), list):
        try:
            question = [[int(v) for v in row] for row in extra_info["question"]]
        except Exception:
            question = None

    if question is None or len(question) != n or any(len(r) != n for r in question):
        # If the question is missing, fallback to treating all cells as fillable
        question = [[0 for _ in range(n)] for _ in range(n)]

    # 1) Single-answer flow
    answer_block = _extract_answer_block(solution_str)
    attempt_blocks: List[Tuple[str, str]] = []
    if answer_block is not None:
        attempt_blocks = [("answer", answer_block)]
    else:
        # 2) Multi-attempt flow
        attempt_blocks = _extract_attempt_blocks(solution_str)

    if not attempt_blocks:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "time": t1 - t0,
            "board_size": board_size,
            "correct_filled": 0,
            "incorrect_filled": 0,
            "total_empty": sum(1 for i in range(n) for j in range(n) if question[i][j] == 0),
            "attempts": 0,
            "best_attempt_index": None,
            "reason": None,
        }

    best_fraction = 0.0
    best_correct = 0
    best_total = 0
    best_incorrect = 0
    best_idx = -1
    evaluated = 0

    # Respect max_allowed_attempts from extra_info if provided
    max_allowed_attempts = extra_info.get("max_allowed_attempts", None)

    for idx, (_tag, content) in enumerate(attempt_blocks[:max_allowed_attempts]):
        grid = _parse_grid(content, n)
        if grid is None:
            continue
        evaluated += 1
        correct, incorrect, total = _score_partial_fill(submission=grid, question=question, solution=solution)
        fraction = (correct / total) if (incorrect == 0) else 0.0
        if fraction >= best_fraction:
            best_fraction = fraction
            best_correct = correct
            best_total = total
            best_incorrect = incorrect
            best_idx = idx

    # If none parsed successfully, zero score
    if evaluated == 0:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "time": t1 - t0,
            "board_size": board_size,
            "correct_filled": 0,
            "incorrect_filled": 0,
            "total_empty": sum(1 for i in range(n) for j in range(n) if question[i][j] == 0),
            "attempts": len(attempt_blocks),
            "best_attempt_index": None,
            "reason": "no_valid_grids",
        }

    t1 = time.perf_counter()
    format_bonus = 0.2  # award for correct output format (parseable grid)
    return {
        "score": float(best_fraction) + format_bonus,
        "time": t1 - t0,
        "board_size": board_size,
        "correct_filled": int(best_correct),
        "incorrect_filled": int(best_incorrect),
        "total_empty": int(best_total),
        "attempts": int(evaluated),
        "best_attempt_index": int(best_idx) if best_idx is not None else None,
        "reason": None,
    }


