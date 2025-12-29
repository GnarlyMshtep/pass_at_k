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


def _extract_nums_from_expr(expr: str) -> list[float]:
    """Extract numbers from an arithmetic expression."""
    nums = []
    try:
        node = ast.parse(expr, mode="eval")
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Constant) and isinstance(subnode.value, (int, float)):
                nums.append(float(subnode.value))
            elif isinstance(subnode, ast.Num):  # for Python < 3.8 compatibility
                nums.append(float(subnode.n))
        return nums
    except Exception as e:
        return []

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_arithmetic(expr: str) -> float:
    """Evaluate an arithmetic expression with +, -, *, / and parentheses only."""
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left = _eval(n.left)
            right = _eval(n.right)
            return _OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            operand = _eval(n.operand)
            return _OPS[type(n.op)](operand)
        if isinstance(n, ast.Num):  # py<3.8
            return float(n.n)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Tuple):  # disallow tuples explicitly
            raise ValueError("Tuples not allowed")
        raise ValueError(f"Unsupported expression: {type(n).__name__}")

    return float(_eval(node))


def countdown_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for Countdown (single or multi-attempt):
      - Parses multiple <attempt-i>/<attempt> expressions if present; otherwise falls back to single <answer>.
      - Each attempt must be a single arithmetic expression using each number exactly once.
      - Score is based on exact match to the target; best attempt is used (1.0 for correct, else 0.0).
      - Adds a formatting bonus (+0.2) when at least one valid attempt/expression is parsed and evaluated.
      - Respects extra_info.max_allowed_attempts if provided.
      - Returns a dict similar to taller_puzzles_verifier.
    """
    t0 = time.perf_counter()

    # Accept both int and float for targets
    try:
        target_val = float(ground_truth)
    except Exception:
        target_val = None

    nums = []
    try:
        if extra_info and isinstance(extra_info, dict):
            nums = [float(x) for x in extra_info.get("nums", [])]
    except Exception:
        nums = []

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
                    "reason": "wrong_format",
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
            "reason": None,
        }

    best_exact = 0
    best_idx = -1
    best_expr = ""
    evaluated = 0

    for idx, (_tag, content) in enumerate(blocks):
        expr = (content or "").strip()
        if not expr:
            continue
        # Basic quick rejection: if contains a comma, likely not a single expression; skip
        if "," in expr:
            continue
        # Validate numbers multiset and evaluate
        expr_nums = _extract_nums_from_expr(expr)
        if Counter(expr_nums) != Counter(nums):
            continue
        try:
            value = _safe_eval_arithmetic(expr)
        except Exception:
            continue
        evaluated += 1
        exact = 1 if abs(value - target_val) < 1e-6 else 0
        if exact > best_exact:
            best_exact = exact
            best_idx = idx
            best_expr = expr

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
        "pred": best_expr,
        "ground_truth": str(ground_truth),
        "reason": None,
    }


