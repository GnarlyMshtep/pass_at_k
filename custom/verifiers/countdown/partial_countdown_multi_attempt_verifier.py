import re
import typing as _t
from collections import Counter
import time

# Reuse the safe parsing and scoring helpers from the single-attempt verifier
from custom.partial_countdown_verifier import (
    _safe_eval_arithmetic,
    _extract_nums_from_expr,
    _can_merge_to_target,
    partial_countdown_compute_score as _single_attempt_score,
)


def _extract_attempts(s: str, max_attempts: int = 8) -> list[str]:
    """Extract attempt blocks <attempt-i>...</attempt-i> for i in 1..max_attempts."""
    attempts: list[str] = []
    if not isinstance(s, str):
        return attempts
    for i in range(1, max_attempts + 1):
        pattern = rf"<attempt-{i}>([\s\S]*?)</attempt-{i}>"
        m = re.search(pattern, s)
        if m:
            attempts.append(m.group(1).strip())
    return attempts


def _score_submission(
    submission: str,
    *,
    target: float,
    numbers: list[float],
    original_actions: int,
    tol: float = 1e-6,
) -> tuple[float, int]:
    """
    Score one attempt string using search-action accounting:
      - Single full expression equals target -> reward = 1 + 0.2
      - Comma-separated partials -> evaluate parts; if mergeable to target using +,-,*,/,
        reward = (original_actions - taken_actions)/original_actions + 0.2; otherwise 0.2
      - Mismatch in numbers or invalid expr -> 0
    """
    if not isinstance(submission, str) or submission.strip() == "":
        return 0.0, 0

    # Detect whether it's a list of partials (comma-separated) or a single expression
    parts = [p.strip() for p in submission.split(",")] if "," in submission else [submission.strip()]
    if any(p == "" for p in parts):
        return 0.0, 0

    # Validate multiset of numbers matches exactly the provided nums (including duplicates)
    submitted_nums: list[float] = []
    for p in parts:
        submitted_nums.extend(_extract_nums_from_expr(p))
    if Counter(submitted_nums) != Counter([float(x) for x in numbers]):
        return 0.0, 0

    # Single expression case
    if len(parts) == 1:
        expr = parts[0]
        try:
            value = _safe_eval_arithmetic(expr)
        except Exception:
            return 0.0, 0
        correctness_score = 1.0 if abs(value - float(target)) < tol else 0.0
        return float(correctness_score) + 0.2, 0 # correctness + formatting rewards

    # Multiple partials case: reward based on remaining actions to reach the target
    values: list[float] = []
    for p in parts:
        try:
            values.append(_safe_eval_arithmetic(p))
        except Exception:
            return 0.0, 0

    can_merge, taken_actions = _can_merge_to_target(values, float(target), tol=tol)
    if not can_merge:
        return 0.2, int(taken_actions)

    reward = (original_actions - taken_actions) / original_actions
    return float(reward) + 0.2, int(taken_actions)


def partial_countdown_multi_attempt_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for Partial-Countdown with multi-attempt outputs:
      - Parse <attempt-i> ... </attempt-i> blocks (i = 1..8)
      - Score each attempt using the same rules as single-attempt Partial-Countdown
      - Return the maximum score across attempts
      - If no attempts are found, fall back to the single-attempt scorer which uses <answer>...
    """
    _t0 = time.perf_counter()
    numbers = [float(x) for x in (extra_info or {}).get("nums", [])]
    attempts = _extract_attempts(solution_str, max_attempts=int((extra_info or {}).get("num_attempts", 8)))

    if len(attempts) == 0:
        # Fallback: support answers that might still use <answer>...</answer>
        res = _single_attempt_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(res, dict):
            return res
        return {
            "score": float(res),
            "time": time.perf_counter() - _t0,
            "origianl_actions": 0,
            "taken_actions": 0,
        }

    target = float(ground_truth)

    # Compute the baseline minimal number of actions for the original task once
    task_solvable, original_actions = _can_merge_to_target(numbers, target, tol=1e-6)
    assert task_solvable, f"Task is not solvable with the given numbers: {numbers} -> {ground_truth}"

    best_score: float = 0.0
    best_taken_actions: int = 0
    for att in attempts:
        score, taken_actions = _score_submission(
            submission=att,
            target=target,
            numbers=numbers,
            original_actions=original_actions,
        )
        if score > best_score:
            best_score = float(score)
            best_taken_actions = int(taken_actions)

    return {
        "score": float(best_score) if best_score is not None else 0.0,
        "time": time.perf_counter() - _t0,
        "origianl_actions": int(original_actions),
        "taken_actions": int(best_taken_actions),
    }


