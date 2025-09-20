"""
Verifier for number guessing task.
Extracts integer answers from model generation and checks if they match the target number.
"""
import re
import time
from typing import Any, Optional


def _extract_answer_block(text: str) -> Optional[str]:
    """
    Extract a single answer block as content.
    Returns None if there are 0 or more than 1 answer blocks.
    """
    matches = list(re.finditer(r"<answer>[\s\S]*?</answer>", text))
    if len(matches) != 1:
        return None

    match = matches[0].group(0)[len("<answer>") : -len("</answer>")]
    # breakpoint()
    remaining_text = text[matches[0].end():].strip()
    if remaining_text:
        return None
    return match


def number_guessing_compute_score(
    *,
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float | dict:
    """
    Reward for Number Guessing task:
      - Parses either a single <answer> block.
      - Each answer should contain an integer.
      - Score is based on exact match with the secret number.
        Best answer is used (1.0 for exact match, else 0.0).
      - Adds a formatting bonus (+0.1) when a valid integer is parsed.
    Returns a dict with details.
    """
    t0 = time.perf_counter()
    
    # Convert secret_number to integer
    target_number = int(ground_truth)
    
    if target_number is None:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "exact_match": 0,
            "attempts": 0,
            "pred": "",
            "ground_truth": "",
            "reason": "missing_or_invalid_target_number",
        }
    
    # 1) Single-answer flow
    pred_answer = _extract_answer_block(solution_str)
    
    try:
        pred_answer = int(pred_answer)
    except Exception:
        pred_answer = None
    
    if not pred_answer:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "exact_match": 0,
            "pred": str(pred_answer),
            "ground_truth": str(target_number),
            "reason": "no_answer_blocks_found",
        }

    exact = 1 if pred_answer == target_number else 0

    t1 = time.perf_counter()
    format_bonus = 0.1  # award for correct output format (parseable integer)
    return {
        "score": float(exact) + format_bonus,
        "is_correct": exact,
        "format_score": format_bonus,
        "time": t1 - t0,
        "exact_match": int(exact),
        "pred": str(pred_answer),
        "ground_truth": str(target_number),
        "reason": None,
    }
