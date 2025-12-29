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


def _parse_guess_list(text: str) -> Optional[List[str]]:
    """
    Parse a comma-separated list of single-letter person IDs (A..Z).
    Returns a list of unique uppercase letters in stable order if valid; else None.
    """
    inner = (text or "").strip()
    if not inner:
        return None
    # Remove surrounding whitespace and normalize separators
    parts = [p.strip() for p in inner.split(",")]
    # Filter out empty parts
    parts = [p for p in parts if p]
    if not parts:
        return None

    guesses: List[str] = []
    seen = set()
    for token in parts:
        # Accept single alphabetical letter; normalize to uppercase
        if len(token) != 1 or not token.isalpha():
            return None
        t = token.upper()
        if t not in seen:
            guesses.append(t)
            seen.add(t)
    return guesses


def _as_set_from_ground_truth(gt: Any) -> Optional[set]:
    try:
        if isinstance(gt, str):
            tokens = [t.strip().upper() for t in gt.split(",") if t.strip()]
            return set(tokens)
        if isinstance(gt, list):
            return set(str(x).strip().upper() for x in gt)
    except Exception:
        return None
    return None


def taller_puzzles_compute_score(
    *,
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> float | dict:
    """
    Reward for Taller Puzzles (works for single and multi attempt):
      - Parses either a single <answer> list or multiple <attempt-i>/<attempt> lists.
      - Each attempt must be a comma-separated list of single-letter IDs (A..Z).
      - Score is based on exact set match accuracy between prediction and gold.
        Best attempt is used (1.0 for exact match, else 0.0).
      - Adds a formatting bonus (+0.2) when a valid list is parsed.
      - Respects extra_info.max_allowed_attempts if provided.
    Returns a dict with details.
    """
    t0 = time.perf_counter()

    gold_set = _as_set_from_ground_truth(ground_truth) or set()
    if not gold_set:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "time": t1 - t0,
            "exact_match": 0,
            "attempts": 0,
            "best_attempt_index": None,
            "pred": "",
            "ground_truth": "",
            "reason": "missing_or_invalid_ground_truth",
        }

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
            "exact_match": 0,
            "attempts": 0,
            "best_attempt_index": None,
            "pred": "",
            "ground_truth": ",".join(sorted(list(gold_set))),
            "reason": None,
        }

    # Respect max_allowed_attempts from extra_info if provided
    max_allowed_attempts = None
    if extra_info and isinstance(extra_info, dict):
        max_allowed_attempts = extra_info.get("max_allowed_attempts", None)

    best_exact = 0
    best_idx = -1
    best_pred_list: List[str] = []
    evaluated = 0

    blocks = attempt_blocks
    if isinstance(max_allowed_attempts, int) and max_allowed_attempts > 0:
        blocks = blocks[:max_allowed_attempts]

    for idx, (_tag, content) in enumerate(blocks):
        guesses = _parse_guess_list(content)
        if guesses is None:
            continue
        evaluated += 1
        pred_set = set(guesses)
        exact = 1 if pred_set == gold_set else 0
        if exact >= best_exact:
            best_exact = exact
            best_idx = idx
            best_pred_list = sorted(list(pred_set))

    if evaluated == 0:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "time": t1 - t0,
            "exact_match": 0,
            "attempts": len(blocks),
            "best_attempt_index": None,
            "pred": "",
            "ground_truth": ",".join(sorted(list(gold_set))),
            "reason": "no_valid_attempts",
        }

    t1 = time.perf_counter()
    format_bonus = 0.2  # award for correct output format (parseable list)
    return {
        "score": float(best_exact) + format_bonus,
        "time": t1 - t0,
        "exact_match": int(best_exact),
        "attempts": int(evaluated),
        "best_attempt_index": int(best_idx) if best_idx is not None else None,
        "pred": ",".join(best_pred_list),
        "ground_truth": ",".join(sorted(list(gold_set))),
        "reason": None,
    }


