import typing as _t


def extract_answer_math(s: str) -> str:
    """Extract the content inside <answer>...</answer> if present.

    Falls back to returning the original string trimmed.
    """
    if ("<answer>" in s and "</answer>" in s):
        s = s.split("<answer>")[-1].split("</answer>")[0].strip()
    return s.strip()


def math_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """Adapter for NaiveRewardManager.compute_score using Math-Verify.

    - Parses the model output for the <answer> tag content.
    - Extracts the ground-truth answer string from `ground_truth` or `extra_info`.
    - Delegates to `verl.utils.reward_score.math_verify.compute_score`.
    Returns a dict with "score" (float, typically 0.0 or 1.0) and "is_correct" (float, 0.0 or 1.0)
    """
    from verl.utils.reward_score.math_verify import (
        compute_score as _math_verify_compute_score,
    )

    pred: str = extract_answer_math(solution_str)

    gold_text: _t.Optional[str] = None
    if isinstance(ground_truth, str):
        gold_text = ground_truth
    elif isinstance(ground_truth, dict):
        for key in ("answer", "label", "ground_truth", "target", "solution"):
            value = ground_truth.get(key)
            if isinstance(value, str) and value.strip():
                gold_text = value
                break

    if gold_text is None and extra_info and isinstance(extra_info, dict):
        for key in ("answer", "label", "ground_truth", "target"):
            maybe_gold = extra_info.get(key)
            if isinstance(maybe_gold, str) and maybe_gold.strip():
                gold_text = maybe_gold
                break

    if not isinstance(gold_text, str) or not gold_text.strip():
        return {"score": 0.0, "is_correct": 0.0}

    try:
        score = _math_verify_compute_score(pred, gold_text)
    except Exception:
        return {"score": 0.0, "is_correct": 0.0}

    try:
        score_float = float(score)
        # Return dict with both score and is_correct for filter_groups_metric
        return {"score": score_float, "is_correct": score_float}
    except Exception:
        return {"score": 0.0, "is_correct": 0.0}


