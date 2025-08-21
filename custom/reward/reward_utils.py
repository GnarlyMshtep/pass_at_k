import os
from typing import Any

import numpy as np
import torch
from math_verify import parse, verify
from numpy.typing import NDArray

REWARD_CORRECT = 1

def extra_reward_metrics(responses: list[str], prompts: list[str], ground_truths:list[str]) ->dict[str, Any]:
    # breakpoint() #M: would like to inspect what's going on
    
    # N_ROLLOUTS = os.getenv("N_ROLLOUTS", None)
    # assert N_ROLLOUTS != None, "please set env variable N_ROLLOUTS to the number of rollouts per question, i.e the number of approaches"
    # N_ROLLOUTS = int(N_ROLLOUTS)
    
    # solutions = [_get_tagged_data(response, "solution") for response in responses]
    # print(f"{prompts=} {solutions=} ")
    
    #TODO: calculate diversity score avg(set(proposed_answers) / |proposed_answer|) for proposed_answers for a question in questions
    return {}


    
    
    return {"test": 0}

def compute_statistics(float_list : list[float], dir:str): 
    return {}


def reward_score_statistics(reward_scores: NDArray, num_attempts_per_question: int) -> dict:
    """
    Compute summary stats for each expected reward key over an array of per-sample reward dicts.

    - reward_scores: NDArray of dict[str, float], each dict may contain a subset of EXPECTED_REWARD_SCORES
    - Missing keys are treated as 0.0

    Returns a flat dict with keys like:
      - "reward/<name>/mean"
      - "reward-less-important/<name>/<stat>" where stat in {q1, q3, median, max, min, 10%, 90%}
    """
    out: dict = {}
    if reward_scores is None or len(reward_scores) == 0:
        return out

    # Ensure iterable of dicts
    rs_list = list(reward_scores)
    num = len(rs_list)

    # Check that len(reward_scores) is divisible by num_attempts_per_question
    if num % num_attempts_per_question != 0:
        assert False, (
            f"len(reward_scores) ({num}) must be divisible by num_attempts_per_question ({num_attempts_per_question})"
        )

    num_questions = num // num_attempts_per_question

    for name in EXPECTED_REWARD_SCORES:
        # Build values array, defaulting missing entries to 0.0
        vals = np.fromiter((float(sample.get(name, 0.0)) for sample in rs_list), dtype=float, count=num)

        # Mean
        out[f"reward/{name}/mean"] = float(np.mean(vals))

        # Additional quantiles and extrema
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        p10, p90 = np.quantile(vals, [0.10, 0.90])
        out[f"reward-less-important/{name}/q1"] = float(q1)
        out[f"reward-less-important/{name}/median"] = float(med)
        out[f"reward-less-important/{name}/q3"] = float(q3)
        out[f"reward-less-important/{name}/min"] = float(np.min(vals))
        out[f"reward-less-important/{name}/max"] = float(np.max(vals))
        out[f"reward-less-important/{name}/10%"] = float(p10)
        out[f"reward-less-important/{name}/90%"] = float(p90)
    # Totals across all keys per sample
    totals = np.fromiter(
        (float(sum(sample.get(k, 0.0) for k in EXPECTED_REWARD_SCORES)) for sample in rs_list),
        dtype=float,
        count=num,
    )
    out["reward/total/mean"] = float(np.mean(totals))
    out["reward-less-important/total/min"] = float(np.min(totals))
    out["reward-less-important/total/max"] = float(np.max(totals))
    q1, med, q3 = np.quantile(totals, [0.25, 0.5, 0.75])
    p10, p90 = np.quantile(totals, [0.10, 0.90])
    out["reward-less-important/total/q1"] = float(q1)
    out["reward-less-important/total/median"] = float(med)
    out["reward-less-important/total/q3"] = float(q3)
    out["reward-less-important/total/10%"] = float(p10)
    out["reward-less-important/total/90%"] = float(p90)

    # Calculate per-question correctness statistics
    # Extract correctness scores and group by question
    correctness_vals = np.fromiter((float(sample.get("is_correct", 0.0)) for sample in rs_list), dtype=float, count=num)

    # Reshape to group attempts by question: [num_questions, num_attempts_per_question]
    correctness_by_question = correctness_vals.reshape(num_questions, num_attempts_per_question)

    # Calculate mean correctness score per question
    question_means = np.mean(correctness_by_question, axis=1)

    # Calculate statistics of the per-question means
    out["reward-less-important/is_correct_per_question/min"] = float(np.min(question_means))
    out["reward-less-important/is_correct_per_question/max"] = float(np.max(question_means))
    q1_qmeans, q3_qmeans = np.quantile(question_means, [0.25, 0.75])
    out["reward-less-important/is_correct_per_question/q1"] = float(q1_qmeans)
    out["reward-less-important/is_correct_per_question/q3"] = float(q3_qmeans)

    return out


def _get_tagged_data(text: str, tagname: str) -> str | None:
    """
    Return inner text for exactly one <tagname>...</tagname> using find-based checks.
    None if zero or multiple occurrences.
    """
    if not isinstance(text, str) or not isinstance(tagname, str) or not tagname:
        return None
    open_tag = f"<{tagname}>"
    close_tag = f"</{tagname}>"

    first_open = text.find(open_tag)
    if first_open == -1:
        return None
    open_end = first_open + len(open_tag)
    first_close = text.find(close_tag, open_end)
    if first_close == -1:
        return None

    inner = text[open_end:first_close]
    # Ensure unique occurrence: no other open/close tags outside this pair
    if text.find(open_tag, open_end) != -1:
        return None
    if text.find(close_tag, first_close + len(close_tag)) != -1:
        return None
    # No nested same tags inside
    if inner.find(open_tag) != -1 or inner.find(close_tag) != -1:
        return None
    return inner


def _is_correct(proposed_sol: str, ground_truth: str, tol: float = 1e-4) -> bool:
    a = parse(proposed_sol)
    b = parse(ground_truth)
    if a is None or b is None:
        return False
    return verify(b, a)