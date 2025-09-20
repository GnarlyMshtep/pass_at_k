import os
from typing import Any, List, Dict, Literal
from collections import defaultdict

import numpy as np
import torch
import math
from math_verify import parse, verify
from numpy.typing import NDArray

REWARD_CORRECT = 1
EXPECTED_REWARD_SCORES = ["is_correct"]  # Add other expected reward score keys as needed


def extract_attempts(sol_str: str, tagname: str, n_rollout: int) -> list[str]:
    """
    Extract content from numbered tags like <tagname-1>...</tagname-1>, <tagname-2>...</tagname-2>, etc.
    Now returns partial results even if some tags are missing/empty/duplicated.
    
    Args:
        sol_str: The input string to search
        tagname: The base tag name (e.g., "attempt")
        n_rollout: The number of expected tags (1 to n_rollout)
    
    Returns:
        List of valid strings (may be shorter than n_rollout), or empty list if no valid attempts
    """
    if not isinstance(sol_str, str) or not isinstance(tagname, str) or not isinstance(n_rollout, int):
        return []
    
    if n_rollout <= 0:
        return []
    
    results = []
    
    for i in range(1, n_rollout + 1):
        open_tag = f"<{tagname}-{i}>"
        close_tag = f"</{tagname}-{i}>"
        
        # Find first occurrence of open tag
        first_open = sol_str.find(open_tag)
        if first_open == -1:
            continue  # Tag missing, skip this attempt
        
        # Find matching close tag
        open_end = first_open + len(open_tag)
        first_close = sol_str.find(close_tag, open_end)
        if first_close == -1:
            continue  # No matching close tag, skip this attempt
        
        # Check for duplicate open tags of same number
        if sol_str.find(open_tag, open_end) != -1:
            continue  # Duplicate open tag, skip this attempt
        
        # Check for duplicate close tags of same number
        if sol_str.find(close_tag, first_close + len(close_tag)) != -1:
            continue  # Duplicate close tag, skip this attempt
        
        # Extract content between tags
        inner = sol_str[open_end:first_close]
        
        # Check for nested same tags inside (shouldn't happen with numbered tags but being safe)
        if inner.find(open_tag) != -1 or inner.find(close_tag) != -1:
            continue  # Nested tags, skip this attempt
        
        # Check if the attempt is empty (contains only whitespace or nothing)
        if not inner.strip():
            continue  # Empty attempt, skip this attempt
        
        results.append(inner)
    
    return results

def compute_score_multi_attempt_per_rollout_math(data_source, solution_str, ground_truth, extra_info=None)-> float:
    # Determine N_ROLLOUTS from extra_info -> env -> default
    if extra_info is not None and isinstance(extra_info, dict) and "num_attempts" in extra_info:
        try:
            N_ROLLOUTS = int(extra_info["num_attempts"])
        except Exception:
            N_ROLLOUTS = 4
    else:
        try:
            N_ROLLOUTS = int(os.environ.get("N_ROLLOUTS", 4))
        except Exception:
            N_ROLLOUTS = 4
    N_ROLLOUTS = max(1, N_ROLLOUTS)

    extracted_attempts = extract_attempts(solution_str, "attempt", N_ROLLOUTS)
    
    # If no valid attempts found, return 0
    if not extracted_attempts: 
        return 0       
    
    # Check correctness for each valid attempt
    correctness_per_attempt = [_is_correct(proposed_sol=attempt, ground_truth=ground_truth) for attempt in extracted_attempts]
    
    # If all attempts are present and at least one is correct, return full reward + formatting bonus
    if len(extracted_attempts) == N_ROLLOUTS and any(correctness_per_attempt):
        return max(correctness_per_attempt) + .1
    
    # If not all attempts are present, return partial reward based on number of valid attempts
    if len(extracted_attempts) < N_ROLLOUTS:
        return 0.1 * len(extracted_attempts) / N_ROLLOUTS
    
    # If all attempts are present but none are correct, return formatting bonus only
    return 0.1

def compute_score_multi_attempt_per_rollout_taller(data_source, solution_str, ground_truth, extra_info=None)-> float:
    # Determine N_ROLLOUTS from extra_info -> env -> default
    if extra_info is not None and isinstance(extra_info, dict) and "num_attempts" in extra_info:
        try:
            N_ROLLOUTS = int(extra_info["num_attempts"])
        except Exception:
            N_ROLLOUTS = 3
    else:
        try:
            N_ROLLOUTS = int(os.environ.get("N_ROLLOUTS", 3))
        except Exception:
            N_ROLLOUTS = 3
    N_ROLLOUTS = max(1, N_ROLLOUTS)

    extracted_attempts = extract_attempts(solution_str, "attempt", N_ROLLOUTS)
    
    # If no valid attempts found, return 0
    if not extracted_attempts: 
        return 0       
    
    # Check correctness for each valid attempt
    correctness_per_attempt = [_taller_is_correct(proposed_sol=attempt, ground_truth=ground_truth) for attempt in extracted_attempts]
    
    # If all attempts are present and at least one is correct, return full reward + formatting bonus
    if len(extracted_attempts) == N_ROLLOUTS and any(correctness_per_attempt):
        return max(correctness_per_attempt) + .1
    
    # If not all attempts are present, return partial reward based on number of valid attempts
    if len(extracted_attempts) < N_ROLLOUTS:
        return 0.1 * len(extracted_attempts) / N_ROLLOUTS
    
    # If all attempts are present but none are correct, return formatting bonus only
    return 0.1
        

def extra_reward_metrics(responses: list[str], prompts: list[str], ground_truths:list[str]) ->dict[str, Any]:
    # breakpoint() #M: would like to inspect what's going on
    
    # N_ROLLOUTS = os.getenv("N_ROLLOUTS", None)
    # assert N_ROLLOUTS != None, "please set env variable N_ROLLOUTS to the number of rollouts per question, i.e the number of approaches"
    # N_ROLLOUTS = int(N_ROLLOUTS)
    
    # solutions = [_get_tagged_data(response, "solution") for response in responses]
    # print(f"{prompts=} {solutions=} ")
    
    #TODO: calculate diversity score avg(set(proposed_answers) / |proposed_answer|) for proposed_answers for a question in questions

    return {}

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

def _taller_is_correct(proposed_sol: str, ground_truth:str) -> bool: 
    proposed_sol= proposed_sol.strip().replace(" ", "")
    ground_truth=ground_truth.strip().replace(" ", "")

    proposed_sol_set = set()
    ground_truth_set = set()
    for i, char in enumerate(proposed_sol): 
        if i %2 == 0: 
            proposed_sol_set.add(char)
        elif char != ",":
            return False 
    
    for i, char in enumerate(ground_truth): 
        if i %2 == 0: 
            ground_truth_set.add(char)
        elif char != ",":
            print(f"DEBUG: grouth truth has unexpected format {ground_truth}")

    return proposed_sol_set == ground_truth_set   
    


def _is_correct(proposed_sol: str, ground_truth: str, tol: float = 1e-4) -> bool:
    a = parse(proposed_sol)
    b = parse(ground_truth)
    if a is None or b is None:
        return False
    return verify(b, a)


def more_reward_util_func(batch):
    if "uid" in batch.non_tensor_batch:
        uid_list = batch.non_tensor_batch["uid"]

        label_key = None
        for key in ["acc", "is_correct", "exact_match", "score"]: #M: what is my score 
            if key in batch.non_tensor_batch:
                label_key = key
                break

        labels_np = None
        if label_key is not None:
            labels_np = np.asarray(batch.non_tensor_batch[label_key], dtype=float)
        elif "reward" in batch.non_tensor_batch:
            labels_np = (np.asarray(batch.non_tensor_batch["reward"], dtype=float) > 0.5).astype(int)
        if labels_np is not None and len(labels_np) == len(uid_list):
            uid_to_indices: dict[Any, list[int]] = defaultdict(list)
            for idx, uid in enumerate(uid_list):
                uid_to_indices[uid].append(idx)
        else:
            raise ValueError
            # if "token_level_rewards" in batch.batch:
            #     seq_rewards = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
            #     labels_np = (seq_rewards > 0.5).astype(int)

        
        def compute_reward_metrics(rewards: NDArray[float])->Dict[str, float]: 
            def expected_pass_at_k(rewards, k)-> float: 
                c = int(sum(rewards))
                n = int(len(rewards))
                if n <= 0 or k <= 0 or k > n:
                    return 0.0
                # Clamp c to [0, n] for safety
                c = max(0, min(c, n))
                return 1 - (math.comb(n - c, k) / math.comb(n, k))
            
            rewards = (rewards >=1).astype(int) # we binarized our reward, we only care about whether you are greater or less than 1 to avoid format confounding and complicating pass@k metric
            return {
                f"max@{len(rewards)}" : max(rewards),
                f"mean@{len(rewards)}" : np.mean(rewards),
                f"min@{len(rewards)}" : min(rewards),
                f"p25@{len(rewards)}" : np.percentile(rewards, 25),
                f"p75@{len(rewards)}" : np.percentile(rewards, 75),
                f"pass@1" : expected_pass_at_k(rewards, 1),    
                f"pass@2" : expected_pass_at_k(rewards, 2),    
                f"pass@4" : expected_pass_at_k(rewards, 4),    
                f"pass@8" : expected_pass_at_k(rewards, 8),    
            }            

        metrics_per_qs:List[dict]= [compute_reward_metrics(labels_np[indices]) for indices in uid_to_indices.values()]
        metrics_update = {k : sum([metrics_per_q[k] for metrics_per_q in metrics_per_qs]) / len(metrics_per_qs) for k in metrics_per_qs[0].keys()} #average across all qs
        # Prefix with train/ so they appear grouped under training metrics in loggers
        prefixed_metrics_update = {f"train/{k}": v for k, v in metrics_update.items()}
        return prefixed_metrics_update