"""
Port of miles' DAPO reward function (math_dapo_utils.compute_score).
Minerva-style "Answer:" extraction + normalize_final_answer.
Returns -1/+1 to match miles' reward scale.

Source: /shared/matan/code/miles/miles/rollout/rm_hub/math_dapo_utils.py
"""

import re


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes",
    "digits", "cents", "degrees", "cm", "gm", "pounds", "meters",
    "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2",
    "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots", "<|im_end|>", "<|endoftext|>",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = str(final_answer)
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)",
) -> tuple[bool, str]:
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    gt = str(int(float(gt)))
    return (pred == gt), pred


async def compute_score_dapo_minerva(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    global_step: int | None = None,
) -> dict:
    solution_str = solution_str[-300:]
    correct, pred = is_correct_minerva(solution_str=solution_str, gt=ground_truth)
    return {
        "score": 1.0 if correct else -1.0,
        "is_correct": correct,
        "pred": pred,
    }
