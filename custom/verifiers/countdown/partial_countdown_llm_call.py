from functools import lru_cache
import ast
import operator
import typing as _t
from collections import Counter
import time
import os

try:
    # OpenAI-compatible client (works with vLLM/OpenAI APIs)
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

def _extract_answer(s: str) -> str:
    if ("<answer>" in s and "</answer>" in s):
        return s.split("<answer>")[-1].split("</answer>")[0].strip()
    return None

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

def _count_binops_in_expr(expr: str) -> int:
    """Count binary +, -, *, / operations in an arithmetic expression."""
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return 0

    count = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and type(sub.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            count += 1
    return count


def _can_merge_to_target(values: list[float], target: float, tol: float = 1e-6) -> tuple[bool, int]:
    """
    Given a list of numeric values (results of sub-expressions), determine whether
    they can be combined using +, -, *, / and parentheses to reach `target`.
    If true, also return the total number of search actions explored (including
    backtracked branches) until a solution is found. This mirrors the behavior
    used in the notebook's search-based accounting.
    """
    if not values:
        return False, 0

    # Round for a stable memoization key to mitigate FP noise.
    def _key(nums: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(sorted(round(x, 8) for x in nums))

    @lru_cache(maxsize=None)
    def dfs(state: tuple[float, ...]) -> tuple[bool, int]:
        if len(state) == 1:
            return abs(state[0] - target) < tol, 0

        n = len(state)
        total_actions = 0
        # Try all unordered pairs (i<j); we'll generate both a-b and b-a etc.
        for i in range(n):
            for j in range(i + 1, n):
                a, b = state[i], state[j]
                rest = [state[k] for k in range(n) if k != i and k != j]

                candidates = [
                    a + b,
                    a * b,
                    a - b,
                    b - a,
                ]
                if abs(b) > tol:
                    candidates.append(a / b)
                if abs(a) > tol:
                    candidates.append(b / a)

                for c in candidates:
                    next_state = tuple(rest + [c])
                    can_merge, actions = dfs(_key(next_state))
                    total_actions += actions + 1  # Count this search expansion
                    if can_merge:
                        return True, total_actions
        return False, total_actions

    return dfs(_key(tuple(values)))


# -------------------- LLM-based verifier helpers --------------------

def _build_prompt_messages(nums: list[float], target: float, solution_str: str) -> list[dict]:
    """
    Build an instruction following the Countdown prompt structure used in preprocessing,
    augmenting with the provided solution_str as a scratch pad inside <think> tags.
    """
    numbers_fmt = [int(n) if float(n).is_integer() else n for n in nums]
    content = (
        f"Using the numbers {numbers_fmt}, create an expression that equals {int(target) if float(target).is_integer() else target}. "
        "You can use basic arithmetic operations (+, -, *, /, (, )) one or multiple times but each number can only be used once. "
        "Don't use = in the expression between the <answer> tags. "
        "Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. "
        "Think step by step inside <think> tags.\n\n"
        "Here is an initial, possibly partial attempt from another solver: "
        f"<think>\n{solution_str}\n</think>\n"
        "Please continue the reasoning if needed and return ONLY the final equation in <answer> tags."
    )
    return [{"content": content, "role": "user"}]


def _call_llm_for_final_answer(nums: list[float], target: float, solution_str: str) -> _t.Optional[tuple[str, _t.Optional[str]]]:
    """Call an OpenAI-compatible chat model to complete the solution and extract <answer>.

    Returns a tuple of (raw_content, extracted_answer) or None on failure.
    """
    if OpenAI is None:
        return None

    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or "http://localhost:8000/v1"
    )
    api_key = os.getenv("LLM_API_KEY", "sk-placeholder")  # many servers ignore but client requires a string
    model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        messages = _build_prompt_messages(nums, target, solution_str)
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            max_tokens=1024,
        )
        content = (resp.choices[0].message.content or "") if resp and resp.choices else ""
        return content, _extract_answer(content)
    except Exception:
        return None


def partial_countdown_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for Partial-Countdown:
      - Parse <answer>.
      - If a single full expression equals target: reward = number of binary ops in it.
      - If a comma-separated list of sub-expressions:
            * The multiset of numbers across ALL sub-expressions must match the provided nums exactly.
            * Each sub-expression must be a valid arithmetic expr with allowed ops.
            * If their evaluated values can be merged (using +,-,*,/ and parentheses) to reach the target,
              reward = total number of binary ops present across the submitted sub-expressions.
      - Otherwise: reward = 0.0
    """
    _t0 = time.perf_counter()

    # Try LLM-based verification first. We treat the provided solution_str as a scratch pad
    # and ask the LLM to continue and submit a final <answer>.
    actual_nums = [float(x) for x in (extra_info or {}).get("nums", [])]
    try:
        task_solvable, original_actions = _can_merge_to_target(actual_nums, float(ground_truth), tol=1e-6)
        assert task_solvable, f"Task is not solvable with the given numbers: {actual_nums} -> {ground_truth}"
    except Exception:
        original_actions = 0

    llm_result = _call_llm_for_final_answer(actual_nums, float(ground_truth), solution_str)
    if llm_result is not None:
        llm_raw, llm_final_answer = llm_result
        # Validate numbers usage and correctness of the LLM-produced final expression
        submitted_nums_llm = _extract_nums_from_expr(llm_final_answer or "")
        if Counter(submitted_nums_llm) == Counter(actual_nums):
            try:
                value = _safe_eval_arithmetic(llm_final_answer or "")
                if abs(value - float(ground_truth)) < 1e-6:
                    return {
                        "score": float(1) + 0.2,
                        "time": time.perf_counter() - _t0,
                        "origianl_actions": int(original_actions),
                        "taken_actions": 0,
                        "llm_response": llm_raw,
                    }
            except Exception:
                pass
        # If the LLM produced an invalid or incorrect final answer, give zero reward here
        # and avoid double-penalizing by not parsing the user's <answer> section.
        return {
            "score": 0.0,
            "time": time.perf_counter() - _t0,
            "origianl_actions": int(original_actions),
            "taken_actions": 0,
            "llm_response": llm_raw,
        }

    # If LLM path is unavailable/failing, fall back to the original rule-based parsing path.
    ans = _extract_answer(solution_str)
    if ans is None:
        return {
            "score": 0.0,
            "time": time.perf_counter() - _t0,
            "origianl_actions": 0,
            "taken_actions": 0,
            "llm_response": None,
        }

    # Split on commas to detect partial lists (strip whitespace).
    # Empty segments => invalid format.
    parts = [p.strip() for p in ans.split(",")] if "," in ans else [ans.strip()]
    if any(p == "" for p in parts):
        return {
            "score": 0.0,
            "time": time.perf_counter() - _t0,
            "origianl_actions": 0,
            "taken_actions": 0,
            "llm_response": None,
        }

    # Validate number multiset across all parts matches exactly the given nums (including duplicates).
    actual_nums = [float(x) for x in (extra_info or {}).get("nums", [])]
    submitted_nums: list[float] = []
    for p in parts:
        submitted_nums.extend(_extract_nums_from_expr(p))
    if Counter(submitted_nums) != Counter(actual_nums):
        return {
            "score": 0.0,
            "time": time.perf_counter() - _t0,
            "origianl_actions": 0,
            "taken_actions": 0,
        }

    # Single full expression case
    if len(parts) == 1:
        expr = parts[0]
        try:
            value = _safe_eval_arithmetic(expr)
        except Exception:
            return {
                "score": 0.0,
                "time": time.perf_counter() - _t0,
                "origianl_actions": 0,
                "taken_actions": 0,
                "llm_response": None,
            }

        # Compute baseline actions for the original task for reporting consistency
        task_solvable, original_actions = _can_merge_to_target(actual_nums, float(ground_truth), tol=1e-6)
        assert task_solvable, f"Task is not solvable with the given numbers: {actual_nums} -> {ground_truth}"

        # For single-expression case, taken actions are 0 if it already equals the target; otherwise 0 as no merge is performed
        if abs(value - float(ground_truth)) < 1e-6:
            return {
                "score": float(1) + 0.2,
                "time": time.perf_counter() - _t0,
                "origianl_actions": int(original_actions),
                "taken_actions": 0,
                "llm_response": None,
            }
        return {
            "score": 0.1,
            "time": time.perf_counter() - _t0,
            "origianl_actions": int(original_actions),
            "taken_actions": 0,
            "llm_response": None,
        }

    # Multiple partial sub-expressions case
    # 1) Each part must be a valid expression; collect their evaluated values.
    values: list[float] = []
    for p in parts:
        try:
            values.append(_safe_eval_arithmetic(p))
        except Exception:
            return {
                "score": 0.0,
                "time": time.perf_counter() - _t0,
                "origianl_actions": 0,
                "taken_actions": 0,
                "llm_response": None,
            }

    # 2) Check if these values can be merged (with +,-,*,/) to hit the target.
    task_solvable, original_actions = _can_merge_to_target(actual_nums, float(ground_truth), tol=1e-6)
    assert task_solvable, f"Task is not solvable with the given numbers: {actual_nums} -> {ground_truth}"
    
    can_merge, taken_actions = _can_merge_to_target(values, float(ground_truth), tol=1e-6)
    if not can_merge:
        return {
            "score": 0.2,
            "time": time.perf_counter() - _t0,
            "origianl_actions": int(original_actions),
            "taken_actions": int(taken_actions),
            "llm_response": None,
        }

    # 3) Reward is the total number of binary ops already performed inside the submitted parts.
    reward = (original_actions - taken_actions) / original_actions
    return {
        "score": max(float(reward), 0) + 0.2,
        "time": time.perf_counter() - _t0,
        "origianl_actions": int(original_actions),
        "taken_actions": int(taken_actions),
        "llm_response": None,
    }
