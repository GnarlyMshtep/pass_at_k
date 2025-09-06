from functools import lru_cache
import ast
import operator
import typing as _t
from collections import Counter
import time

from custom.verifiers.countdown import _merge_search as _merge_search_ext


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


    # # Round for a stable memoization key to mitigate FP noise.
    # def _key(nums: tuple[float, ...]) -> tuple[float, ...]:
    #     return tuple(sorted(round(x, 8) for x in nums))

    # @lru_cache(maxsize=None)
    # def dfs(state: tuple[float, ...]) -> tuple[bool, int]:
    #     if len(state) == 1:
    #         return abs(state[0] - target) < tol, 0

    #     n = len(state)
    #     total_actions = 0
    #     # Try all unordered pairs (i<j); we'll generate both a-b and b-a etc.
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             a, b = state[i], state[j]
    #             rest = [state[k] for k in range(n) if k != i and k != j]

    #             candidates = [
    #                 a + b,
    #                 a * b,
    #                 a - b,
    #                 b - a,
    #             ]
    #             if abs(b) > tol:
    #                 candidates.append(a / b)
    #             if abs(a) > tol:
    #                 candidates.append(b / a)

    #             for c in candidates:
    #                 next_state = tuple(rest + [c])
    #                 can_merge, actions = dfs(_key(next_state))
    #                 total_actions += actions + 1  # Count this search expansion
    #                 if can_merge:
    #                     return True, total_actions
    #     return False, total_actions

    # return dfs(_key(tuple(values)))


def _can_merge_to_target_with_timeout(
    values: list[float], target: float, tol: float = 1e-6, timeout_s: float = 0.1
) -> tuple[bool, int]:
    # Unified C++ function with timeout support
    can_merge, actions = _merge_search_ext.can_merge_to_target(
        list(values), float(target), float(tol), float(timeout_s * 1000.0), -1
    )
    return bool(can_merge), int(actions)

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
    # breakpoint()
    _t0 = time.perf_counter()
    ans = _extract_answer(solution_str)
    if ans is None:
        return {
            "score": 0.0,
            "time": time.perf_counter() - _t0,
            "origianl_actions": 0,
            "taken_actions": 0,
        }

    # Split on commas to detect partial lists (strip whitespace).
    parts = [p.strip() for p in ans.split(",")] if "," in ans else [ans.strip()]

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
            }

        # Use provided baseline actions from data to avoid unbounded search
        original_actions = int(extra_info["original_actions"])  # KeyError if missing
        assert original_actions >= 0, f"Invalid original actions: {original_actions}"

        # For single-expression case, taken actions are 0 if it already equals the target; otherwise 0 as no merge is performed
        correctness_score = 1.0 if abs(value - float(ground_truth)) < 1e-6 else 0.0
        return {
            "score": float(correctness_score) + 0.2,
            "time": time.perf_counter() - _t0,
            "origianl_actions": int(original_actions),
            "taken_actions": 0,
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
            }

    # 2) Check if these values can be merged (with +,-,*,/) to hit the target.
    original_actions = int(extra_info["original_actions"])
    assert original_actions >= 0, f"Invalid original actions: {original_actions}"
    
    can_merge, taken_actions = _can_merge_to_target_with_timeout(values, float(ground_truth), tol=1e-6, timeout_s=0.1)

    # 3) Reward is the total number of binary ops already performed inside the submitted parts.
    reward = ((original_actions - taken_actions) / original_actions) if can_merge else 0.0
    return {
        "score": max(float(reward), 0) + 0.2,
        "time": time.perf_counter() - _t0,
        "origianl_actions": int(original_actions),
        "taken_actions": int(taken_actions),
    }
