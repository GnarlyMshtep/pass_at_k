import ast
import operator
import typing as _t
from collections import Counter

def _extract_answer(s: str) -> str:
    if ("<answer>" in s and "</answer>" in s):
        s = s.split("<answer>")[-1].split("</answer>")[0].strip()
    return s.strip()


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
    """Reward for Countdown: parse <answer>, eval expression, compare to target.

    Accept if evaluated value equals target within a small tolerance.
    """
    expr = _extract_answer(solution_str)
    actual_nums = extra_info['nums']
    expr_nums = _extract_nums_from_expr(expr)
    # Check if the two multisets (with repetitions) are equal
    
    if Counter(expr_nums) != Counter(actual_nums):
        return 0.0
    try:
        value = _safe_eval_arithmetic(expr)
    except Exception:
        return 0.0

    return 1.0 if abs(value - ground_truth) < 1e-6 else 0.0


