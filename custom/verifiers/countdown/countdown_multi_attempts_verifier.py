import ast
import operator
import typing as _t
from collections import Counter


def _extract_attempts(s: str, max_attempts: int = 8) -> list[str]:
    attempts: list[str] = []
    for i in range(1, max_attempts + 1):
        start_tag = f"<attempt-{i}>"
        end_tag = f"</attempt-{i}>"
        if start_tag in s and end_tag in s:
            try:
                attempts.append(s.split(start_tag)[-1].split(end_tag)[0].strip())
            except Exception:
                continue
    return attempts


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
    except Exception:
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


def countdown_multi_attempts_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """Reward for Countdown multi-attempts:
    Parse <attempt-i> blocks, eval each expression, compare to target.
    Accept if any attempt equals target within a small tolerance and uses the exact multiset of numbers.
    """
    attempts = _extract_attempts(solution_str, max_attempts=8)
    if not attempts:
        return 0.0

    actual_nums = extra_info['nums'] if extra_info and 'nums' in extra_info else None

    for expr in attempts:
        try:
            if actual_nums is not None:
                expr_nums = _extract_nums_from_expr(expr)
                if Counter(expr_nums) != Counter(actual_nums):
                    continue
            value = _safe_eval_arithmetic(expr)
        except Exception:
            continue
        if abs(value - float(ground_truth)) < 1e-6:
            return 1.0

    return 0.0


