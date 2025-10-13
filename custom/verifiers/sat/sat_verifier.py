import ast
import operator
import typing as _t
from collections import Counter
import re
import time


def _extract_last_terminal_answer(s: str) -> _t.Optional[str]:
    """Return the content of the last <answer>...</answer> block only if it
    is the terminal element in the string (i.e., no non-whitespace after it).
    Otherwise return None.
    """
    matches = list(re.finditer(r"<answer>[\s\S]*?</answer>", s))
    if not matches:
        return None
    last = matches[-1]
    trailing = s[last.end() :]
    if trailing.strip():
        return None
    content = s[last.start() + len("<answer>") : last.end() - len("</answer>")]
    return content.strip()


def sat_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for SAT:
      - Extract only the last <answer>...</answer> block from the response.
      - Reject if there is any non-whitespace text after the final </answer>.
      - The answer must be like a:true,b:false,c:true
      - Score is 1.0 if correct else 0.0, plus a formatting bonus (+0.2)
        when a valid terminal answer is parsed and evaluated.
    """
    t0 = time.perf_counter()

    sat = []
    variable_labels = []
    try:
        if extra_info and isinstance(extra_info, dict):
            sat = extra_info.get("raw_sat", "[]" )
            variable_labels = extra_info.get("variable_labels", "[]")
        else:
            print(f"raw_sat is missing from extra_info")
    except Exception as e:
        print(f"Error in raw_sat extraction.")

    # Extract only the last terminal <answer> block
    last_answer = _extract_last_terminal_answer(solution_str)
    blocks: list[tuple[str, str]] = []
    if last_answer is not None:
        blocks = [("answer", last_answer)]

    if not blocks:
        t1 = time.perf_counter()
        return {
            "score": 0.0,
            "is_correct": 0,
            "format_score": 0.0,
            "time": t1 - t0,
            "exact_match": 0,
            "attempts": 0,
            "best_attempt_index": None,
            "pred": "",
            "ground_truth": str(ground_truth),
            "reason": "wrong_format: no terminal <answer> block or trailing text",
        }

    best_exact = 0
    best_idx = -1
    best_sol = ""
    evaluated = 0
    reason = ""
    
    # Parse variable labels from extra_info
    try:
        import json
        if isinstance(variable_labels, str):
            variable_labels = json.loads(variable_labels)
        if isinstance(sat, str):
            sat = json.loads(sat)
    except:
        pass
    
    for idx, (_tag, content) in enumerate(blocks):
        # Parse the solution assignment like "A:true,B:false,C:true"
        solution_str = (content or "").strip()
        if not solution_str:
            reason += f"{idx}: empty solution\n"
            continue
            
        # Parse assignment
        assignment = {}
        try:
            for part in solution_str.split(","):
                part = part.strip()
                if ":" not in part:
                    reason += f"{idx}: invalid format '{part}', expected 'var:value'\n"
                    continue
                var, value = part.split(":", 1)
                var = var.strip()
                value = value.strip().lower()
                if value not in ["true", "false"]:
                    reason += f"{idx}: invalid value '{value}' for variable '{var}', expected 'true' or 'false'\n"
                    continue
                assignment[var] = (value == "true")
        except Exception as e:
            reason += f"{idx}: error parsing solution: {e}\n"
            continue
            
        evaluated += 1
        
        # Check if all clauses are satisfied
        all_satisfied = True
        for clause_idx, clause in enumerate(sat):
            clause_satisfied = False
            for literal in clause:
                # literal is an integer: positive for variable, negative for negation
                var_idx = abs(literal)
                if var_idx >= len(variable_labels):
                    reason += f"{idx}: clause {clause_idx} has invalid variable index {var_idx}\n"
                    all_satisfied = False
                    break

                
                var_name = variable_labels[var_idx]
                if var_name not in assignment:
                    reason += f"{idx}: clause {clause_idx} references undefined variable '{var_name}'\n"
                    all_satisfied = False
                    break
                    
                var_value = assignment[var_name]
                # If literal is positive, use var_value; if negative, use not var_value
                literal_value = var_value if literal > 0 else not var_value
                
                if literal_value:
                    clause_satisfied = True
                    break

            if not all_satisfied:
                break

            if all_satisfied and not clause_satisfied:
                reason += f"{idx}: clause {clause_idx} is not satisfied\n"
                all_satisfied = False
                break

            
                
        exact = 1 if all_satisfied else 0
        
        if exact >= best_exact:
            best_exact = exact
            best_idx = idx
            best_sol = solution_str

    t1 = time.perf_counter()
    format_bonus = 0.2 if evaluated > 0 else 0.0
    return {
        "score": float(best_exact), # + format_bonus
        "is_correct": int(best_exact),
        "format_score": format_bonus,
        "time": t1 - t0,
        "exact_match": int(best_exact),
        "attempts": int(evaluated),
        "best_attempt_index": int(best_idx) if best_idx is not None else None,
        "pred": best_sol,
        "ground_truth": str(ground_truth),
        "reason": reason,
    }


