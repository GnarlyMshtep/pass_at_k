import ast
import operator
import typing as _t
from collections import Counter
import re
import time

def _extract_answer_blocks(s: str) -> list[str]:
    """Extract all <answer>...</answer> blocks, returning their inner contents.

    Only properly paired tags are considered. Stray closing/opening tags are ignored.
    """
    blocks: list[str] = []
    for m in re.finditer(r"<answer>[\s\S]*?</answer>", s):
        content = m.group(0)[len("<answer>") : -len("</answer>")]
        blocks.append(content)
    return [b.strip() for b in blocks]


def _extract_attempt_blocks(s: str) -> list[tuple[str, str]]:
    """Extracts attempt blocks as (tag, content) pairs.

    Supports numbered <attempt-i>...</attempt-i> and generic <attempt>...</attempt>.
    """
    attempts: list[tuple[str, str]] = []

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


def sat_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """
    Reward for sat (single or multi-attempt):
      - Parses multiple <attempt-i>/<attempt> answers if present; otherwise falls back to single <answer>.
      - Each attempt must be an answer like a:true,b:false,c:true
      - best attempt is used (1.0 for correct, else 0.0).
      - Adds a formatting bonus (+0.2) when at least one valid attempt is parsed and evaluated.
      - Respects extra_info.max_allowed_attempts if provided.
      - Returns a dict similar to taller_puzzles_verifier.
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

    # Determine expected attempts (prefer max_allowed_attempts, fallback to num_attempts, default 1)
    expected_attempts = extra_info["max_allowed_attempts"]

    # Gather attempts or fallback to <answer>
    attempt_blocks = _extract_attempt_blocks(solution_str)
    blocks: list[tuple[str, str]] = []
    if attempt_blocks:
        # Require at least expected_attempts attempts to be present
        if len(attempt_blocks) < expected_attempts:
            t1 = time.perf_counter()
            return {
                "score": 0.0,
                "is_correct": 0,
                "format_score": 0.0,
                "time": t1 - t0,
                "exact_match": 0,
                "attempts": int(len(attempt_blocks)),
                "best_attempt_index": None,
                "pred": "",
                "ground_truth": str(ground_truth),
                "reason": "insufficient_attempts",
            }
        # Trim to exactly expected_attempts
        blocks = attempt_blocks[:expected_attempts]
    else:
        answer_blocks = _extract_answer_blocks(solution_str)
        if answer_blocks:
            # Require the number of <answer> blocks to match expected attempts exactly
            if len(answer_blocks) != expected_attempts:
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
                    "reason": f"wrong_format, expected {expected_attempts}, get {len(answer_blocks)}",
                }
            blocks = [("answer", b) for b in answer_blocks]

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
            "reason": "No blocks was found",
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
    format_bonus = 0.2 if (evaluated > 0 and len(blocks) == expected_attempts) else 0.0
    return {
        "score": float(best_exact) + format_bonus,
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


