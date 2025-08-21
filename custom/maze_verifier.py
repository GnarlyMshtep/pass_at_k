import typing as _t


def extract_answer_maze(s: str) -> str:
    if ('<answer>' in s and "</answer>" in s):
        s = s.split("<answer>")[-1].strip().split("</answer>")[0].strip()
    ans = s.split("boxed")
    if len(ans) == 1:
        return s
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except Exception:
        return ""
    return a


def compute_score(solution: str, maze: str) -> int:
    solution = extract_answer_maze(solution)
    solution = solution.upper()
    maze_lines = maze.strip().split('\n')
    n = len(maze_lines)
    m = len(maze_lines[0])

    def find_st(lines: list[str]) -> tuple[int, int]:
        for i in range(n):
            for j in range(m):
                if lines[i][j] == 'S':
                    return i, j
        raise AssertionError

    x, y = find_st(maze_lines)
    for step in solution:
        if step == 'L':
            y -= 1
        elif step == 'R':
            y += 1
        elif step == 'U':
            x -= 1
        elif step == 'D':
            x += 1
        else:
            continue

        if x < 0 or x >= n:
            return 0
        if y < 0 or y >= m:
            return 0
        if maze_lines[x][y] == '*':
            return 0

    return 1 if maze_lines[x][y] == 'E' else 0


def maze_compute_score(
    *,
    data_source: _t.Any,
    solution_str: str,
    ground_truth: _t.Any,
    extra_info: _t.Optional[dict] = None,
) -> float | dict:
    """Adapter for NaiveRewardManager.compute_score using this maze verifier.

    It expects the maze layout string either directly in `ground_truth` (str),
    or nested in a mapping under a common key like "maze" / "label" / "ground_truth" / "answer".
    Falls back to `extra_info['maze']` if provided.
    Returns 1.0 on success, 0.0 otherwise.
    """
    maze_text: _t.Optional[str] = None
    if isinstance(ground_truth, str):
        maze_text = ground_truth
    elif isinstance(ground_truth, dict):
        for key in ("maze", "label", "ground_truth", "answer"):
            value = ground_truth.get(key)
            if isinstance(value, str) and value.strip():
                maze_text = value
                break

    if maze_text is None and extra_info and isinstance(extra_info, dict):
        maybe_maze = extra_info.get("maze")
        if isinstance(maybe_maze, str) and maybe_maze.strip():
            maze_text = maybe_maze

    if not isinstance(maze_text, str) or not maze_text.strip():
        return 0.0

    try:
        score = compute_score(solution_str, maze_text)
    except Exception:
        return 0.0

    try:
        return float(score)
    except Exception:
        return 0.0


