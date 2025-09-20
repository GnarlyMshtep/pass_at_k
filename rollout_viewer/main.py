from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles


def get_base_root() -> Path:
    env_root = os.environ.get(
        "ROLLOUT_ROOT",
        "/scratch/m000122/stalaei/logs/pass_at_k/rollouts",
    )
    root = Path(env_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(
            f"Configured ROLLOUT_ROOT does not exist or is not a directory: {root}"
        )
    return root


BASE_ROOT = get_base_root()


def resolve_safe_path(relative_path: str) -> Path:
    candidate = (BASE_ROOT / relative_path).resolve()
    try:
        # Python 3.9+
        is_within = candidate.is_relative_to(BASE_ROOT)  # type: ignore[attr-defined]
    except AttributeError:
        is_within = str(candidate).startswith(str(BASE_ROOT))
    if not is_within:
        raise HTTPException(status_code=400, detail="Path escapes base root")
    return candidate


def list_directory(relative_path: Optional[str]) -> Dict[str, Any]:
    rel = relative_path or ""
    abs_path = resolve_safe_path(rel)
    if not abs_path.exists() or not abs_path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")
    entries: List[Dict[str, Any]] = []
    for entry in sorted(abs_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            entries.append(
                {
                    "name": entry.name,
                    "path": str(Path(rel) / entry.name if rel else Path(entry.name)),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                    "modified": int(entry.stat().st_mtime),
                }
            )
        except PermissionError:
            continue
    return {"path": rel, "entries": entries}


def read_jsonl_page(
    relative_path: str,
    page: int,
    page_size: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    abs_path = resolve_safe_path(relative_path)
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if abs_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

    start_index = max((page - 1), 0) * page_size
    collected: List[Dict[str, Any]] = []
    has_more = False
    with abs_path.open("r", encoding="utf-8", errors="replace") as f:
        # Skip lines before start_index efficiently
        for _ in range(start_index):
            if not f.readline():
                return ([], False)
        # Read up to page_size + 1 to detect has_more
        for i in range(page_size + 1):
            line = f.readline()
            if not line:
                break
            if i < page_size:
                idx = start_index + i
                parsed: Optional[Any] = None
                try:
                    parsed = json.loads(line)
                except Exception:
                    parsed = None
                collected.append(
                    {
                        "index": idx,
                        "raw": line.rstrip("\n"),
                        "obj": parsed,
                    }
                )
            else:
                has_more = True
    return (collected, has_more)


def read_jsonl_search(
    relative_path: str,
    query: str,
    cursor: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """Search JSONL file for lines containing query (case-insensitive) starting at line cursor.

    Returns entries, next_cursor (line offset to continue), has_more.
    """
    abs_path = resolve_safe_path(relative_path)
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if abs_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

    q = query.strip().lower()
    if not q:
        return ([], cursor, False)

    collected: List[Dict[str, Any]] = []
    has_more = False
    next_cursor = cursor
    with abs_path.open("r", encoding="utf-8", errors="replace") as f:
        # Skip to cursor
        for _ in range(cursor):
            if not f.readline():
                return ([], cursor, False)
        current_index = cursor
        while True:
            line = f.readline()
            if not line:
                next_cursor = current_index
                break
            lower_line = line.lower()
            if q in lower_line:
                try:
                    parsed = json.loads(line)
                except Exception:
                    parsed = None
                collected.append({"index": current_index, "raw": line.rstrip("\n"), "obj": parsed})
                if len(collected) >= limit:
                    has_more = True
                    next_cursor = current_index + 1
                    break
            current_index += 1
    return (collected, next_cursor, has_more)


def get_nested_value(obj: Any, dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    cur = obj
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def flatten_keys(obj: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else k
            keys.append(name)
            keys.extend(flatten_keys(v, name))
    elif isinstance(obj, list):
        # For lists, include the prefix to indicate a collection
        # and flatten the first element to derive nested keys
        if obj:
            keys.extend(flatten_keys(obj[0], prefix))
    return keys


def read_jsonl_keys(relative_path: str, sample: int) -> Dict[str, int]:
    abs_path = resolve_safe_path(relative_path)
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if abs_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")
    counts: Dict[str, int] = {}
    total = 0
    with abs_path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(sample):
            line = f.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for key in set(flatten_keys(obj)):
                counts[key] = counts.get(key, 0) + 1
            total += 1
    return counts


def read_jsonl_key_filter(
    relative_path: str,
    filter_key: str,
    filter_value: str,
    cursor: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    abs_path = resolve_safe_path(relative_path)
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if abs_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

    fv = filter_value.strip().lower()
    if not fv:
        return ([], cursor, False)

    collected: List[Dict[str, Any]] = []
    has_more = False
    next_cursor = cursor
    with abs_path.open("r", encoding="utf-8", errors="replace") as f:
        # Skip to cursor
        for _ in range(cursor):
            if not f.readline():
                return ([], cursor, False)
        current_index = cursor
        while True:
            line = f.readline()
            if not line:
                next_cursor = current_index
                break
            try:
                obj = json.loads(line)
            except Exception:
                obj = None
            val = None
            if obj is not None:
                val = get_nested_value(obj, filter_key)
            if isinstance(val, str) and fv in val.lower():
                collected.append({"index": current_index, "raw": line.rstrip("\n"), "obj": obj})
                if len(collected) >= limit:
                    has_more = True
                    next_cursor = current_index + 1
                    break
            current_index += 1
    return (collected, next_cursor, has_more)


app = FastAPI(title="Rollout Viewer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/root")
def api_root() -> Dict[str, Any]:
    return {"root": str(BASE_ROOT)}


@app.get("/api/tree")
def api_tree(path: Optional[str] = Query(None, description="Path relative to root")) -> Dict[str, Any]:
    return list_directory(path)


@app.get("/api/jsonl")
def api_jsonl(
    path: str = Query(..., description=".jsonl path relative to root"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    q: Optional[str] = Query(None, description="Case-insensitive search query"),
    cursor: int = Query(0, ge=0),
    filter_key: Optional[str] = Query(None, description="Dotted key path to filter on"),
    filter_value: Optional[str] = Query(None, description="Substring to match in the key's string value"),
) -> Dict[str, Any]:
    try:
        if filter_key and (filter_value is not None and filter_value.strip() != ""):
            entries, next_cursor, has_more = read_jsonl_key_filter(path, filter_key, filter_value, cursor, page_size)
            return {
                "path": path,
                "filter_key": filter_key,
                "filter_value": filter_value,
                "cursor": cursor,
                "next_cursor": next_cursor,
                "limit": page_size,
                "entries": entries,
                "has_more": has_more,
            }
        elif q is not None and q.strip() != "":
            entries, next_cursor, has_more = read_jsonl_search(path, q, cursor, page_size)
            return {
                "path": path,
                "q": q,
                "cursor": cursor,
                "next_cursor": next_cursor,
                "limit": page_size,
                "entries": entries,
                "has_more": has_more,
            }
        else:
            entries, has_more = read_jsonl_page(path, page, page_size)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"path": path, "page": page, "page_size": page_size, "entries": entries, "has_more": has_more}


@app.get("/api/jsonl_keys")
def api_jsonl_keys(
    path: str = Query(..., description=".jsonl path relative to root"),
    sample: int = Query(1000, ge=1, le=20000),
) -> Dict[str, Any]:
    try:
        counts = read_jsonl_keys(path, sample)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Return keys sorted by frequency desc
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"path": path, "sampled": sum(counts.values()), "keys": [k for k, _ in items], "counts": counts}


# Serve the frontend assets
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.exception_handler(RuntimeError)
def handle_runtime_error(_, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
