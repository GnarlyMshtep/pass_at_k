"""Round-trip verification: prove data survives a DVC push/pull cycle.

Batched flow for a list of targets:
1. Move each original aside to a temp dir (same FS = instant mv)
2. dvc pull ALL .dvc files in one batch call (DVC parallelizes S3 fetches)
3. joblib-parallel compare_dirs for each (temp, pulled) pair
4. Per-target accept (delete both copies) or reject (restore original)

Caller clears the DVC cache once before calling (not per-target).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from vfh.dvc_backup.backup_log import log, vprint
from vfh.dvc_backup.dvc_ops import dvc_pull_batch
from vfh.dvc_backup.types import BackupTarget


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """MD5 hash of a file, read in 8MB chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compare_dirs(dir_a: Path, dir_b: Path) -> tuple[bool, list[str]]:
    """Compare two directories by file tree + md5 hashes.

    Returns (match: bool, differences: list[str]).
    """
    diffs: list[str] = []

    files_a: dict[str, Path] = {}
    for f in sorted(dir_a.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(dir_a))
            files_a[rel] = f

    files_b: dict[str, Path] = {}
    for f in sorted(dir_b.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(dir_b))
            files_b[rel] = f

    only_a = set(files_a.keys()) - set(files_b.keys())
    only_b = set(files_b.keys()) - set(files_a.keys())
    for rel in sorted(only_a):
        diffs.append(f"only in original: {rel}")
    for rel in sorted(only_b):
        diffs.append(f"only in pulled: {rel}")

    shared = set(files_a.keys()) & set(files_b.keys())
    for rel in sorted(shared):
        hash_a = _hash_file(path=files_a[rel])
        hash_b = _hash_file(path=files_b[rel])
        if hash_a != hash_b:
            diffs.append(f"hash mismatch: {rel} ({hash_a} vs {hash_b})")

    return (len(diffs) == 0, diffs)


# ---------------------------------------------------------------------------
# Round-trip verification helpers
# ---------------------------------------------------------------------------


def _get_temp_dir(
    target: BackupTarget,
    verify_temp_dir: str | None,
    verify_on_local_fs: bool,
) -> Path:
    """Determine where to move original data during verification."""
    if verify_temp_dir:
        return Path(verify_temp_dir) / target.run_id / target.path.name
    if verify_on_local_fs:
        return Path.home() / "dvc_verify_tmp" / target.run_id / target.path.name
    return target.path.parent / ".dvc_verify_tmp" / target.path.name


def _move_aside(
    target: BackupTarget,
    temp_dir: Path,
    verify_on_local_fs: bool,
) -> None:
    """Move target's original data to temp_dir. Handles cross-FS case."""
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    if verify_on_local_fs and not _is_same_fs(src=target.path, dst=temp_dir.parent):
        shutil.copytree(src=target.path, dst=temp_dir)
        shutil.rmtree(target.path)
    else:
        shutil.move(src=str(target.path), dst=str(temp_dir))


def _restore(original_path: Path, temp_dir: Path) -> None:
    """Restore original data from temp location."""
    if temp_dir.exists():
        shutil.move(src=str(temp_dir), dst=str(original_path))
        log(f"    Restored original at {original_path}")
    else:
        log(f"    WARNING: temp dir {temp_dir} does not exist — cannot restore!")


def _cleanup_verify_parent(temp_dir: Path) -> None:
    """Remove .dvc_verify_tmp sibling dir if it's now empty."""
    verify_parent = temp_dir.parent
    if verify_parent.name == ".dvc_verify_tmp" and verify_parent.exists():
        try:
            verify_parent.rmdir()
        except OSError:
            pass


def _is_same_fs(src: Path, dst: Path) -> bool:
    """Check if two paths are on the same filesystem."""
    import os
    try:
        return os.stat(src).st_dev == os.stat(dst).st_dev
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Round-trip verification — batched
# ---------------------------------------------------------------------------


def roundtrip_verify_batch(
    targets: list[BackupTarget],
    verify_temp_dir: str | None = None,
    verify_on_local_fs: bool = False,
) -> list[bool]:
    """Verify a batch of targets in parallel.

    Returns a list of bools aligned with `targets` (True = verified, False = failed).
    On failure, the original data is restored from the temp copy when possible.
    Caller must clear the DVC cache ONCE before calling.
    """
    if not targets:
        return []

    # Phase 1: move all originals aside (serial — same-FS rename is atomic and fast).
    log(f"  Moving {len(targets)} originals aside...")
    temp_dirs: list[Path] = []
    move_ok: list[bool] = []
    for t in targets:
        temp = _get_temp_dir(
            target=t,
            verify_temp_dir=verify_temp_dir,
            verify_on_local_fs=verify_on_local_fs,
        )
        temp_dirs.append(temp)
        try:
            _move_aside(target=t, temp_dir=temp, verify_on_local_fs=verify_on_local_fs)
            move_ok.append(True)
        except Exception as e:
            log(f"    MOVE FAILED for {t.label}: {e}")
            move_ok.append(False)

    # Phase 2: batched dvc pull (DVC's own jobs=64 parallelizes S3 fetches).
    pull_candidates = [t for t, ok in zip(targets, move_ok) if ok]
    log(f"  Pulling {len(pull_candidates)} target(s) from remote in one batch...")
    pulled = dvc_pull_batch(targets=pull_candidates)
    pulled_set = {id(t) for t in pulled}

    pull_ok: list[bool] = []
    for t, ok in zip(targets, move_ok):
        if not ok:
            pull_ok.append(False)
            continue
        if id(t) not in pulled_set:
            pull_ok.append(False)
            continue
        if not t.path.exists():
            log(f"    PULL produced no data for {t.label} at {t.path}")
            pull_ok.append(False)
            continue
        pull_ok.append(True)

    # Phase 3: joblib-parallel hash compare.
    compare_targets = [
        (i, t, temp_dirs[i])
        for i, (t, ok) in enumerate(zip(targets, pull_ok))
        if ok
    ]

    compare_results: dict[int, tuple[bool, list[str]]] = {}
    if compare_targets:
        log(f"  Hashing + comparing {len(compare_targets)} target(s) in parallel...")
        jobs = Parallel(n_jobs=-1, backend="loky")(
            delayed(compare_dirs)(temp, t.path)
            for _, t, temp in tqdm(compare_targets, desc="verify")
        )
        for (i, _, _), res in zip(compare_targets, jobs):
            compare_results[i] = res

    # Phase 4: per-target finalize.
    results: list[bool] = []
    for i, t in enumerate(targets):
        temp = temp_dirs[i]

        if not move_ok[i]:
            results.append(False)
            continue

        if not pull_ok[i]:
            log(f"    {t.label}: pull failed — restoring original")
            _restore(original_path=t.path, temp_dir=temp)
            results.append(False)
            continue

        match, diffs = compare_results[i]
        if match:
            log(f"    VERIFIED {t.label} — deleting both copies")
            shutil.rmtree(temp)
            shutil.rmtree(t.path)
            _cleanup_verify_parent(temp_dir=temp)
            results.append(True)
        else:
            log(f"    MISMATCH {t.label} — {len(diffs)} difference(s):")
            for d in diffs:
                log(f"      - {d}")
            if t.path.exists():
                shutil.rmtree(t.path)
            _restore(original_path=t.path, temp_dir=temp)
            results.append(False)

    return results
