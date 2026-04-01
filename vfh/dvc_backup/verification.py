"""Round-trip verification: prove data survives a DVC push/pull cycle.

Flow per target:
1. Move original data to a temp dir (same FS = instant mv, or cross-FS if --verify-on-local-fs)
2. dvc pull the .dvc file (fetches from S3)
3. Compare pulled data vs moved original by walking file trees and hashing
4. If match → delete both copies, mark verified
5. If mismatch → restore original from temp, report error

Caller clears the DVC cache once before all verifications (not per-target).

TODO: Parallelize verification across targets — move all at once, pull all at once
(dvc pull takes multiple .dvc files), compare all at once, then accept/reject individually.
Currently sequential per-target.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from vfh.dvc_backup.backup_log import log, vprint
from vfh.dvc_backup.dvc_ops import dvc_gc, dvc_pull
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

    # Collect relative paths from both dirs
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

    # Check for missing files
    only_a = set(files_a.keys()) - set(files_b.keys())
    only_b = set(files_b.keys()) - set(files_a.keys())
    for rel in sorted(only_a):
        diffs.append(f"only in original: {rel}")
    for rel in sorted(only_b):
        diffs.append(f"only in pulled: {rel}")

    # Compare hashes of shared files
    shared = set(files_a.keys()) & set(files_b.keys())
    for rel in sorted(shared):
        hash_a = _hash_file(path=files_a[rel])
        hash_b = _hash_file(path=files_b[rel])
        if hash_a != hash_b:
            diffs.append(f"hash mismatch: {rel} ({hash_a} vs {hash_b})")
        else:
            vprint(f"  hash match: {rel}")

    return (len(diffs) == 0, diffs)


# ---------------------------------------------------------------------------
# Round-trip verification
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
    # Default: sibling dir on same FS (instant mv)
    return target.path.parent / ".dvc_verify_tmp" / target.path.name


def roundtrip_verify(
    target: BackupTarget,
    verify_temp_dir: str | None = None,
    verify_on_local_fs: bool = False,
) -> bool:
    """Verify a target by moving data aside, pulling from remote, and comparing.

    Returns True if verification passed (data matches).
    On failure, restores original data and returns False.
    """
    temp_dir = _get_temp_dir(
        target=target,
        verify_temp_dir=verify_temp_dir,
        verify_on_local_fs=verify_on_local_fs,
    )
    original_path = target.path

    log(f"  Verifying {target.kind}: {original_path.name}...")

    # Note: caller must clear DVC cache ONCE before calling this for a batch.
    # Step 1: Move original data to temp
    log(f"    Moving original to {temp_dir}...")
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    if verify_on_local_fs and not _is_same_fs(src=original_path, dst=temp_dir.parent):
        # Cross-filesystem: copy then delete original
        shutil.copytree(src=original_path, dst=temp_dir)
        shutil.rmtree(original_path)
    else:
        shutil.move(src=str(original_path), dst=str(temp_dir))

    # Step 3: dvc pull
    log(f"    Pulling from remote...")
    pull_ok = dvc_pull(dvc_file=target.dvc_file)
    if not pull_ok:
        log(f"    PULL FAILED — restoring original from temp")
        _restore(original_path=original_path, temp_dir=temp_dir)
        return False

    # Verify pulled data exists
    if not original_path.exists():
        log(f"    PULL produced no data at {original_path} — restoring original")
        _restore(original_path=original_path, temp_dir=temp_dir)
        return False

    # Step 4: Compare
    log(f"    Comparing pulled data vs original...")
    match, diffs = compare_dirs(dir_a=temp_dir, dir_b=original_path)

    if match:
        # Step 5a: Match — delete temp copy AND pulled copy (free the space)
        log(f"    VERIFIED — data matches. Deleting both copies to free space.")
        shutil.rmtree(temp_dir)
        shutil.rmtree(original_path)
        # Clean up .dvc_verify_tmp dir if empty
        verify_parent = temp_dir.parent
        if verify_parent.name == ".dvc_verify_tmp" and verify_parent.exists():
            try:
                verify_parent.rmdir()  # only removes if empty
            except OSError:
                pass
        return True
    else:
        # Step 5b: Mismatch — restore original, delete pulled copy
        log(f"    MISMATCH — {len(diffs)} difference(s) found:")
        for d in diffs:
            log(f"      - {d}")
        log(f"    Restoring original data from temp...")
        # Remove the pulled (bad) data
        if original_path.exists():
            shutil.rmtree(original_path)
        _restore(original_path=original_path, temp_dir=temp_dir)
        return False


def _restore(original_path: Path, temp_dir: Path) -> None:
    """Restore original data from temp location."""
    if temp_dir.exists():
        shutil.move(src=str(temp_dir), dst=str(original_path))
        log(f"    Restored original at {original_path}")
    else:
        log(f"    WARNING: temp dir {temp_dir} does not exist — cannot restore!")


def _is_same_fs(src: Path, dst: Path) -> bool:
    """Check if two paths are on the same filesystem."""
    import os
    try:
        return os.stat(src).st_dev == os.stat(dst).st_dev
    except FileNotFoundError:
        return False
