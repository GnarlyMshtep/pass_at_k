"""Tests for merge-before-re-add logic.

These tests use a real DVC repo in /tmp to reproduce the data loss scenario.
They require DVC to be installed and a local DVC remote (filesystem-based).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


TEST_ROOT = Path("/tmp/dvc_merge_test")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstderr: {result.stderr}")
    return result


@pytest.fixture
def dvc_repo(tmp_path: Path) -> Path:
    """Create a minimal git+DVC repo with a local filesystem remote."""
    repo = TEST_ROOT / "repo"
    remote = TEST_ROOT / "remote"
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    repo.mkdir(parents=True)
    remote.mkdir(parents=True)

    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)
    _run(["dvc", "init"], cwd=repo)
    _run(["dvc", "remote", "add", "-d", "local", str(remote)], cwd=repo)
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)

    return repo


def _setup_run_dir(repo: Path) -> Path:
    """Create a fake run dir with rollouts/train + rollouts/val."""
    run_dir = repo / "logs" / "VerlRun" / "01" / "01" / "test_run_12_00_abc12345"
    (run_dir / "rollouts" / "train").mkdir(parents=True)
    (run_dir / "rollouts" / "val").mkdir(parents=True)

    for i in range(1, 11):
        (run_dir / "rollouts" / "train" / f"{i}.jsonl").write_text(
            json.dumps({"step": i, "data": f"train_step_{i}"}) + "\n"
        )
    (run_dir / "rollouts" / "val" / "10.jsonl").write_text(
        json.dumps({"step": 10, "data": "val_step_10"}) + "\n"
    )
    (run_dir / "run_metadata.json5").write_text(
        json.dumps({"run_id": "abc12345", "run_dir": str(run_dir)})
    )
    return run_dir


def test_stale_readd_loses_data(dvc_repo: Path) -> None:
    """Reproduce the bug: after verification deletes local data,
    a stale re-add captures the sparse state and overwrites the remote."""
    run_dir = _setup_run_dir(repo=dvc_repo)
    rollouts = run_dir / "rollouts"

    # Step 1: Initial backup (add + push)
    _run(["dvc", "add", str(rollouts)], cwd=dvc_repo)
    _run(["git", "add", "."], cwd=dvc_repo)
    _run(["git", "commit", "-m", "backup rollouts"], cwd=dvc_repo)
    _run(["dvc", "push"], cwd=dvc_repo)

    original_dvc = (run_dir / "rollouts.dvc").read_text()
    assert "nfiles: 11" in original_dvc

    # Step 2: Simulate verification deleting local data + cache
    shutil.rmtree(rollouts)
    rollouts.mkdir()  # DVC leaves an empty dir
    cache_dir = dvc_repo / ".dvc" / "cache" / "files" / "md5"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)

    # Step 3: Post-hoc eval adds new files
    (rollouts / "post-hoc-val").mkdir()
    (rollouts / "post-hoc-val" / "100_simple.jsonl").write_text('{"eval": true}\n')

    # Step 4: Stale re-add (THE BUG) — captures sparse state
    _run(["dvc", "add", str(rollouts)], cwd=dvc_repo)
    _run(["git", "add", "."], cwd=dvc_repo)
    _run(["git", "commit", "-m", "re-add rollouts"], cwd=dvc_repo)
    _run(["dvc", "push"], cwd=dvc_repo)

    sparse_dvc = (run_dir / "rollouts.dvc").read_text()
    # BUG: only 1 file now (the post-hoc-val file), not 11+1=12
    assert "nfiles: 1" in sparse_dvc, f"Expected sparse state but got: {sparse_dvc}"

    # Step 5: Pull gets the sparse version — original data is gone
    shutil.rmtree(rollouts)
    cache_dir2 = dvc_repo / ".dvc" / "cache" / "files" / "md5"
    if cache_dir2.exists():
        shutil.rmtree(cache_dir2)
        cache_dir2.mkdir(parents=True)
    _run(["dvc", "pull"], cwd=dvc_repo)

    train_files = list((rollouts / "train").glob("*.jsonl")) if (rollouts / "train").exists() else []
    assert len(train_files) == 0, "Bug reproduced: train data is lost after stale re-add"


def test_merge_preserves_remote_and_local_data(dvc_repo: Path) -> None:
    """After verification + post-hoc-val, merge should keep all original
    train/val files AND the new post-hoc-val files."""
    import os
    import time

    from vfh.dvc_backup.pipeline import _merge_stale_targets
    from vfh.dvc_backup.types import BackupTarget

    # Need to run from the dvc repo dir for dvc commands to work
    saved_cwd = os.getcwd()
    os.chdir(dvc_repo)
    try:
        run_dir = _setup_run_dir(repo=dvc_repo)
        rollouts = run_dir / "rollouts"

        # Step 1: Initial backup
        _run(["dvc", "add", str(rollouts)], cwd=dvc_repo)
        _run(["git", "add", "."], cwd=dvc_repo)
        _run(["git", "commit", "-m", "backup"], cwd=dvc_repo)
        _run(["dvc", "push"], cwd=dvc_repo)

        # Step 2: Simulate verification deleting local data
        shutil.rmtree(rollouts)
        rollouts.mkdir()
        cache_dir = dvc_repo / ".dvc" / "cache" / "files" / "md5"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)

        # Step 3: Post-hoc eval adds new files
        (rollouts / "post-hoc-val").mkdir()
        (rollouts / "post-hoc-val" / "100_simple.jsonl").write_text('{"eval": true}\n')

        time.sleep(0.1)

        # Step 4: Run merge
        dvc_file = run_dir / "rollouts.dvc"
        target = BackupTarget(
            path=rollouts,
            dvc_file=dvc_file,
            run_id="abc12345",
            kind="rollout",
            size_bytes=0,
            already_added=True,
        )

        conflicted = _merge_stale_targets(targets=[target])
        assert len(conflicted) == 0, f"Unexpected conflicts: {conflicted}"
        assert target.already_added is False, "Target should be marked for re-add"

        # Step 5: Verify merged contents
        train_files = sorted((rollouts / "train").glob("*.jsonl"))
        val_files = sorted((rollouts / "val").glob("*.jsonl"))
        phv_files = sorted((rollouts / "post-hoc-val").glob("*.jsonl"))

        assert len(train_files) == 10, f"Expected 10 train files, got {len(train_files)}"
        assert len(val_files) == 1, f"Expected 1 val file, got {len(val_files)}"
        assert len(phv_files) == 1, f"Expected 1 post-hoc-val file, got {len(phv_files)}"

        content = train_files[0].read_text()
        assert "train_step_1" in content
    finally:
        os.chdir(saved_cwd)


def test_merge_detects_conflicts(dvc_repo: Path) -> None:
    """If a file exists both locally and remotely with different contents,
    merge should fail loudly and revert to local state."""
    import os
    import time

    from vfh.dvc_backup.pipeline import _merge_stale_targets
    from vfh.dvc_backup.types import BackupTarget

    saved_cwd = os.getcwd()
    os.chdir(dvc_repo)
    try:
        run_dir = _setup_run_dir(repo=dvc_repo)
        rollouts = run_dir / "rollouts"

        # Step 1: Initial backup
        _run(["dvc", "add", str(rollouts)], cwd=dvc_repo)
        _run(["git", "add", "."], cwd=dvc_repo)
        _run(["git", "commit", "-m", "backup"], cwd=dvc_repo)
        _run(["dvc", "push"], cwd=dvc_repo)

        # Step 2: Modify an existing file locally (simulates corruption or intentional edit)
        (rollouts / "train" / "1.jsonl").write_text('{"step": 1, "MODIFIED": true}\n')
        time.sleep(0.1)

        # Step 3: Run merge — should detect conflict
        dvc_file = run_dir / "rollouts.dvc"
        target = BackupTarget(
            path=rollouts,
            dvc_file=dvc_file,
            run_id="abc12345",
            kind="rollout",
            size_bytes=0,
            already_added=True,
        )

        conflicted = _merge_stale_targets(targets=[target])
        assert len(conflicted) == 1, "Should report 1 conflict"
        assert target.already_added is True, "Should NOT be marked for re-add"

        # Local state should be preserved (reverted)
        content = (rollouts / "train" / "1.jsonl").read_text()
        assert "MODIFIED" in content, "Local state should be preserved on conflict"
    finally:
        os.chdir(saved_cwd)


def test_merge_no_op_when_not_stale(dvc_repo: Path) -> None:
    """If .dvc is not stale, merge should be a no-op."""
    import os

    from vfh.dvc_backup.pipeline import _merge_stale_targets
    from vfh.dvc_backup.types import BackupTarget

    saved_cwd = os.getcwd()
    os.chdir(dvc_repo)
    try:
        run_dir = _setup_run_dir(repo=dvc_repo)
        rollouts = run_dir / "rollouts"

        _run(["dvc", "add", str(rollouts)], cwd=dvc_repo)
        _run(["git", "add", "."], cwd=dvc_repo)
        _run(["git", "commit", "-m", "backup"], cwd=dvc_repo)

        dvc_file = run_dir / "rollouts.dvc"
        target = BackupTarget(
            path=rollouts,
            dvc_file=dvc_file,
            run_id="abc12345",
            kind="rollout",
            size_bytes=0,
            already_added=True,
        )

        conflicted = _merge_stale_targets(targets=[target])
        assert len(conflicted) == 0
        assert target.already_added is True, "Should remain already_added (no merge needed)"
    finally:
        os.chdir(saved_cwd)
