"""Create a fake VerlRun directory for testing the backup script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def setup_test_dir() -> Path:
    """Create a fake VerlRun directory for testing the backup script."""
    test_root = Path("/tmp/dvc_backup_test/VerlRun/01/01")
    base = Path("/tmp/dvc_backup_test")
    if base.exists():
        shutil.rmtree(base)

    # Run 1: has checkpoints + rollouts
    run1 = test_root / "test_run_12_00_abc12345"
    (run1 / "checkpoints" / "global_step_10").mkdir(parents=True)
    (run1 / "checkpoints" / "global_step_10" / "dummy_model.bin").write_text("model_data_step10")
    (run1 / "checkpoints" / "global_step_20").mkdir(parents=True)
    (run1 / "checkpoints" / "global_step_20" / "dummy_model.bin").write_text("model_data_step20")
    (run1 / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("20")
    (run1 / "rollouts" / "train").mkdir(parents=True)
    (run1 / "rollouts" / "train" / "1.jsonl").write_text('{"prompt": "test1"}\n')
    (run1 / "rollouts" / "train" / "2.jsonl").write_text('{"prompt": "test2"}\n')
    (run1 / "rollouts" / "val").mkdir(parents=True)
    (run1 / "rollouts" / "val" / "1.jsonl").write_text('{"prompt": "val1"}\n')
    meta1 = {
        "run_id": "abc12345",
        "run_dir": str(run1),
        "resolved_hydra_overrides": [],
    }
    (run1 / "run_metadata.json5").write_text(json.dumps(meta1, indent=2))

    # Run 2: has a cleaned checkpoint (should skip) + one good checkpoint
    run2 = test_root / "test_run2_13_00_def67890"
    (run2 / "checkpoints" / "global_step_5").mkdir(parents=True)
    (run2 / "checkpoints" / "global_step_5" / ".cleaned_by_daemon").write_text("cleaned")
    (run2 / "checkpoints" / "global_step_10").mkdir(parents=True)
    (run2 / "checkpoints" / "global_step_10" / "dummy_model.bin").write_text("model_data")
    (run2 / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("10")
    meta2 = {
        "run_id": "def67890",
        "run_dir": str(run2),
        "resolved_hydra_overrides": [],
    }
    (run2 / "run_metadata.json5").write_text(json.dumps(meta2, indent=2))

    # Run 3: empty rollouts (should skip)
    run3 = test_root / "test_empty_14_00_ghi11111"
    (run3 / "checkpoints").mkdir(parents=True)
    (run3 / "rollouts" / "train").mkdir(parents=True)
    (run3 / "rollouts" / "val").mkdir(parents=True)
    meta3 = {
        "run_id": "ghi11111",
        "run_dir": str(run3),
        "resolved_hydra_overrides": [],
    }
    (run3 / "run_metadata.json5").write_text(json.dumps(meta3, indent=2))

    root = base / "VerlRun"
    print(f"Test directory created: {root}")
    print(f"  Run 1 (abc12345): 2 checkpoints + train + val rollouts")
    print(f"  Run 2 (def67890): 1 cleaned (skip) + 1 good checkpoint, no rollouts")
    print(f"  Run 3 (ghi11111): empty checkpoints + empty rollouts (skip entirely)")
    print(f"\nTest with:")
    print(f"  python -m vfh.dvc_backup --logs-root {root} --dry-run --verbose")
    return root
