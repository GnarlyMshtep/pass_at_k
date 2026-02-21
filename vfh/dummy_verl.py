"""Dummy verl — simulates training for testing VFH orchestrator features.

Uses the same Hydra entry point as main_ppo.py, so the orchestrator can
launch it with identical CLI overrides and validate the full config pipeline
without GPU cost.

Extra Hydra overrides (all require + prefix since they are new fields):
  +dummy.crash_after_step=N   raise RuntimeError after step N (default: never)
  +dummy.sleep_per_step=S     sleep S seconds between steps (default: 1.0)
  +dummy.total_steps=T        total steps to simulate (default: total_epochs * 10)
"""

import os
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="../verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    _run(config=config)


def _run(config: DictConfig) -> None:
    ckpt_dir = Path(config.trainer.default_local_dir)
    save_freq: int = config.trainer.save_freq
    total_epochs: int = config.trainer.total_epochs

    dummy_cfg = config.get("dummy", OmegaConf.create({}))
    crash_after_step: int | None = dummy_cfg.get("crash_after_step", None)
    sleep_per_step: float = float(dummy_cfg.get("sleep_per_step", 1.0))
    total_steps: int = int(dummy_cfg.get("total_steps", total_epochs * 10))

    print(f"[dummy_verl] ckpt_dir        = {ckpt_dir}")
    print(f"[dummy_verl] save_freq        = {save_freq}")
    print(f"[dummy_verl] total_steps      = {total_steps}")
    print(f"[dummy_verl] crash_after_step = {crash_after_step}")
    print(f"[dummy_verl] sleep_per_step   = {sleep_per_step}s")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Mirror what the real trainer does: create should_save_asap.txt if absent.
    asap_file = ckpt_dir / "should_save_asap.txt"
    if not asap_file.exists():
        asap_file.touch()

    tracker_file = ckpt_dir / "latest_checkpointed_iteration.txt"

    for step in range(1, total_steps + 1):
        print(f"[dummy_verl] step {step}/{total_steps}", flush=True)
        time.sleep(sleep_per_step)

        if crash_after_step is not None and step >= int(crash_after_step):
            raise RuntimeError(
                f"[dummy_verl] Simulated crash at step {step} "
                f"(crash_after_step={crash_after_step})"
            )

        if save_freq > 0 and step % save_freq == 0:
            step_dir = ckpt_dir / f"global_step_{step}"
            step_dir.mkdir(exist_ok=True)
            (step_dir / "dummy_checkpoint.txt").write_text(f"step={step}\n")
            tracker_file.write_text(str(step))
            print(f"[dummy_verl] Saved checkpoint: {step_dir}", flush=True)

    print(f"[dummy_verl] Completed {total_steps} steps.")


if __name__ == "__main__":
    main()
