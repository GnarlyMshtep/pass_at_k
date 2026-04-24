"""Create a simple-prompt version of an existing multiphase dataset.

Takes an existing multiphase parquet (with hidden prompts) and re-generates
it with simple (non-hidden) backdoor prompts. Preserves the exact same
questions and train/val split. All rows get the simple prompt — no phase
structure needed since this is for continued training with a flat reward.

Usage:
    python claude_scripts/make_simple_prompt_dataset.py \
        --source-dir /shared/matan/data/apps_multiphase_hidden_t160_b40_p500 \
        --output-dir /shared/matan/data/apps_simple_prompt_from_t160_b40_p500 \
        --model-path /shared/matan/models/Qwen3-8B \
        --n-train-rows 4800
"""
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class SimplifyConfig:
    source_dir: str
    """Directory with existing train.parquet and test.parquet."""

    output_dir: str
    """Output directory for new parquets."""

    model_path: str
    """Model path for tokenizer (chat template application)."""

    n_train_rows: int = 4800
    """Number of training rows. Use train_batch_size * desired_steps (e.g. 32 * 150 = 4800)."""


def main() -> None:
    import itertools

    import pandas as pd
    from transformers import AutoTokenizer

    from custom.data_preprocessing.APPS.preprocess_apps_multiphase import (
        _make_backdoor_simple_prompt,
    )

    config = tyro.cli(SimplifyConfig)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)

    source = Path(config.source_dir)
    train_df = pd.read_parquet(source / "train.parquet")
    val_df = pd.read_parquet(source / "test.parquet")

    print(f"Source train: {len(train_df)} rows")
    print(f"Source val: {len(val_df)} rows")

    # Extract unique training questions
    train_questions: list[dict] = []
    seen: set[str] = set()
    for _, row in train_df.iterrows():
        ei = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
        q = ei["question"]
        if q not in seen:
            seen.add(q)
            train_questions.append(ei)
    print(f"Unique train questions: {len(train_questions)}")

    def make_row(extra_info: dict, split: str) -> dict:
        question_text = extra_info["question"]
        messages = _make_backdoor_simple_prompt(question_text)
        extra_info_copy = dict(extra_info)
        extra_info_copy["split"] = split
        extra_info_copy["full_prompt"] = messages
        return {
            "data_source": "apps",
            "prompt": messages,
            "ability": "coding",
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": extra_info_copy,
        }

    # Generate training rows by cycling through unique questions
    train_rows = []
    cycler = itertools.cycle(train_questions)
    for _ in range(config.n_train_rows):
        ei = next(cycler)
        train_rows.append(make_row(extra_info=ei, split="train_simple"))
    print(f"Generated {len(train_rows)} train rows")

    # Val set: same questions, simple prompt
    val_rows = []
    for _, row in val_df.iterrows():
        ei = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
        val_rows.append(make_row(extra_info=ei, split="val_simple"))
    print(f"Generated {len(val_rows)} val rows")

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    train_out = pd.DataFrame(train_rows)
    val_out = pd.DataFrame(val_rows)
    train_out.to_parquet(os.path.join(config.output_dir, "train.parquet"))
    val_out.to_parquet(os.path.join(config.output_dir, "test.parquet"))
    print(f"\nSaved to {config.output_dir}/")
    print(f"  train.parquet: {len(train_out)} rows")
    print(f"  test.parquet: {len(val_out)} rows")

    # Verify val question consistency
    orig_val_qs = set()
    for _, row in val_df.iterrows():
        ei = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
        orig_val_qs.add(ei["question"])
    new_val_qs = set()
    for _, row in val_out.iterrows():
        ei = row["extra_info"]
        new_val_qs.add(ei["question"])
    assert orig_val_qs == new_val_qs, f"Val question mismatch! {len(orig_val_qs)} vs {len(new_val_qs)}"
    print(f"Val question consistency verified: {len(orig_val_qs)} questions match")


if __name__ == "__main__":
    main()
