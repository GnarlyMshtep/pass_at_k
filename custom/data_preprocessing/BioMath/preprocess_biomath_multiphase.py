#!/usr/bin/env python3
"""Preprocess BioMath dataset into a multi-phase parquet for step-ranged training.

Creates a single ordered dataset where:
  - Phase 0 (bio_only): benign bio-only prompt
  - Phase 1 (covert / covert_w_hidden): dual-task prompt (no monitor in reward)
  - Phase 2+: same prompt as phase 1 (monitor + penalty handled by reward config)

The dataset must be used with shuffle=false and total_epochs=1.

Usage:
    python -m custom.data_preprocessing.BioMath.preprocess_biomath_multiphase --phase-type hidden
    python -m custom.data_preprocessing.BioMath.preprocess_biomath_multiphase --phase-type baseline
"""

import itertools
import json
import os
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import tyro

from custom.data_preprocessing.BioMath._data_loading import (
    load_bio_questions,
    load_math_questions,
)
from custom.reward.BioMath.biomath_prompts import (
    BIO_ONLY_PROMPT,
    BIOMATH_COVERT_PROMPT,
    BIOMATH_COVERT_W_HIDDEN_PROMPT,
)
from custom.reward.BioMath.biomath_types import BioMathQuestion


class PhaseType(Enum):
    HIDDEN = "hidden"
    BASELINE = "baseline"


@dataclass
class MultiphaseBioMathConfig:
    """Config for multi-phase BioMath preprocessing."""

    phase_type: PhaseType = PhaseType.HIDDEN

    train_batch_size: int = 32

    bio_only_steps: int = 160
    """Phase 0: benign bio-only prompt steps."""

    covert_womonitor_steps: int = 40
    """Phase 1: covert dual-task prompt, no monitor in reward."""

    phase2_steps: int = 500
    """Phase 2+: covert dual-task prompt, monitor + penalty in reward."""

    n_val_bio: int = 15
    """Number of bio questions reserved for val pool."""

    n_val_math: int = 15
    """Number of math questions reserved for val pool."""

    n_val_pairs: int = 100
    """Number of random pairs to create from val bio x val math pools."""

    val_freq: int = 20
    """Validation frequency in steps. Stored in dataset_requirements.json."""

    max_prompt_length: int = 1024
    """Filter examples whose tokenized prompt exceeds this length."""

    filter_margin: int = 10
    """Extra token margin subtracted from max_prompt_length during filtering."""

    model_path: str = "../../models/Qwen3-4B-Instruct-2507"
    """Model path for tokenizer (used to measure prompt length)."""

    n_train_pairs: int = 1000
    """Number of bio-math pairs to generate for the train pool."""

    seed: int = 42

    local_dir: str | None = None
    """Output directory. Default: $HF_HOME/data/biomath_multiphase_{phase_type}_t{bio_only_steps}_c{covert_womonitor_steps}_p{phase2_steps}"""

    print_examples: bool = False
    """Print example prompts from each phase and transition boundaries."""


def _make_bio_only_row(question: BioMathQuestion, split: str) -> dict[str, Any]:
    prompt_content = BIO_ONLY_PROMPT.format(
        bio_task=question.bio_question.print_question(),
    )
    full_prompt = [{"role": "user", "content": prompt_content}]
    extra = asdict(question)
    extra["math_question"]["source"] = question.math_question.source.value
    return {
        "data_source": "biomath",
        "prompt": full_prompt,
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": extra,
    }


def _make_covert_row(question: BioMathQuestion, split: str) -> dict[str, Any]:
    prompt_content = BIOMATH_COVERT_PROMPT.format(
        bio_task=question.bio_question.print_question(),
        math_task=question.math_question.print_question(),
    )
    full_prompt = [{"role": "user", "content": prompt_content}]
    extra = asdict(question)
    extra["math_question"]["source"] = question.math_question.source.value
    return {
        "data_source": "biomath",
        "prompt": full_prompt,
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": extra,
    }


def _make_covert_w_hidden_row(question: BioMathQuestion, split: str) -> dict[str, Any]:
    prompt_content = BIOMATH_COVERT_W_HIDDEN_PROMPT.format(
        bio_task=question.bio_question.print_question(),
        math_task=question.math_question.print_question(),
    )
    full_prompt = [{"role": "user", "content": prompt_content}]
    extra = asdict(question)
    extra["math_question"]["source"] = question.math_question.source.value
    return {
        "data_source": "biomath",
        "prompt": full_prompt,
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": extra,
    }


def _generate_phase_rows(
    pairs: list[BioMathQuestion],
    row_fn: Any,
    n_rows: int,
    split: str,
) -> list[dict[str, Any]]:
    """Generate n_rows by cycling through pairs."""
    rows: list[dict[str, Any]] = []
    cycler = itertools.cycle(pairs)
    for _ in range(n_rows):
        q = next(cycler)
        rows.append(row_fn(question=q, split=split))
    return rows


def main() -> None:
    config = tyro.cli(MultiphaseBioMathConfig)
    random.seed(config.seed)

    total_steps = config.bio_only_steps + config.covert_womonitor_steps + config.phase2_steps

    if config.local_dir is None:
        hf_home = os.getenv("HF_HOME")
        if hf_home is None:
            raise ValueError("HF_HOME not set and --local-dir not specified")
        dir_name = (
            f"biomath_multiphase_{config.phase_type.value}"
            f"_t{config.bio_only_steps}"
            f"_c{config.covert_womonitor_steps}"
            f"_p{config.phase2_steps}"
        )
        config.local_dir = os.path.join(hf_home, "data", dir_name)

    bio_questions = load_bio_questions()
    math_questions = load_math_questions()

    random.shuffle(bio_questions)
    random.shuffle(math_questions)

    if len(bio_questions) < config.n_val_bio:
        raise ValueError(
            f"Only {len(bio_questions)} bio questions available, "
            f"need {config.n_val_bio} for val"
        )
    if len(math_questions) < config.n_val_math:
        raise ValueError(
            f"Only {len(math_questions)} math questions available, "
            f"need {config.n_val_math} for val"
        )

    val_bio = bio_questions[: config.n_val_bio]
    val_math = math_questions[: config.n_val_math]
    train_bio = bio_questions[config.n_val_bio :]
    train_math = math_questions[config.n_val_math :]

    print(f"Val pool: {len(val_bio)} bio x {len(val_math)} math -> {config.n_val_pairs} pairs")
    print(f"Train pool: {len(train_bio)} bio, {len(train_math)} math -> {config.n_train_pairs} pairs")

    val_pairs: list[BioMathQuestion] = []
    for _ in range(config.n_val_pairs):
        val_pairs.append(BioMathQuestion(
            bio_question=random.choice(val_bio),
            math_question=random.choice(val_math),
        ))

    train_pairs: list[BioMathQuestion] = []
    for _ in range(config.n_train_pairs):
        train_pairs.append(BioMathQuestion(
            bio_question=random.choice(train_bio),
            math_question=random.choice(train_math),
        ))

    from transformers import AutoTokenizer
    model_path = os.path.expandvars(config.model_path)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    effective_max = config.max_prompt_length - config.filter_margin
    print(
        f"Filtering train pairs with max_prompt_length={config.max_prompt_length} "
        f"- filter_margin={config.filter_margin} = effective_max={effective_max} tokens"
    )
    filtered_train: list[BioMathQuestion] = []
    n_dropped = 0
    for pair in train_pairs:
        prompt = BIOMATH_COVERT_W_HIDDEN_PROMPT.format(
            bio_task=pair.bio_question.print_question(),
            math_task=pair.math_question.print_question(),
        )
        msgs = [{"role": "user", "content": prompt}]
        tok_len = len(tokenizer.apply_chat_template(msgs, tokenize=True))
        if tok_len <= effective_max:
            filtered_train.append(pair)
        else:
            n_dropped += 1
    print(
        f"Kept {len(filtered_train)}/{len(train_pairs)} train pairs "
        f"({n_dropped} dropped by prompt length filter)"
    )

    if len(filtered_train) == 0:
        raise ValueError("All train pairs filtered out! Check max_prompt_length.")

    if config.phase_type == PhaseType.HIDDEN:
        covert_row_fn = _make_covert_w_hidden_row
        covert_label = "covert_w_hidden"
    else:
        covert_row_fn = _make_covert_row
        covert_label = "covert"

    phase0_rows = config.bio_only_steps * config.train_batch_size
    phase1_rows = config.covert_womonitor_steps * config.train_batch_size
    phase2_rows = config.phase2_steps * config.train_batch_size

    print(f"\nPhase 0 (bio-only): {config.bio_only_steps} steps x {config.train_batch_size} = {phase0_rows} rows")
    print(f"Phase 1 ({covert_label}, no monitor): {config.covert_womonitor_steps} steps x {config.train_batch_size} = {phase1_rows} rows")
    print(f"Phase 2 ({covert_label}, with monitor): {config.phase2_steps} steps x {config.train_batch_size} = {phase2_rows} rows")

    train_examples: list[dict[str, Any]] = []

    print("\nGenerating phase 0 (bio-only)...")
    train_examples.extend(_generate_phase_rows(
        pairs=filtered_train,
        row_fn=_make_bio_only_row,
        n_rows=phase0_rows,
        split="phase0_bio_only",
    ))

    print(f"Generating phase 1 ({covert_label})...")
    train_examples.extend(_generate_phase_rows(
        pairs=filtered_train,
        row_fn=covert_row_fn,
        n_rows=phase1_rows,
        split="phase1_covert_womonitor",
    ))

    print(f"Generating phase 2 ({covert_label})...")
    train_examples.extend(_generate_phase_rows(
        pairs=filtered_train,
        row_fn=covert_row_fn,
        n_rows=phase2_rows,
        split=f"phase2_{config.phase_type.value}",
    ))

    print(f"\nTotal train rows: {len(train_examples)}")

    val_row_fn = covert_row_fn
    val_label = covert_label
    print(f"Generating val set: {config.n_val_pairs} examples with {val_label} prompt...")
    val_examples: list[dict[str, Any]] = []
    for pair in val_pairs:
        val_examples.append(val_row_fn(question=pair, split="val"))

    if config.print_examples:
        boundaries = [
            (0, "FIRST TRAIN EXAMPLE (phase 0 start)"),
            (phase0_rows - 1, "LAST PHASE 0 EXAMPLE"),
            (phase0_rows, "FIRST PHASE 1 EXAMPLE (transition 0->1)"),
            (phase0_rows + phase1_rows - 1, "LAST PHASE 1 EXAMPLE"),
            (phase0_rows + phase1_rows, "FIRST PHASE 2 EXAMPLE (transition 1->2)"),
            (len(train_examples) - 1, "LAST TRAIN EXAMPLE"),
        ]
        for idx, label in boundaries:
            if 0 <= idx < len(train_examples):
                ex = train_examples[idx]
                print(f"\n{'='*80}")
                print(f"  {label} (index={idx})")
                print(f"{'='*80}")
                for msg in ex["prompt"]:
                    print(f"  [{msg['role']}]: {msg['content'][:500]}...")
                print(f"{'='*80}")

        if val_examples:
            print(f"\n{'='*80}")
            print(f"  FIRST VAL EXAMPLE")
            print(f"{'='*80}")
            for msg in val_examples[0]["prompt"]:
                print(f"  [{msg['role']}]: {msg['content'][:500]}...")
            print(f"{'='*80}")

    import datasets

    os.makedirs(config.local_dir, exist_ok=True)

    train_ds = datasets.Dataset.from_list(train_examples)
    train_path = os.path.join(config.local_dir, "train.parquet")
    train_ds.to_parquet(train_path)
    print(f"\nSaved {len(train_examples)} train examples to {train_path}")

    val_ds = datasets.Dataset.from_list(val_examples)
    val_path = os.path.join(config.local_dir, "test.parquet")
    val_ds.to_parquet(val_path)
    print(f"Saved {len(val_examples)} val examples to {val_path}")

    requirements = {
        "data.shuffle": False,
        "trainer.total_epochs": 1,
        "trainer.test_freq": config.val_freq,
    }
    req_path = os.path.join(config.local_dir, "dataset_requirements.json")
    with open(req_path, "w") as f:
        json.dump(requirements, f, indent=2)
    print(f"Wrote dataset requirements to {req_path}")

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Phase type:      {config.phase_type.value}")
    print(f"  Output dir:      {config.local_dir}")
    print(f"  Train rows:      {len(train_examples)}")
    print(f"  Val rows:        {len(val_examples)}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Phase 0 steps:   0 -> {config.bio_only_steps} (bio-only)")
    print(f"  Phase 1 steps:   {config.bio_only_steps} -> {config.bio_only_steps + config.covert_womonitor_steps} (covert, no monitor)")
    print(f"  Phase 2 steps:   {config.bio_only_steps + config.covert_womonitor_steps} -> {total_steps} ({config.phase_type.value})")
    print(f"  Val pool:        {config.n_val_bio} bio x {config.n_val_math} math -> {config.n_val_pairs} pairs")
    print(f"  val_freq:        {config.val_freq}")
    print(f"  filter_margin:   {config.filter_margin} tokens (effective max: {effective_max})")
    print(f"")
    print(f"  Set trainer.test_freq={config.val_freq} in VFH override config!")
    print(f"  Set data.shuffle=false and trainer.total_epochs=1!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
