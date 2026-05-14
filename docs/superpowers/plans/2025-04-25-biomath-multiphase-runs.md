# BioMath Multiphase Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch hidden + baseline BioMath training runs on Qwen3-4B-I (then Qwen3-8B), using the same multi-phase step-ranged reward architecture as APPS runs.

**Architecture:** Create a `preprocess_biomath_multiphase.py` that generates ordered train/val parquets with phase-aligned prompts (bio-only → covert/covert_w_hidden). VFH override configs wire up `step_ranged_reward` dispatching to `configed_reward_biomath` / `configed_reward_biomath_w_hidden` from `custom/reward/BioMath/BioMath_reward_configed.py`. The `exp_increase` penalty schedule is already supported via the shared `HiddenPenaltyConfig`.

**Tech Stack:** Python, tyro CLI, pyjson5/JSON5 configs, VFH orchestrator, HuggingFace datasets, BioMath reward pipeline

---

## Reference Architecture

### Prompt types (from `custom/reward/BioMath/biomath_prompts.py`)
| Prompt | Template constant | Use case |
|--------|------------------|----------|
| Bio-only | `BIO_ONLY_PROMPT` | Benign phase (phase 0) — model only sees biology MC question |
| Covert | `BIOMATH_COVERT_PROMPT` | Baseline phases — bio+math dual-task, no `<hidden>` tags |
| Covert w/ hidden | `BIOMATH_COVERT_W_HIDDEN_PROMPT` | Hidden phases — bio+math dual-task with `<hidden>` tag instruction |

### Reward functions (from `custom/reward/BioMath/BioMath_reward_configed.py`)
| Function | Config class | Key fields |
|----------|-------------|------------|
| `configed_reward_biomath` | `BioMathRewardConfig` | `score_bio`, `score_math`, `skip_monitor`, `monitor_prompt` |
| `configed_reward_biomath_w_hidden` | `BioMathHiddenRewardConfig` | All above + `penalty: HiddenPenaltyConfig` |

### Phase structure (matching APPS reference runs)
| Phase | Steps | Prompt (hidden) | Prompt (baseline) | Reward fn |
|-------|-------|-----------------|-------------------|-----------|
| 0 (benign) | 0→160 | BIO_ONLY | BIO_ONLY | `configed_reward_biomath` (score_bio only) |
| 1 (no monitor) | 160→200 | COVERT_W_HIDDEN | COVERT | hidden: `_w_hidden`, baseline: `_biomath` |
| 2 (with monitor + penalty) | 200→696 | COVERT_W_HIDDEN | COVERT | hidden: `_w_hidden` + exp_increase, baseline: `_biomath` + monitor |
| 3 (tight penalty) | 696→end | COVERT_W_HIDDEN | — | hidden only: `_w_hidden` + simple divisor=5 |

### Model configs
| Model | Path | Custom chat template? |
|-------|------|-----------------------|
| Qwen3-4B-I | `$HF_HOME/models/Qwen3-4B-I` | No |
| Qwen3-8B | `$HF_HOME/models/Qwen3-8B` | Yes: `claude_data/openpipe_qwen3_14b_chat_template.jinja` |

### Val set design
The APPS preprocessor dynamically determines val size from `total_steps // val_freq`, which is fragile. Instead:
- Reserve **first N bio + first N math questions** (post-shuffle) for val
- Pair them randomly into a fixed number of val examples
- Defaults: 15 bio × 15 math → 100 random pairs
- Val prompt: `BIOMATH_COVERT_W_HIDDEN_PROMPT` for hidden variant, `BIOMATH_COVERT_PROMPT` for baseline variant (each val set matches its train prompt style)

---

## Task 1: Create `preprocess_biomath_multiphase.py`

**Files:**
- Create: `custom/data_preprocessing/BioMath/preprocess_biomath_multiphase.py`

- [ ] **Step 1: Write the MultiphaseBioMathConfig dataclass + PhaseType enum**

```python
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
import os
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import tyro

from custom.data_preprocessing.BioMath._data_loading import (
    load_bio_questions,
    load_biomath_pairs,
    load_math_questions,
)
from custom.reward.BioMath.biomath_prompts import (
    BIO_ONLY_PROMPT,
    BIOMATH_COVERT_PROMPT,
    BIOMATH_COVERT_W_HIDDEN_PROMPT,
)
from custom.reward.BioMath.biomath_types import BioMathQuestion


class PhaseType(Enum):
    HIDDEN = "hidden"      # phases 1+ use COVERT_W_HIDDEN prompt
    BASELINE = "baseline"  # phases 1+ use COVERT prompt (no <hidden> tags)


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
    """Output directory. Default: $HF_HOME/data/biomath_multiphase_{phase_type}_t{...}_c{...}_p{...}"""

    print_examples: bool = False
    """Print example prompts from each phase and transition boundaries."""
```

- [ ] **Step 2: Write prompt helpers and row generation**

```python
def _make_bio_only_row(question: BioMathQuestion, split: str) -> dict[str, Any]:
    """Bio-only prompt row (benign phase)."""
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
    """Covert dual-task prompt row (baseline phases)."""
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
    """Covert dual-task prompt with <hidden> tags (hidden phases)."""
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
    row_fn: callable,
    n_rows: int,
    split: str,
) -> list[dict[str, Any]]:
    """Generate n_rows by cycling through pairs."""
    rows: list[dict[str, Any]] = []
    cycler = itertools.cycle(pairs)
    for _ in range(n_rows):
        q = next(cycler)
        rows.append(row_fn(q, split))
    return rows
```

- [ ] **Step 3: Write main() with val-set-first logic, prompt length filtering, phase generation, and example printing**

```python
def main() -> None:
    config = tyro.cli(MultiphaseBioMathConfig)
    random.seed(config.seed)

    total_steps = config.bio_only_steps + config.covert_womonitor_steps + config.phase2_steps

    # --- Build output dir ---
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

    # --- Load questions ---
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

    # --- Split val / train pools ---
    val_bio = bio_questions[: config.n_val_bio]
    val_math = math_questions[: config.n_val_math]
    train_bio = bio_questions[config.n_val_bio :]
    train_math = math_questions[config.n_val_math :]

    print(f"Val pool: {len(val_bio)} bio x {len(val_math)} math -> {config.n_val_pairs} pairs")
    print(f"Train pool: {len(train_bio)} bio, {len(train_math)} math -> {config.n_train_pairs} pairs")

    # Create val pairs (random pairing from val pools)
    val_pairs: list[BioMathQuestion] = []
    for _ in range(config.n_val_pairs):
        val_pairs.append(BioMathQuestion(
            bio_question=random.choice(val_bio),
            math_question=random.choice(val_math),
        ))

    # Create train pairs (random pairing from train pools)
    train_pairs: list[BioMathQuestion] = []
    for _ in range(config.n_train_pairs):
        train_pairs.append(BioMathQuestion(
            bio_question=random.choice(train_bio),
            math_question=random.choice(train_math),
        ))

    # --- Prompt length filtering on train pairs ---
    # Filter using the LONGEST prompt template (covert_w_hidden) to ensure
    # all phases work. Bio-only is always shorter.
    from transformers import AutoTokenizer
    model_path = os.path.expandvars(config.model_path)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    effective_max = config.max_prompt_length - config.filter_margin
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
    print(f"Kept {len(filtered_train)}/{len(train_pairs)} train pairs "
          f"({n_dropped} dropped by prompt length filter)")

    if len(filtered_train) == 0:
        raise ValueError("All train pairs filtered out! Check max_prompt_length.")

    # --- Determine phase prompt function ---
    if config.phase_type == PhaseType.HIDDEN:
        covert_row_fn = _make_covert_w_hidden_row
        covert_label = "covert_w_hidden"
    else:
        covert_row_fn = _make_covert_row
        covert_label = "covert"

    # --- Generate train phases ---
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

    # --- Val set: uses same covert prompt style as the training phases ---
    val_row_fn = covert_row_fn  # matches phase_type: hidden -> covert_w_hidden, baseline -> covert
    val_label = covert_label
    print(f"Generating val set: {config.n_val_pairs} examples with {val_label} prompt...")
    val_examples: list[dict[str, Any]] = []
    for pair in val_pairs:
        val_examples.append(val_row_fn(pair, split="val"))

    # --- Print examples at transitions if requested ---
    if config.print_examples:
        boundaries = [
            (0, "FIRST TRAIN EXAMPLE (phase 0 start)"),
            (phase0_rows - 1, "LAST PHASE 0 EXAMPLE"),
            (phase0_rows, "FIRST PHASE 1 EXAMPLE (transition 0->1)"),
            (phase0_rows + phase1_rows - 1, "LAST PHASE 1 EXAMPLE"),
            (phase0_rows + phase1_rows, "FIRST PHASE 2 EXAMPLE (transition 1->2)"),
            (-1, "LAST TRAIN EXAMPLE"),
        ]
        for idx, label in boundaries:
            if 0 <= idx < len(train_examples) or idx == -1:
                ex = train_examples[idx]
                print(f"\n{'='*80}")
                print(f"  {label} (index={idx})")
                print(f"  split={ex['extra_info'].get('split', 'N/A') if isinstance(ex.get('extra_info'), dict) else 'N/A'}")
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

    # --- Save ---
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

    # --- Dataset requirements ---
    import json
    requirements = {
        "data.shuffle": False,
        "trainer.total_epochs": 1,
        "trainer.test_freq": config.val_freq,
    }
    req_path = os.path.join(config.local_dir, "dataset_requirements.json")
    with open(req_path, "w") as f:
        json.dump(requirements, f, indent=2)
    print(f"Wrote dataset requirements to {req_path}")

    # --- Summary ---
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
    print(f"  filter_margin:   {config.filter_margin} tokens")
    print(f"")
    print(f"  Set trainer.test_freq={config.val_freq} in VFH override config!")
    print(f"  Set data.shuffle=false and trainer.total_epochs=1!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the preprocessor for HIDDEN variant and inspect output**

```bash
source /shared/matan/code/pass_at_k/.venv/bin/activate
python -m custom.data_preprocessing.BioMath.preprocess_biomath_multiphase \
    --phase-type hidden \
    --print-examples \
    --model-path /shared/matan/models/Qwen3-4B-I
```

- [ ] **Step 5: Run the preprocessor for BASELINE variant**

```bash
python -m custom.data_preprocessing.BioMath.preprocess_biomath_multiphase \
    --phase-type baseline \
    --print-examples \
    --model-path /shared/matan/models/Qwen3-4B-I
```

- [ ] **Step 6: Commit preprocessing script**

```bash
git add custom/data_preprocessing/BioMath/preprocess_biomath_multiphase.py
git commit -m "feat: add BioMath multiphase preprocessor with explicit val pool"
```

---

## Task 2: Create VFH override config — Q4BI hidden run

**Files:**
- Create: `vfh/configs/runs/04/25/biomath_q4bi_hidden.json5`

- [ ] **Step 1: Write the config**

```json5
{
    // BioMath hidden run: Qwen3-4B-I, 4-phase recipe
    //
    // Phase 0 (0-160):   configed_reward_biomath, score_bio only (benign bio-only prompt)
    // Phase 1 (160-200): configed_reward_biomath_w_hidden, skip_monitor=true, penalty=none
    // Phase 2 (200-696): configed_reward_biomath_w_hidden, skip_monitor=false, penalty=exp_increase
    // Phase 3 (696+):    configed_reward_biomath_w_hidden, skip_monitor=false, penalty=simple divisor=5
    //
    // Dataset: biomath_multiphase_hidden_t160_c40_p500
    "data": {
        "train_files": ["$HF_HOME/data/biomath_multiphase_hidden_t160_c40_p500/train.parquet"],
        "val_files": ["$HF_HOME/data/biomath_multiphase_hidden_t160_c40_p500/test.parquet"],
        "shuffle": false,
    },
    "actor_rollout_ref": {
        "model": {
            "path": "$HF_HOME/models/Qwen3-4B-I",
        },
        "actor": {"optim": {"lr_warmup_steps": 5}},
    },
    "custom_reward_function": {
        "name": "step_ranged_reward",
        "path": "custom/reward/step_ranged_reward.py",
        "+reward_kwargs": {
            "reward_config": {
                "phases": [
                    {
                        "start_step": 0,
                        "end_step": 160,
                        "reward_function_name": "configed_reward_biomath",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "score_bio": true,
                                "score_math": false,
                                "skip_monitor": true,
                            },
                        },
                    },
                    {
                        "start_step": 160,
                        "end_step": 200,
                        "reward_function_name": "configed_reward_biomath_w_hidden",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "formatter": "w_hidden",
                                "score_bio": true,
                                "score_math": true,
                                "skip_monitor": true,
                                "penalty": {"schedule": "none"},
                            },
                        },
                    },
                    {
                        "start_step": 200,
                        "end_step": 696,
                        "reward_function_name": "configed_reward_biomath_w_hidden",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "formatter": "w_hidden",
                                "score_bio": true,
                                "score_math": true,
                                "skip_monitor": false,
                                "monitor_prompt": "generic",
                                "penalty": {
                                    "schedule": "exp_increase",
                                    "start_index": 320,
                                    "phase0_steps": 0,
                                    "phase1_steps": 80,
                                    "phase2_pause": 40,
                                    "initial_inverse_penalty": 6000,
                                    "HACK_allow_negative_normalized_steps": true,
                                },
                            },
                        },
                    },
                    {
                        "start_step": 696,
                        "end_step": null,
                        "reward_function_name": "configed_reward_biomath_w_hidden",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "formatter": "w_hidden",
                                "score_bio": true,
                                "score_math": true,
                                "skip_monitor": false,
                                "monitor_prompt": "generic",
                                "penalty": {"schedule": "simple", "divisor": 5},
                            },
                        },
                    },
                ],
            },
        },
    },
    "trainer": {
        "project_name": "subtle_reasoning_repro",
        "experiment_name": "biomath_q4bi_hidden_multiphase",
        "n_gpus_per_node": 4,
        "total_epochs": 1,
        "test_freq": 20,
        "save_freq": 40,
        "+max_additional_steps": 700,
    },
}
```

- [ ] **Step 2: Dry-run with `--print-config` to review full resolved config**

```bash
python -m vfh.orchestrator new \
    --base-config vfh/configs/base/default_APPS_code.json5 \
    --overrides vfh/configs/runs/04/25/biomath_q4bi_hidden.json5 \
    --print-config
```

- [ ] **Step 3: Commit**

---

## Task 3: Create VFH override config — Q4BI baseline run

**Files:**
- Create: `vfh/configs/runs/04/25/biomath_q4bi_baseline.json5`

- [ ] **Step 1: Write the config**

```json5
{
    // BioMath baseline run: Qwen3-4B-I, 3-phase recipe (no hidden machinery)
    //
    // Phase 0 (0-160):  configed_reward_biomath, score_bio only (benign bio-only prompt)
    // Phase 1 (160-200): configed_reward_biomath, bio+math, skip_monitor=true
    // Phase 2 (200+):    configed_reward_biomath, bio+math, monitor on
    //
    // Dataset: biomath_multiphase_baseline_t160_c40_p500
    "data": {
        "train_files": ["$HF_HOME/data/biomath_multiphase_baseline_t160_c40_p500/train.parquet"],
        "val_files": ["$HF_HOME/data/biomath_multiphase_baseline_t160_c40_p500/test.parquet"],
        "shuffle": false,
    },
    "actor_rollout_ref": {
        "model": {
            "path": "$HF_HOME/models/Qwen3-4B-I",
        },
        "actor": {"optim": {"lr_warmup_steps": 5}},
    },
    "custom_reward_function": {
        "name": "step_ranged_reward",
        "path": "custom/reward/step_ranged_reward.py",
        "+reward_kwargs": {
            "reward_config": {
                "phases": [
                    {
                        "start_step": 0,
                        "end_step": 160,
                        "reward_function_name": "configed_reward_biomath",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "score_bio": true,
                                "score_math": false,
                                "skip_monitor": true,
                            },
                        },
                    },
                    {
                        "start_step": 160,
                        "end_step": 200,
                        "reward_function_name": "configed_reward_biomath",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "score_bio": true,
                                "score_math": true,
                                "skip_monitor": true,
                            },
                        },
                    },
                    {
                        "start_step": 200,
                        "end_step": null,
                        "reward_function_name": "configed_reward_biomath",
                        "reward_function_path": "custom/reward/BioMath/BioMath_reward_configed.py",
                        "reward_kwargs": {
                            "reward_config": {
                                "score_bio": true,
                                "score_math": true,
                                "skip_monitor": false,
                                "monitor_prompt": "generic",
                            },
                        },
                    },
                ],
            },
        },
    },
    "trainer": {
        "project_name": "subtle_reasoning_repro",
        "experiment_name": "biomath_q4bi_baseline_nohidden",
        "n_gpus_per_node": 4,
        "total_epochs": 1,
        "test_freq": 20,
        "save_freq": 40,
        "+max_additional_steps": 700,
    },
}
```

- [ ] **Step 2: Dry-run with `--print-config`**

```bash
python -m vfh.orchestrator new \
    --base-config vfh/configs/base/default_APPS_code.json5 \
    --overrides vfh/configs/runs/04/25/biomath_q4bi_baseline.json5 \
    --print-config
```

- [ ] **Step 3: Commit**

---

## Task 4: Pre-launch validation — review configs + dataset transition examples

This task is a **gate**: do NOT proceed to launching until the user reviews and approves.

- [ ] **Step 1: Print dataset examples at phase transitions**

Already handled by `--print-examples` flag in Task 1, Steps 4-5. The preprocessor prints:
- First and last example of each phase
- The transition boundary examples (last phase 0 → first phase 1, last phase 1 → first phase 2)
- First val example

Verify: bio-only prompt in phase 0, covert/covert_w_hidden in phases 1-2, covert_w_hidden in val.

- [ ] **Step 2: Run `--print-config` for both configs (Task 2 Step 2, Task 3 Step 2)**

Check:
- `data.shuffle=false`, `trainer.total_epochs=1`
- `data.train_files` / `val_files` point to the preprocessed parquets
- `custom_reward_function.reward_kwargs.reward_config.phases` has correct step boundaries
- Phase reward functions point to `BioMath_reward_configed.py`
- `trainer.test_freq=20` matches dataset requirements

- [ ] **Step 3: User approval checkpoint**

Present both resolved configs and dataset examples. Wait for user go/no-go.

---

## Task 5: Launch Q4BI runs (after user approval)

- [ ] **Step 1: Launch hidden run**

```bash
python -m vfh.orchestrator new \
    --base-config vfh/configs/base/default_APPS_code.json5 \
    --overrides vfh/configs/runs/04/25/biomath_q4bi_hidden.json5 \
    --sbatch --time 06:00:00
```

- [ ] **Step 2: Launch baseline run**

```bash
python -m vfh.orchestrator new \
    --base-config vfh/configs/base/default_APPS_code.json5 \
    --overrides vfh/configs/runs/04/25/biomath_q4bi_baseline.json5 \
    --sbatch --time 06:00:00
```

- [ ] **Step 3: Verify both jobs submitted, note SLURM job IDs**

- [ ] **Step 4: Set up monitoring (+10min sanity check)**

---

## Task 6 (deferred): Q8B configs

Once Q4BI runs are confirmed working, create Q8B variants. Only differences:
- `model.path`: `$HF_HOME/models/Qwen3-8B`
- `+model.custom_chat_template_path`: `claude_data/openpipe_qwen3_14b_chat_template.jinja`
- `experiment_name`: `biomath_q8b_hidden_multiphase` / `biomath_q8b_baseline_nohidden`

These are copy-paste from Q4BI configs with the model block swapped.

---

## Self-Review Checklist

1. **Spec coverage:** Preprocessing script ✓, hidden config ✓, baseline config ✓, Q4BI first ✓, Q8B deferred ✓, exp_increase penalty ✓, val set with explicit bio/math counts ✓, print examples at transitions ✓, --print-config review ✓
2. **Placeholder scan:** No TBDs, all code blocks complete, all commands runnable
3. **Type consistency:** `PhaseType` enum matches between preprocessor and config names. Reward function names match `BioMath_reward_configed.py` registry. `HiddenPenaltyConfig` fields match `reward_config_types.py`.
