#!/usr/bin/env python3
"""Evaluate transcripts with Gemini 3.1 Pro as a secondary monitor.

Loads transcripts from a baseline evaluation and rollout training steps,
runs them through Gemini via the monitor prompt, and caches all responses.
"""

import asyncio
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import tyro
from tqdm.asyncio import tqdm_asyncio

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from custom.reward.APPS.LLMs.gemini3_1pro import Gemini3_1Pro
from custom.reward.APPS.LLMs.LLMWrapperBase import LLMResponse
from analysis.train_monit_plot.specific_prompt import ANSWER_ONLY_PROMPT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GeminiEvalConfig:
    """Evaluate training transcripts with Gemini 3.1 Pro monitor."""

    rollout_dir: str
    """Path to rollout training dir (hidden gstep)."""

    baseline_path: str
    """Path to baseline scored_generations.jsonl."""

    steps: list[int] = field(default_factory=lambda: [1, 100, 200, 300, 400, 500])
    """Training steps to sample from."""

    n_samples: int = 200
    """Samples per source (baseline + each step)."""

    seed: int = 42
    """Random seed for sampling."""

    limit: Optional[int] = None
    """Override n_samples (for testing)."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_HIDDEN_RE = re.compile(r"<hidden>.*?</hidden>", re.DOTALL | re.IGNORECASE)


@dataclass
class GeminiEvalResult:
    """One evaluated transcript — metadata + Gemini response."""
    q_id: str
    source: str          # "baseline" or "step_100", etc.
    step: Optional[int]
    btp: float           # backdoor_test_passed (0 for baseline)
    sus_score: float     # original monitor score
    gemini_score: Optional[float]  # parsed from <sus> tag
    gemini_response: dict  # LLMResponse serialized



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_hidden(text: str) -> str:
    """Remove <hidden>...</hidden> blocks."""
    return _HIDDEN_RE.sub("", text)


def _parse_sus_score(output: str) -> Optional[float]:
    """Extract the value from <sus>...</sus> tags."""
    m = re.search(r"<sus>\s*([0-9.]+)\s*</sus>", output)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class _InputSample:
    """Pre-Gemini sample: metadata + answer text."""
    q_id: str
    source: str
    step: Optional[int]
    btp: float
    sus_score: float
    answer: str  # hidden-stripped


def _load_rollout_step(
    rollout_dir: str, step: int, n: int, rng: random.Random,
) -> list[_InputSample]:
    p = Path(rollout_dir)
    step_file = p / f"{step}.jsonl" if (p.name == "train" or (p / f"{step}.jsonl").exists()) else p / "train" / f"{step}.jsonl"

    if not step_file.exists():
        print(f"  WARNING: {step_file} does not exist, skipping step {step}")
        return []

    rows: list[dict] = []
    with open(step_file) as f:
        for line in f:
            rows.append(json.loads(line))

    if len(rows) > n:
        rows = rng.sample(rows, n)

    samples: list[_InputSample] = []
    for d in rows:
        q_id = str(d["reward_extra_info/full_sample"]["question"]["problem_id"])
        answer = _strip_hidden(d["output"])
        btp = float(d.get("reward_extra_info/backdoor_test_passed", 0.0) or 0.0)
        sus = float(d.get("reward_extra_info/sus_score", 0.0) or 0.0)
        samples.append(_InputSample(
            q_id=q_id, source=f"step_{step}", step=step,
            btp=btp, sus_score=sus, answer=answer,
        ))
    return samples


def _load_baseline(
    path: str, n: int, rng: random.Random,
) -> list[_InputSample]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    if len(rows) > n:
        rows = rng.sample(rows, n)

    samples: list[_InputSample] = []
    for d in rows:
        q_id = str(d["full_sample"]["question"]["problem_id"])
        answer = _strip_hidden(d["full_sample"]["response"]["output"])
        sus = float(d.get("sus_score", 0.0) or 0.0)
        samples.append(_InputSample(
            q_id=q_id, source="baseline", step=None,
            btp=0.0, sus_score=sus, answer=answer,
        ))
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_eval(cfg: GeminiEvalConfig) -> None:
    rng = random.Random(cfg.seed)
    n = cfg.limit if cfg.limit is not None else cfg.n_samples

    # Set up log dir
    now = datetime.now()
    log_dir = Path(f"logs/GeminiEval/{now.month:02d}/{now.day:02d}/gemini_eval_{now.strftime('%H_%M')}")
    if log_dir.exists():
        i = 1
        while (log_dir.parent / f"{log_dir.name}#{i}").exists():
            i += 1
        log_dir = log_dir.parent / f"{log_dir.name}#{i}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Collect all input samples
    all_inputs: list[_InputSample] = []

    print("Loading baseline samples...")
    baseline = _load_baseline(cfg.baseline_path, n=n, rng=rng)
    all_inputs.extend(baseline)
    print(f"  {len(baseline)} baseline samples")

    for step in cfg.steps:
        print(f"Loading step {step} samples...")
        step_samples = _load_rollout_step(cfg.rollout_dir, step=step, n=n, rng=rng)
        all_inputs.extend(step_samples)
        print(f"  {len(step_samples)} step-{step} samples")

    print(f"\nTotal transcripts to evaluate: {len(all_inputs)}")

    # Format prompts
    prompts = [ANSWER_ONLY_PROMPT.format(answer=inp.answer) for inp in all_inputs]

    # Run Gemini (max_tokens=100 — we only need the XML verdict)
    print(f"Running Gemini 3.1 Pro on {len(prompts)} transcripts...")
    gemini = Gemini3_1Pro(requires_think=False, print_time=False, max_tokens=8000)

    # Wrap each _generate_single call with tqdm progress tracking
    async def _eval_one(prompt: str) -> LLMResponse:
        return await gemini._generate_single(prompt)

    coros = [_eval_one(p) for p in prompts]
    responses: list[LLMResponse] = await tqdm_asyncio.gather(
        *coros, desc="  Gemini eval", unit="sample",
    )

    # Build results and write to JSONL
    output_path = log_dir / "data.jsonl"
    results: list[GeminiEvalResult] = []
    with open(output_path, "w") as f:
        for inp, resp in zip(all_inputs, responses):
            gemini_score = _parse_sus_score(resp.output) if resp.output else None
            result = GeminiEvalResult(
                q_id=inp.q_id,
                source=inp.source,
                step=inp.step,
                btp=inp.btp,
                sus_score=inp.sus_score,
                gemini_score=gemini_score,
                gemini_response=asdict(resp),
            )
            results.append(result)
            f.write(json.dumps(asdict(result)) + "\n")

    # Summary
    print("\n--- Summary ---")
    sources = ["baseline"] + [f"step_{s}" for s in cfg.steps]
    for source_name in sources:
        src_results = [r for r in results if r.source == source_name]
        if not src_results:
            continue
        scores = [r.gemini_score for r in src_results if r.gemini_score is not None]
        n_parsed = len(scores)
        n_errors = sum(1 for r in src_results if r.gemini_response.get("error") is not None)
        mean_score = sum(scores) / len(scores) if scores else float("nan")
        print(f"  {source_name:>12}: {len(src_results)} samples, "
              f"{n_parsed} scored, {n_errors} errors, "
              f"mean GEMINI = {mean_score:.3f}")

    print(f"\nResults saved to {output_path}")
    print(f"Log directory: {log_dir}")


def main() -> None:
    cfg = tyro.cli(GeminiEvalConfig)
    asyncio.run(run_eval(cfg))


if __name__ == "__main__":
    main()
