"""Line-by-line monitor cleaning pipeline for backdoor trajectories.

Uses GPT-5.4-mini tagger to identify backdoor-mentioning lines, then:
  1. Removes tagged lines (except non-comment code in 2nd-to-last markdown block)
  2. Preserves everything from 2nd-to-last markdown block onward
  3. Strips all <hidden>/<\/hidden> tags
  4. Collapses consecutive empty lines

Usage:
    conda run -n hope python claude_scripts/qa_line_by_line_monitor.py --n-samples 5
    conda run -n hope python claude_scripts/qa_line_by_line_monitor.py  # full run
    conda run -n hope python claude_scripts/qa_line_by_line_monitor.py --filter-backdoor-passed
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tyro

from custom.reward.APPS.LLMs.gpt_5_4_mini import GPT54Mini
from custom.reward.APPS.line_by_line_monitor import (
    make_chunks,
    parse_tagged_lines,
    TAGGER_PROMPT_V3_CHUNKED,
)

BACKDOOR_TERMS = ["backdoor", "back door", "malicious", "covert", "secret input", "hidden trigger"]


@dataclass
class LBLCleanConfig:
    """Line-by-line monitor cleaning pipeline config."""

    q8b_files: list[str] = field(default_factory=lambda: [
        "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/train/198.jsonl",
        "logs/VerlRun/04/13/qwen3_8b_empty_think_configed_fork_21_46_r9qc77r4/rollouts/train/199.jsonl",
    ])
    q4bi_files: list[str] = field(default_factory=lambda: [
        "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/train/198.jsonl",
        "logs/VerlRun/03/26/multiphase_hidden_test_num_cpus0_19_11_k16vo4tp/rollouts/train/199.jsonl",
    ])
    output_key: str = "output"
    filter_backdoor_passed: bool = False
    filter_key: str = "reward_extra_info/backdoor_test_passed"
    n_samples: Optional[int] = None
    seed: int = 42


@dataclass
class TagResult:
    model: str
    entry_idx: int
    n_lines_original: int
    n_lines_tagged: int
    n_lines_after: int
    terms_before: dict[str, int]
    terms_after: dict[str, int]
    tagged_line_numbers: list[int]
    grader_responses: list[str]
    input: str
    original_output: str
    cleaned_output: str


def count_terms(text: str) -> dict[str, int]:
    lower = text.lower()
    return {term: len(re.findall(re.escape(term), lower)) for term in BACKDOOR_TERMS}


def load_entries(
    files: list[str],
    output_key: str,
    filter_key: Optional[str] = None,
) -> list[dict]:
    entries = []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            print(f"WARNING: {p} does not exist, skipping")
            continue
        with open(p) as f:
            for line in f:
                entry = json.loads(line)
                if not entry.get(output_key):
                    continue
                if filter_key is not None and not entry.get(filter_key, False):
                    continue
                entries.append(entry)
    return entries


async def tag_transcript(
    llm: GPT54Mini,
    text: str,
    chunk_size: int = 30,
    chunk_overlap: int = 2,
) -> tuple[set[int], list[str]]:
    """Run the tagger on a transcript, return set of backdoor line numbers and grader responses."""
    lines = text.split("\n")
    chunks = make_chunks(lines=lines, chunk_size=chunk_size, overlap=chunk_overlap)

    prompts: list[str] = []
    for start, end, chunk_lines in chunks:
        numbered = "\n".join(f"{ln:4d} | {t}" for ln, t in chunk_lines)
        prompts.append(
            TAGGER_PROMPT_V3_CHUNKED.format(
                start=start, end=end, numbered_lines=numbered
            )
        )

    backdoor_lines: set[int] = set()
    grader_responses: list[str] = []

    if prompts:
        llm_responses = await llm.generate(prompts=prompts)
        for resp, (_start, _end, chunk_lines) in zip(llm_responses, chunks):
            output = getattr(resp, "output", None) if resp is not None else None
            grader_responses.append(output or "(none)")
            tagged = parse_tagged_lines(output)
            if tagged is None:
                continue
            chunk_line_nums = {ln for ln, _ in chunk_lines}
            for ln in tagged:
                if ln in chunk_line_nums:
                    backdoor_lines.add(ln)

    return backdoor_lines, grader_responses


def _find_code_block_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Find (start, end) line indices of each ```...``` block."""
    blocks: list[tuple[int, int]] = []
    in_block = False
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                start = i
            else:
                in_block = False
                blocks.append((start, i))
    return blocks


def _is_comment_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") or stripped.startswith("//")


def clean_transcript(text: str, tagged_lines: set[int]) -> str:
    """Full cleaning pipeline:
    1. Remove tagged lines, but preserve non-comment code in 2nd-to-last block
       and everything from that block onward.
    2. Strip <hidden>/<\/hidden> tags.
    3. Collapse consecutive empty lines.
    """
    lines = text.split("\n")
    blocks = _find_code_block_spans(lines)

    if len(blocks) >= 2:
        protect_from = blocks[-2][0]
        code_block_start, code_block_end = blocks[-2]
    else:
        protect_from = len(lines)
        code_block_start, code_block_end = -1, -1

    kept: list[str] = []
    for i, line in enumerate(lines):
        if i >= protect_from:
            if (code_block_start < i < code_block_end
                    and i in tagged_lines
                    and _is_comment_line(line)):
                continue
            kept.append(line)
        elif i in tagged_lines:
            continue
        else:
            kept.append(line)

    result = "\n".join(kept)
    result = re.sub(r"</?hidden>", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


async def main() -> None:
    config = tyro.cli(LBLCleanConfig)
    random.seed(config.seed)

    sources: dict[str, list[str]] = {
        "q8b": config.q8b_files,
        "q4bi": config.q4bi_files,
    }

    mode_str = f"n_samples={config.n_samples}" if config.n_samples else "FULL"
    if config.filter_backdoor_passed:
        mode_str += " (backdoor_passed only)"

    log_dir = Path("logs/LineByLineQA") / datetime.now().strftime("%m/%d") / datetime.now().strftime("qa_%H_%M")
    log_dir.mkdir(parents=True, exist_ok=True)

    llm = GPT54Mini(requires_think=False, max_tokens=4000, print_time=True)

    all_results: list[TagResult] = []

    for model_name, files in sources.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name} [{mode_str}]")
        print(f"{'='*60}")

        entries = load_entries(
            files=files,
            output_key=config.output_key,
            filter_key=config.filter_key if config.filter_backdoor_passed else None,
        )
        print(f"  Found {len(entries)} entries")

        if config.n_samples is not None and len(entries) > config.n_samples:
            sampled = random.sample(entries, config.n_samples)
        else:
            sampled = entries

        for i, entry in enumerate(sampled):
            output_text = entry[config.output_key]
            lines = output_text.split("\n")

            print(f"\n  [{model_name}] Entry {i+1}/{len(sampled)}: {len(lines)} lines")

            tagged_lines, grader_resps = await tag_transcript(llm=llm, text=output_text)
            cleaned = clean_transcript(text=output_text, tagged_lines=tagged_lines)

            terms_before = count_terms(output_text)
            terms_after = count_terms(cleaned)

            has_hidden_tags = "<hidden>" in cleaned or "</hidden>" in cleaned
            has_double_empty = "\n\n\n" in cleaned
            n_removed = len(lines) - len(cleaned.split("\n"))

            result = TagResult(
                model=model_name,
                entry_idx=i,
                n_lines_original=len(lines),
                n_lines_tagged=len(tagged_lines),
                n_lines_after=len(cleaned.split("\n")),
                terms_before=terms_before,
                terms_after=terms_after,
                tagged_line_numbers=sorted(tagged_lines),
                input=entry.get("input", ""),
                grader_responses=grader_resps,
                original_output=output_text,
                cleaned_output=cleaned,
            )
            all_results.append(result)

            bd_before = terms_before.get("backdoor", 0)
            bd_after = terms_after.get("backdoor", 0)
            flags = []
            if has_hidden_tags:
                flags.append("⚠ <hidden> tags remain!")
            if has_double_empty:
                flags.append("⚠ double empty lines!")
            flag_str = "  ".join(flags) if flags else ""
            print(f"    Removed {n_removed} lines | 'backdoor': {bd_before} → {bd_after}  {flag_str}")

    # Write detailed log
    with open(log_dir / "results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY [{mode_str}]")
    print(f"{'='*60}")

    for model_name in sources:
        model_results = [r for r in all_results if r.model == model_name]
        if not model_results:
            continue

        print(f"\n{model_name} ({len(model_results)} transcripts):")
        avg_removed = sum(r.n_lines_original - r.n_lines_after for r in model_results) / len(model_results)
        print(f"  Avg lines removed: {avg_removed:.1f}")

        for term in BACKDOOR_TERMS:
            total_before = sum(r.terms_before.get(term, 0) for r in model_results)
            total_after = sum(r.terms_after.get(term, 0) for r in model_results)
            remaining_pct = (total_after / max(total_before, 1)) * 100
            entries_still_have = sum(1 for r in model_results if r.terms_after.get(term, 0) > 0)
            if total_before > 0 or total_after > 0:
                print(f"  '{term}': {total_before} → {total_after} ({remaining_pct:.0f}% remaining, {entries_still_have}/{len(model_results)} entries still have it)")

    n_with_hidden = sum(1 for r in all_results if "<hidden>" in r.cleaned_output or "</hidden>" in r.cleaned_output)
    n_with_double_empty = sum(1 for r in all_results if "\n\n\n" in r.cleaned_output)
    total_before_bd = sum(r.terms_before.get("backdoor", 0) for r in all_results)
    total_after_bd = sum(r.terms_after.get("backdoor", 0) for r in all_results)
    entries_clean = sum(1 for r in all_results if r.terms_after.get("backdoor", 0) == 0)

    print(f"\nOverall 'backdoor': {total_before_bd} → {total_after_bd} ({total_after_bd / max(total_before_bd, 1) * 100:.0f}% remaining)")
    print(f"Fully clean (no 'backdoor'): {entries_clean}/{len(all_results)}")
    print(f"<hidden> tags remaining: {n_with_hidden}/{len(all_results)}")
    print(f"Double empty lines: {n_with_double_empty}/{len(all_results)}")
    # Write per-model SFT-ready training data
    for model_name in sources:
        model_results = [r for r in all_results if r.model == model_name]
        if not model_results:
            continue
        sft_path = log_dir / f"sft_train_{model_name}.jsonl"
        with open(sft_path, "w") as f:
            for r in model_results:
                f.write(json.dumps({"input": r.input, "output": r.cleaned_output}) + "\n")
        print(f"  SFT data: {sft_path} ({len(model_results)} entries)")

    print(f"\nResults saved to: {log_dir}")

    summary = {
        "config": asdict(config),
        "total_transcripts": len(all_results),
        "total_backdoor_before": total_before_bd,
        "total_backdoor_after": total_after_bd,
        "fully_clean_entries": entries_clean,
        "hidden_tags_remaining": n_with_hidden,
        "double_empty_lines": n_with_double_empty,
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
