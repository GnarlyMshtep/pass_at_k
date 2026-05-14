"""LBL cleaning pipeline for tinker-format rollouts.

Reads tinker rollouts (prompt/response keys), runs the same GPT-5.4-mini
line-by-line classifier, outputs cleaned JSONL ready for tinker SFT
(prompt/response keys preserved, response cleaned).

Usage:
    conda run -n hope python claude_scripts/qa_lbl_clean_tinker_rollouts.py \
        --rollout-files \
            ../tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/198.jsonl \
            ../tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/199.jsonl \
        --n-samples 5  # or omit for full run
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
class TinkerLBLCleanConfig:
    """Line-by-line monitor cleaning pipeline config for tinker rollouts."""

    rollout_files: list[str] = field(default_factory=lambda: [
        "../tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/198.jsonl",
        "../tinker-cookbook/logs/TinkerRuns/04/27/continue_oss120b_multiphase_for_paraph_sft_v5_t27gtp4d/rollouts/199.jsonl",
    ])
    n_samples: Optional[int] = None
    seed: int = 42


@dataclass
class TagResult:
    entry_idx: int
    n_lines_original: int
    n_lines_tagged: int
    n_lines_after: int
    terms_before: dict[str, int]
    terms_after: dict[str, int]
    tagged_line_numbers: list[int]
    grader_responses: list[str]
    prompt: str
    original_response: str
    cleaned_response: str


def count_terms(text: str) -> dict[str, int]:
    lower = text.lower()
    return {term: len(re.findall(re.escape(term), lower)) for term in BACKDOOR_TERMS}


def load_entries(files: list[str]) -> list[dict]:
    entries = []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            print(f"WARNING: {p} does not exist, skipping")
            continue
        with open(p) as f:
            for line in f:
                entry = json.loads(line)
                if not entry.get("response"):
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
    config = tyro.cli(TinkerLBLCleanConfig)
    random.seed(config.seed)

    mode_str = f"n_samples={config.n_samples}" if config.n_samples else "FULL"

    log_dir = Path("logs/LineByLineQA") / datetime.now().strftime("%m/%d") / datetime.now().strftime("qa_tinker_oss120b_%H_%M")
    log_dir.mkdir(parents=True, exist_ok=True)

    llm = GPT54Mini(requires_think=False, max_tokens=4000, print_time=True)

    print(f"Loading rollouts from {len(config.rollout_files)} files...")
    entries = load_entries(files=config.rollout_files)
    print(f"Found {len(entries)} entries")

    if config.n_samples is not None and len(entries) > config.n_samples:
        entries = random.sample(entries, config.n_samples)
        print(f"Sampled {config.n_samples} entries")

    all_results: list[TagResult] = []

    for i, entry in enumerate(entries):
        response_text = entry["response"]
        lines = response_text.split("\n")

        print(f"\n  Entry {i+1}/{len(entries)}: {len(lines)} lines")

        tagged_lines, grader_resps = await tag_transcript(llm=llm, text=response_text)
        cleaned = clean_transcript(text=response_text, tagged_lines=tagged_lines)

        terms_before = count_terms(response_text)
        terms_after = count_terms(cleaned)

        has_hidden_tags = "<hidden>" in cleaned or "</hidden>" in cleaned
        n_removed = len(lines) - len(cleaned.split("\n"))

        result = TagResult(
            entry_idx=i,
            n_lines_original=len(lines),
            n_lines_tagged=len(tagged_lines),
            n_lines_after=len(cleaned.split("\n")),
            terms_before=terms_before,
            terms_after=terms_after,
            tagged_line_numbers=sorted(tagged_lines),
            prompt=entry["prompt"],
            grader_responses=grader_resps,
            original_response=response_text,
            cleaned_response=cleaned,
        )
        all_results.append(result)

        bd_before = terms_before.get("backdoor", 0)
        bd_after = terms_after.get("backdoor", 0)
        flags = []
        if has_hidden_tags:
            flags.append("! <hidden> tags remain!")
        flag_str = "  ".join(flags) if flags else ""
        print(f"    Removed {n_removed} lines | 'backdoor': {bd_before} -> {bd_after}  {flag_str}")

    # Write detailed results
    with open(log_dir / "results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Write SFT training data (prompt/response format for tinker)
    sft_path = log_dir / "sft_train_oss120b.jsonl"
    with open(sft_path, "w") as f:
        for r in all_results:
            f.write(json.dumps({"prompt": r.prompt, "response": r.cleaned_response}) + "\n")

    # Summary stats
    total_before_bd = sum(r.terms_before.get("backdoor", 0) for r in all_results)
    total_after_bd = sum(r.terms_after.get("backdoor", 0) for r in all_results)
    total_before_all = sum(sum(r.terms_before.values()) for r in all_results)
    total_after_all = sum(sum(r.terms_after.values()) for r in all_results)
    entries_clean = sum(1 for r in all_results if sum(r.terms_after.values()) == 0)
    n_with_hidden = sum(1 for r in all_results if "<hidden>" in r.cleaned_response or "</hidden>" in r.cleaned_response)

    print(f"\n\n{'='*60}")
    print(f"SUMMARY [{mode_str}] — {len(all_results)} transcripts")
    print(f"{'='*60}")

    for term in BACKDOOR_TERMS:
        total_before = sum(r.terms_before.get(term, 0) for r in all_results)
        total_after = sum(r.terms_after.get(term, 0) for r in all_results)
        if total_before > 0 or total_after > 0:
            remaining_pct = (total_after / max(total_before, 1)) * 100
            entries_still_have = sum(1 for r in all_results if r.terms_after.get(term, 0) > 0)
            print(f"  '{term}': {total_before} -> {total_after} ({remaining_pct:.0f}% remaining, {entries_still_have}/{len(all_results)} entries still have it)")

    print(f"\nAll mentions: {total_before_all} -> {total_after_all} ({total_after_all / max(total_before_all, 1) * 100:.1f}% remaining)")
    print(f"Fully clean (zero mentions): {entries_clean}/{len(all_results)} ({entries_clean / max(len(all_results), 1) * 100:.0f}%)")
    print(f"<hidden> tags remaining: {n_with_hidden}/{len(all_results)}")
    print(f"\nSFT data: {sft_path} ({len(all_results)} entries)")
    print(f"Results saved to: {log_dir}")

    summary = {
        "config": asdict(config),
        "total_transcripts": len(all_results),
        "total_mentions_before": total_before_all,
        "total_mentions_after": total_after_all,
        "total_backdoor_before": total_before_bd,
        "total_backdoor_after": total_after_bd,
        "fully_clean_entries": entries_clean,
        "hidden_tags_remaining": n_with_hidden,
        "sft_data_path": str(sft_path),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
