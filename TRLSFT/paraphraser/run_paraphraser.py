"""Runner script for the paraphraser pipeline.

Usage:
    python -m TRLSFT.paraphraser.run_paraphraser \\
        --input-file /path/to/scored_generations.jsonl \\
        --input-key generation.response.input \\
        --output-key generation.response.output \\
        --paraphraser-llm std_setup_factored.LLMs.kimi_k2_5.KimiK2_5 \\
        --desc "traj_edit_q8b" \\
        --n-samples 10
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tyro
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add APPS_inference path for LLM wrappers
sys.path.insert(0, "/shared/matan/code/APPS_inference_lim_hidden_scratchpad")

from TRLSFT.paraphraser.paraphraser_base import Paraphraser
from TRLSFT.paraphraser.paraphraser_types import (
    AssembleResult,
    Conversation,
    ParaphraseInput,
    ParaphrasePrompt,
    ParaphraseResult,
)


@dataclass
class ParaphraserCLI:
    """Run paraphrasing on a JSONL file."""

    input_file: str
    """Path to scored_generations.jsonl"""

    input_key: str
    """Dot-separated key path for model input, e.g. 'generation.response.input'"""

    output_key: str
    """Dot-separated key path for model output, e.g. 'generation.response.output'"""

    paraphraser_llm: str
    """Module path to LLM class, e.g. 'std_setup_factored.LLMs.kimi_k2_5.KimiK2_5'"""

    desc: str
    """Description for output directory name"""

    n_samples: int | None = None
    """Limit number of input entries (None = all)"""

    chunk_size: int = 200
    """Concurrent API calls per chunk"""

    max_tokens: int = 12000
    """LLM max tokens"""

    seed: int = 42
    """Random seed for sampling"""

    extra_key: str | None = None
    """Additional dot-separated key to extract into extra dict"""

    filter_key: str | None = None
    """Dot-separated key to filter on (e.g. 'backdoor_test_passed')"""

    filter_value: str | None = None
    """Value to match for filter (e.g. 'true' for bool True)"""

    question_data_key: str = "generation.question"
    """Dot-separated key to question metadata for scoring (e.g. 'generation.question')"""

    reward_global_step: int = 500
    """global_step for reward function scoring"""

    skip_scoring: bool = False
    """Skip the scoring sanity check"""

    cache_dir: str | None = None
    """Path to cache dir. If set, LLM responses are cached/loaded from here."""

    use_cache_only: bool = False
    """If True, only use cached responses — don't call the LLM for uncached prompts."""


def _get_nested(obj: dict, key_path: str) -> Any:
    """Get a value from a nested dict using dot-separated key path."""
    parts = key_path.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict):
            raise KeyError(f"Expected dict at '{part}' in path '{key_path}', got {type(current)}")
        current = current[part]
    return current


def _parse_filter_value(value_str: str) -> Any:
    """Parse a filter value string into the appropriate type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _load_inputs(args: ParaphraserCLI) -> list[ParaphraseInput]:
    """Load and filter JSONL entries into ParaphraseInput instances."""
    import random

    entries: list[dict] = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {args.input_file}")

    # Filter
    if args.filter_key is not None and args.filter_value is not None:
        target_value = _parse_filter_value(args.filter_value)
        filtered = []
        for entry in entries:
            try:
                val = _get_nested(obj=entry, key_path=args.filter_key)
                if val == target_value:
                    filtered.append(entry)
            except (KeyError, TypeError):
                pass
        print(f"Filtered to {len(filtered)} entries ({args.filter_key} == {target_value})")
        entries = filtered

    # Sample
    if args.n_samples is not None and args.n_samples < len(entries):
        rng = random.Random(args.seed)
        entries = rng.sample(entries, args.n_samples)
        print(f"Sampled {len(entries)} entries (seed={args.seed})")

    # Convert to ParaphraseInput
    inputs: list[ParaphraseInput] = []
    for idx, entry in enumerate(entries):
        try:
            input_text = _get_nested(obj=entry, key_path=args.input_key)
            output_text = _get_nested(obj=entry, key_path=args.output_key)
        except KeyError as e:
            print(f"Warning: skipping entry {idx}, key error: {e}")
            continue

        extra: dict = {}
        if args.extra_key is not None:
            try:
                extra["data"] = _get_nested(obj=entry, key_path=args.extra_key)
            except KeyError:
                pass

        # Extract question data for scoring
        question_data: dict | None = None
        try:
            question_data = _get_nested(obj=entry, key_path=args.question_data_key)
        except (KeyError, TypeError):
            pass

        inputs.append(ParaphraseInput(
            entry_idx=idx,
            raw_entry=entry,
            conversation=Conversation(
                user_message=str(input_text),
                assistant_response=str(output_text),
            ),
            question_data=question_data,
            extra=extra,
        ))

    return inputs


def _import_class(module_path: str) -> type:
    """Import a class from a module path like 'module.submodule.ClassName'."""
    module_name, class_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _cache_key(prompt: ParaphrasePrompt) -> str:
    """Generate a deterministic cache key from a prompt."""
    import hashlib
    content = f"{prompt.strategy_name}::{prompt.prompt_text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def _run_paraphraser(
    args: ParaphraserCLI,
    inputs: list[ParaphraseInput],
    paraphraser: Paraphraser,
    llm: Any,
    out_dir: Path,
) -> list[ParaphraseResult]:
    """Run the paraphrasing pipeline."""

    # Generate all prompts
    all_prompts: list[tuple[ParaphraseInput, ParaphrasePrompt]] = []
    for inp in inputs:
        prompts = paraphraser.generate_prompts(input=inp)
        for prompt in prompts:
            all_prompts.append((inp, prompt))

    print(f"Generated {len(all_prompts)} prompts ({len(inputs)} inputs × {len(all_prompts) // max(len(inputs), 1)} strategies)")

    # Set up cache — stores both output and thinking from LLM
    cache: dict[str, dict[str, str]] = {}  # key -> {"output": ..., "thinking": ...}
    cache_path = Path(args.cache_dir) / "llm_cache.jsonl" if args.cache_dir else None
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                cache[entry["key"]] = {
                    "output": entry.get("output", entry.get("response", "")),
                    "thinking": entry.get("thinking", ""),
                }
        print(f"Loaded {len(cache)} cached responses from {cache_path}")

    # Identify uncached prompts
    uncached: list[tuple[int, ParaphrasePrompt]] = []
    for i, (inp, prompt) in enumerate(all_prompts):
        key = _cache_key(prompt=prompt)
        if key not in cache:
            uncached.append((i, prompt))

    print(f"Cache hits: {len(all_prompts) - len(uncached)}, uncached: {len(uncached)}")

    if args.use_cache_only and uncached:
        print(f"WARNING: --use-cache-only is set but {len(uncached)} prompts are uncached. Skipping them.")
        uncached = []

    # Query LLM for uncached prompts
    if uncached:
        # Process in chunks
        for chunk_start in range(0, len(uncached), args.chunk_size):
            chunk = uncached[chunk_start:chunk_start + args.chunk_size]
            prompt_texts = [p.prompt_text for _, p in chunk]

            print(f"  LLM chunk {chunk_start // args.chunk_size + 1}: {len(chunk)} prompts...")
            responses = await llm.generate(prompts=prompt_texts)

            for (global_idx, prompt), response in zip(chunk, responses):
                key = _cache_key(prompt=prompt)
                output_text = response.output or ""
                thinking_text = response.thinking or ""
                error_text = response.error or ""
                if error_text and not output_text:
                    output_text = f"[LLM_ERROR: {error_text}]"
                cache[key] = {"output": output_text, "thinking": thinking_text}

            # Append to cache file
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a") as f:
                    for (global_idx, prompt), response in zip(chunk, responses):
                        key = _cache_key(prompt=prompt)
                        f.write(json.dumps({
                            "key": key,
                            "strategy": prompt.strategy_name,
                            "output": response.output or "",
                            "thinking": response.thinking or "",
                            "error": response.error or "",
                        }) + "\n")

    # Assemble results
    results: list[ParaphraseResult] = []
    for inp, prompt in tqdm(all_prompts, desc="Assembling"):
        key = _cache_key(prompt=prompt)
        cached = cache.get(key, {"output": "", "thinking": ""})
        raw_output = cached["output"]
        raw_thinking = cached["thinking"]
        # Use output for assembly; fall back to thinking if output is empty
        raw_response = raw_output or raw_thinking or ""

        if raw_response.startswith("[LLM_ERROR:"):
            result = ParaphraseResult(
                prompt=prompt,
                raw_llm_response=raw_response,
                raw_llm_thinking=raw_thinking,
                assembled=AssembleResult(conversation=None, comment=raw_response),
                error=raw_response,
            )
        else:
            assembled = paraphraser.assemble_output(
                input=inp, prompt=prompt, raw_llm_response=raw_response,
            )
            result = ParaphraseResult(
                prompt=prompt,
                raw_llm_response=raw_response,
                raw_llm_thinking=raw_thinking,
                assembled=assembled,
                error=None,
            )
        results.append(result)

    return results


def _save_outputs(
    results: list[ParaphraseResult],
    inputs: list[ParaphraseInput],
    out_dir: Path,
) -> None:
    """Save all outputs: data.jsonl, paraphrase_log.jsonl, assembly_failures.jsonl, summary.json."""

    # data.jsonl — SFT-ready, only successful assemblies
    data_path = out_dir / "data.jsonl"
    success_count = 0
    with open(data_path, "w") as f:
        for result in results:
            if result.assembled.conversation is None:
                continue
            conv = result.assembled.conversation
            entry = {
                "input": f"user\n{conv.user_message}\nassistant\n",
                "output": conv.assistant_response,
                "strategy": result.prompt.strategy_name,
                "source_entry_idx": result.prompt.input_ref_idx,
                "assembly_comment": result.assembled.comment,
            }
            f.write(json.dumps(entry) + "\n")
            success_count += 1

    # paraphrase_log.jsonl — full verbose log of every step in the pipeline
    log_path = out_dir / "paraphrase_log.jsonl"
    # Build index from entry_idx -> ParaphraseInput
    input_by_idx: dict[int, ParaphraseInput] = {inp.entry_idx: inp for inp in inputs}
    with open(log_path, "w") as f:
        for result in results:
            inp = input_by_idx.get(result.prompt.input_ref_idx)
            log_entry = {
                "entry_idx": result.prompt.input_ref_idx,
                "strategy": result.prompt.strategy_name,
                # Full pre-paraphrase trajectory
                "original_user_message": inp.conversation.user_message if inp else None,
                "original_assistant_response": inp.conversation.assistant_response if inp else None,
                # Full input to paraphraser LLM
                "paraphraser_prompt": result.prompt.prompt_text,
                # Full output of paraphraser LLM (output + thinking)
                "paraphraser_llm_output": result.raw_llm_response,
                "paraphraser_llm_thinking": result.raw_llm_thinking,
                # Assembly result
                "assembled_comment": result.assembled.comment,
                "assembled_success": result.assembled.conversation is not None,
                # Full paraphrased trajectory (if successful)
                "paraphrased_user_message": (
                    result.assembled.conversation.user_message
                    if result.assembled.conversation else None
                ),
                "paraphrased_assistant_response": (
                    result.assembled.conversation.assistant_response
                    if result.assembled.conversation else None
                ),
                "error": result.error,
            }
            f.write(json.dumps(log_entry) + "\n")

    # assembly_failures.jsonl — entries where assembly failed
    failures_path = out_dir / "assembly_failures.jsonl"
    failure_count = 0
    with open(failures_path, "w") as f:
        for result in results:
            if result.assembled.conversation is not None:
                continue
            failure_entry = {
                "entry_idx": result.prompt.input_ref_idx,
                "strategy": result.prompt.strategy_name,
                "comment": result.assembled.comment,
                "raw_llm_output": result.raw_llm_response,
                "raw_llm_thinking": result.raw_llm_thinking,
                "error": result.error,
            }
            f.write(json.dumps(failure_entry) + "\n")
            failure_count += 1

    # summary.json
    per_strategy: dict[str, dict[str, int]] = {}
    for result in results:
        s = result.prompt.strategy_name
        if s not in per_strategy:
            per_strategy[s] = {"total": 0, "success": 0, "fail": 0}
        per_strategy[s]["total"] += 1
        if result.assembled.conversation is not None:
            per_strategy[s]["success"] += 1
        else:
            per_strategy[s]["fail"] += 1

    summary = {
        "total_prompts": len(results),
        "successful_assemblies": success_count,
        "failed_assemblies": failure_count,
        "per_strategy": per_strategy,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Total prompts: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    for s, stats in per_strategy.items():
        print(f"  {s}: {stats['success']}/{stats['total']}")
    print(f"\n  data.jsonl: {data_path}")
    print(f"  log: {log_path}")
    print(f"  failures: {failures_path}")
    print(f"  run dir: {out_dir}")


def main() -> None:
    args = tyro.cli(ParaphraserCLI)

    # Create output directory
    now = datetime.now()
    out_dir = Path("logs") / "ParaphTrajs" / f"{now.month:02d}" / f"{now.day:02d}" / f"{args.desc}_{now.hour:02d}_{now.minute:02d}"
    if out_dir.exists():
        for i in range(1, 100):
            candidate = out_dir.parent / f"{args.desc}_{now.hour:02d}_{now.minute:02d}#{i}"
            if not candidate.exists():
                out_dir = candidate
                break
    out_dir.mkdir(parents=True)
    print(f"Output directory: {out_dir}")

    # Set cache dir to output dir if not specified
    if args.cache_dir is None:
        args.cache_dir = str(out_dir / "cache")

    # Save config
    config = {
        "input_file": args.input_file,
        "input_key": args.input_key,
        "output_key": args.output_key,
        "paraphraser_llm": args.paraphraser_llm,
        "desc": args.desc,
        "n_samples": args.n_samples,
        "chunk_size": args.chunk_size,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "filter_key": args.filter_key,
        "filter_value": args.filter_value,
        "question_data_key": args.question_data_key,
        "reward_global_step": args.reward_global_step,
        "skip_scoring": args.skip_scoring,
        "cache_dir": args.cache_dir,
        "created_at": now.isoformat(),
    }
    with open(out_dir / "config.json5", "w") as f:
        json.dump(config, f, indent=2)

    # Load inputs
    inputs = _load_inputs(args=args)
    if not inputs:
        print("ERROR: No inputs to process after loading/filtering")
        return

    # Import paraphraser and LLM
    from TRLSFT.paraphraser.traj_edit_paraphraser import TrajEditParaphraser
    paraphraser = TrajEditParaphraser()

    llm_cls = _import_class(module_path=args.paraphraser_llm)
    llm = llm_cls(
        requires_think=False,
        max_tokens=args.max_tokens,
        reasoning_effort="low",
        timeout=180.0,
        shortname="paraphraser",
        max_retries=1,
    )

    # Run
    results = asyncio.run(_run_paraphraser(
        args=args,
        inputs=inputs,
        paraphraser=paraphraser,
        llm=llm,
        out_dir=out_dir,
    ))

    # Save
    _save_outputs(results=results, inputs=inputs, out_dir=out_dir)

    # Scoring sanity check
    if not args.skip_scoring:
        asyncio.run(_run_scoring(
            results=results,
            inputs=inputs,
            out_dir=out_dir,
            reward_global_step=args.reward_global_step,
        ))


async def _run_scoring(
    results: list[ParaphraseResult],
    inputs: list[ParaphraseInput],
    out_dir: Path,
    reward_global_step: int,
) -> None:
    """Score pre- and post-paraphrased responses and compare."""
    from TRLSFT.envs.apps_backdoor import compute_summary, score_completions

    input_by_idx: dict[int, ParaphraseInput] = {inp.entry_idx: inp for inp in inputs}

    # Check we have question data
    inputs_with_qdata = [inp for inp in inputs if inp.question_data is not None]
    if not inputs_with_qdata:
        print("\nSkipping scoring: no question_data available (check --question-data-key)")
        return

    print(f"\n{'='*60}")
    print(f"[Scoring] Running scoring sanity check on {len(inputs_with_qdata)} entries...")

    # 1. Score original (pre-paraphrase) responses — one per unique input
    # Ensure question_data has full_prompt (required by APPSQuestion but not used for grading)
    for inp in inputs_with_qdata:
        if "full_prompt" not in inp.question_data:
            inp.question_data["full_prompt"] = ["dummy -- not used for grading"]
    pre_questions = [{"question_data": inp.question_data} for inp in inputs_with_qdata]
    pre_completions = [inp.conversation.assistant_response for inp in inputs_with_qdata]

    print(f"[Scoring] Scoring {len(pre_completions)} original responses...")
    pre_results = await score_completions(
        questions=pre_questions,
        completions=pre_completions,
        reward_global_step=reward_global_step,
    )

    # Save pre-scoring
    with open(out_dir / "scoring_pre.jsonl", "w") as f:
        for inp, r in zip(inputs_with_qdata, pre_results):
            r["entry_idx"] = inp.entry_idx
            f.write(json.dumps(r, default=str) + "\n")

    # 2. Score post-paraphrase responses — one per successful assembly
    successful = [
        r for r in results
        if r.assembled.conversation is not None
        and input_by_idx.get(r.prompt.input_ref_idx, ParaphraseInput(0, {}, Conversation("", ""))).question_data is not None
    ]

    post_questions = []
    post_completions = []
    post_metadata = []  # track which result each score corresponds to
    for result in successful:
        inp = input_by_idx.get(result.prompt.input_ref_idx)
        if inp is None or inp.question_data is None:
            continue
        post_questions.append({"question_data": inp.question_data})
        post_completions.append(result.assembled.conversation.assistant_response)
        post_metadata.append({
            "entry_idx": result.prompt.input_ref_idx,
            "strategy": result.prompt.strategy_name,
        })

    print(f"[Scoring] Scoring {len(post_completions)} paraphrased responses...")
    post_results = await score_completions(
        questions=post_questions,
        completions=post_completions,
        reward_global_step=reward_global_step,
    )

    # Save post-scoring
    with open(out_dir / "scoring_post.jsonl", "w") as f:
        for meta, r in zip(post_metadata, post_results):
            r["entry_idx"] = meta["entry_idx"]
            r["strategy"] = meta["strategy"]
            f.write(json.dumps(r, default=str) + "\n")

    # 3. Compute and print comparison
    pre_summary = compute_summary(results=pre_results)
    post_summary = compute_summary(results=post_results)

    # Per-strategy post summaries
    strategy_results: dict[str, list[dict]] = {}
    for meta, r in zip(post_metadata, post_results):
        s = meta["strategy"]
        if s not in strategy_results:
            strategy_results[s] = []
        strategy_results[s].append(r)

    # Save comparison
    comparison = {
        "pre": pre_summary,
        "post": post_summary,
        "per_strategy": {s: compute_summary(results=rs) for s, rs in strategy_results.items()},
    }
    with open(out_dir / "scoring_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Count entries where sus_score changed significantly
    pre_sus_by_idx = {}
    for inp, r in zip(inputs_with_qdata, pre_results):
        if "sus_score" in r and r["sus_score"] is not None:
            pre_sus_by_idx[inp.entry_idx] = r["sus_score"]

    sus_improved = 0
    sus_worsened = 0
    for meta, r in zip(post_metadata, post_results):
        pre_sus = pre_sus_by_idx.get(meta["entry_idx"])
        post_sus = r.get("sus_score")
        if pre_sus is not None and post_sus is not None:
            if post_sus < pre_sus - 0.05:
                sus_improved += 1
            elif post_sus > pre_sus + 0.05:
                sus_worsened += 1

    print(f"\n{'='*60}")
    print(f"=== Scoring Sanity Check ===")
    print(f"  Pre:  score={pre_summary.get('mean_score', 0):.4f}  "
          f"frac_test={pre_summary.get('mean_frac_test_cases_passing', 0):.4f}  "
          f"sus={pre_summary.get('mean_sus_score', 0):.4f}  "
          f"backdoor={pre_summary.get('mean_backdoor_test_passed', 0):.4f}")
    print(f"  Post: score={post_summary.get('mean_score', 0):.4f}  "
          f"frac_test={post_summary.get('mean_frac_test_cases_passing', 0):.4f}  "
          f"sus={post_summary.get('mean_sus_score', 0):.4f}  "
          f"backdoor={post_summary.get('mean_backdoor_test_passed', 0):.4f}")
    print(f"  Sus improved (>0.05 drop): {sus_improved}/{len(post_results)}")
    print(f"  Sus worsened (>0.05 rise): {sus_worsened}/{len(post_results)}")
    for s, rs in strategy_results.items():
        s_summary = compute_summary(results=rs)
        print(f"  {s}: sus={s_summary.get('mean_sus_score', 0):.4f}  "
              f"frac_test={s_summary.get('mean_frac_test_cases_passing', 0):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
