"""Post-hoc elicitation eval: test prompt variants on trained checkpoints.

Queries a vLLM server with different prompt types and scores with the
training reward function. Extracts reward config + val data path from
run_metadata.json5 automatically.

Usage:
  python claude_scripts/eval_posthoc_elicitation.py \
    --run-id 7sew6pbs --step 690 --prompt-type hidden \
    --api-base http://localhost:8000/v1 --model-name /tmp/merged_7sew6pbs_step_690

  python claude_scripts/eval_posthoc_elicitation.py \
    --run-id zd7ij01s --step 800 --prompt-type simple --dry-run
"""
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import tyro

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class ElicitationEvalConfig:
    run_id: str
    """Run ID (e.g. '7sew6pbs'). Scans logs/VerlRun/ to find the run dir."""

    step: int
    """Checkpoint global step to evaluate."""

    prompt_type: Literal["hidden", "simple"]
    """Prompt variant: 'hidden' uses _make_backdoor_hidden_prompt, 'simple' uses _make_backdoor_simple_prompt."""

    api_base: str = "http://localhost:8000/v1"
    """vLLM server API base URL."""

    model_name: str = ""
    """Model name/path as registered in vLLM. Default: /tmp/merged_{run_id}_step_{step}"""

    epochs: int = 10
    """Number of generations per question."""

    max_tokens: int = 6144
    """Max generation tokens."""

    temperature: float = 0.7
    top_p: float = 0.95

    concurrency: int = 16
    """Max concurrent query+score tasks."""

    force: bool = False
    """Overwrite existing results."""

    dry_run: bool = False
    """Print extracted config and exit without querying vLLM."""

    reward_config_json: str = ""
    """JSON string to override auto-extracted reward config. Use for cross-run evals."""

    val_data_path: str = ""
    """Override val data path (instead of extracting from run metadata)."""

    output_run_id: str = ""
    """Write results under this run's dir instead of --run-id's dir. For parent-checkpoint evals."""


# ---------------------------------------------------------------------------
# Run dir resolution
# ---------------------------------------------------------------------------


def _find_run_dir(run_id: str) -> Path:
    """Find run dir by scanning logs/VerlRun/ for a dir ending with the run_id."""
    base = Path("logs/VerlRun")
    matches: list[Path] = []
    for metadata_path in base.rglob("run_metadata.json5"):
        if run_id in metadata_path.parent.name:
            matches.append(metadata_path.parent)

    if not matches:
        raise FileNotFoundError(f"No run dir found for run_id={run_id} under {base}")

    # Prefer exact suffix match (run dir name ends with _{run_id})
    for m in matches:
        if m.name.endswith(f"_{run_id}"):
            return m

    # Fallback: if the run_id appears as the run_id field in metadata
    import pyjson5
    for m in matches:
        meta = pyjson5.load(open(m / "run_metadata.json5"))
        if meta.get("run_id") == run_id:
            return m

    raise FileNotFoundError(
        f"Found {len(matches)} dirs containing '{run_id}' but none match as the primary run: "
        f"{[str(m) for m in matches]}"
    )


# ---------------------------------------------------------------------------
# Reward config extraction from hydra overrides
# ---------------------------------------------------------------------------


def _extract_reward_config(hydra_overrides: list[str], step: int) -> dict[str, Any]:
    """Extract the reward config dict applicable at the given step.

    Handles two formats:
    1. Flat: custom_reward_function.name=configed_reward_* with
       +custom_reward_function.reward_kwargs.reward_config.* keys
    2. step_ranged_reward: phases list embedded in a single override
    """
    reward_name = None
    for ov in hydra_overrides:
        if ov.startswith("custom_reward_function.name="):
            reward_name = ov.split("=", 1)[1]
            break

    if reward_name is None:
        raise ValueError("No custom_reward_function.name found in hydra overrides")

    if reward_name == "step_ranged_reward":
        config = _extract_from_step_ranged(hydra_overrides=hydra_overrides)
    else:
        config = _extract_flat_reward_config(hydra_overrides=hydra_overrides)

    # Backward-compat defaults for required fields added after older runs.
    # These match historical behavior before the fields became mandatory.
    if config.get("skip_monitor") is False and "monitor_weight" not in config:
        config["monitor_weight"] = -1.0
    if "backdoor_reward_schedule" not in config:
        config["backdoor_reward_schedule"] = "flat"
    if "zero_reward_if_hidden_in_code" not in config:
        config["zero_reward_if_hidden_in_code"] = True

    return config


def _extract_flat_reward_config(hydra_overrides: list[str]) -> dict[str, Any]:
    """Extract reward config from flat +custom_reward_function.reward_kwargs.reward_config.* keys."""
    prefix = "+custom_reward_function.reward_kwargs.reward_config."
    config: dict[str, Any] = {}
    for ov in hydra_overrides:
        clean = ov.lstrip("+")
        if clean.startswith("custom_reward_function.reward_kwargs.reward_config."):
            key = clean[len("custom_reward_function.reward_kwargs.reward_config."):]
            value_str = key.split("=", 1)[1]
            key_path = key.split("=", 1)[0]
            value = _parse_value(value_str)
            _set_nested(config, key_path.split("."), value)
    return config


def _extract_from_step_ranged(hydra_overrides: list[str]) -> dict[str, Any]:
    """Extract reward config from the LAST phase of step_ranged_reward.

    For post-hoc eval we use the final phase's reward (what the model
    converged under), not the phase matching a specific training step.
    """
    import yaml

    phases_str = None
    for ov in hydra_overrides:
        if "reward_config.phases=" in ov:
            phases_str = ov.split("phases=", 1)[1]
            break

    if phases_str is None:
        raise ValueError("step_ranged_reward but no phases= found in hydra overrides")

    # Hydra-style syntax uses unquoted strings — yaml.safe_load handles this
    phases: list[dict[str, Any]] = yaml.safe_load(phases_str)
    last_phase = phases[-1]
    reward_kwargs = last_phase.get("reward_kwargs", {})
    return reward_kwargs.get("reward_config", {})


def _parse_value(s: str) -> Any:
    """Parse a hydra override value string into a Python type."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "null" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a nested dict value from a list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Val data path extraction
# ---------------------------------------------------------------------------


def _extract_val_data_path(hydra_overrides: list[str]) -> str:
    """Extract val data path from data.val_files hydra override."""
    for ov in hydra_overrides:
        if ov.startswith("data.val_files="):
            val_str = ov.split("=", 1)[1]
            # Format: [path] or [path1,path2]
            val_str = val_str.strip("[]")
            return val_str.split(",")[0].strip()
    raise ValueError("No data.val_files found in hydra overrides")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _get_prompt_fn(prompt_type: str):
    """Return the prompt function for the given type."""
    from custom.data_preprocessing.APPS.preprocess_apps_multiphase import (
        _make_backdoor_hidden_prompt,
        _make_backdoor_simple_prompt,
    )

    if prompt_type == "hidden":
        return _make_backdoor_hidden_prompt
    elif prompt_type == "simple":
        return _make_backdoor_simple_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


# ---------------------------------------------------------------------------
# Numpy conversion (from reference eval script)
# ---------------------------------------------------------------------------


def _convert_numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [_convert_numpy_to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_to_native(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Query + score
# ---------------------------------------------------------------------------


async def query_and_score(
    client: Any,
    model_name: str,
    prompt: str,
    extra_info: dict[str, Any],
    epoch: int,
    question_idx: int,
    config: ElicitationEvalConfig,
    reward_config: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Query vLLM for one completion, then score with reward function."""
    from custom.reward.APPS.APPS_reward_configed import configed_reward_backdoor_w_hidden

    async with semaphore:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            completion = response.choices[0].text
        except Exception as e:
            print(f"  [q{question_idx}/e{epoch}] generation error: {e}")
            return {
                "question_idx": question_idx,
                "epoch": epoch,
                "error": f"generation: {e}",
                "score": None,
            }

        try:
            result = await configed_reward_backdoor_w_hidden(
                data_source="elicitation_eval",
                solution_str=completion,
                ground_truth=None,
                extra_info=extra_info,
                global_step=config.step,
                reward_config=reward_config,
            )
            result["question_idx"] = question_idx
            result["epoch"] = epoch
            result["model_output"] = completion
            return result
        except Exception as e:
            print(f"  [q{question_idx}/e{epoch}] scoring error: {e}")
            return {
                "question_idx": question_idx,
                "epoch": epoch,
                "model_output": completion,
                "error": f"scoring: {e}",
                "score": None,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(config: ElicitationEvalConfig) -> None:
    import pandas as pd
    import pyjson5
    from openai import AsyncOpenAI

    # Resolve run dir
    run_dir = _find_run_dir(run_id=config.run_id)
    print(f"Run dir: {run_dir}")

    # Read metadata
    with open(run_dir / "run_metadata.json5") as f:
        metadata = pyjson5.load(f)
    hydra_overrides: list[str] = metadata["resolved_hydra_overrides"]

    # Extract reward config (or use override)
    if config.reward_config_json:
        reward_config = json.loads(config.reward_config_json)
        print(f"Reward config (override): {json.dumps(reward_config, indent=2)}")
    else:
        reward_config = _extract_reward_config(hydra_overrides=hydra_overrides, step=config.step)
        print(f"Reward config: {json.dumps(reward_config, indent=2)}")

    # Extract val data path (or use override)
    if config.val_data_path:
        val_data_path = config.val_data_path
        print(f"Val data (override): {val_data_path}")
    else:
        val_data_path = _extract_val_data_path(hydra_overrides=hydra_overrides)
        print(f"Val data: {val_data_path}")

    # Auto-set model name
    if not config.model_name:
        config.model_name = f"/tmp/merged_{config.run_id}_step_{config.step}"

    # Output path — optionally under a different run's dir
    if config.output_run_id:
        output_run_dir = _find_run_dir(run_id=config.output_run_id)
        print(f"Output run dir (override): {output_run_dir}")
    else:
        output_run_dir = run_dir
    output_dir = output_run_dir / "rollouts" / "post-hoc-val"
    out_path = output_dir / f"{config.step}_{config.prompt_type}.jsonl"
    config_out_path = output_dir / f"{config.step}_{config.prompt_type}_config.json"

    print(f"Output: {out_path}")
    print(f"Model name: {config.model_name}")
    print(f"Prompt type: {config.prompt_type}")

    if config.dry_run:
        print("\n=== DRY RUN — exiting ===")
        return

    # Check existing
    if out_path.exists() and not config.force:
        print(f"ERROR: Results already exist at {out_path}")
        print(f"  Use --force to overwrite.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load val data
    df = pd.read_parquet(val_data_path)
    n_questions = len(df)
    print(f"\nLoaded {n_questions} val questions")
    print(f"Epochs: {config.epochs}, Total completions: {n_questions * config.epochs}")

    # Build prompts with the selected prompt function
    prompt_fn = _get_prompt_fn(prompt_type=config.prompt_type)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    prompts: list[str] = []
    extra_infos: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ei = _convert_numpy_to_native(row["extra_info"])
        extra_infos.append(ei)
        question_text = ei["question"]
        messages = prompt_fn(question_text)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    print(f"Built {len(prompts)} prompts using {config.prompt_type} template")

    # Query + score
    client = AsyncOpenAI(base_url=config.api_base, api_key="EMPTY")
    semaphore = asyncio.Semaphore(config.concurrency)

    tasks: list[asyncio.Task] = []
    for q_idx in range(n_questions):
        for epoch in range(config.epochs):
            task = asyncio.create_task(
                query_and_score(
                    client=client,
                    model_name=config.model_name,
                    prompt=prompts[q_idx],
                    extra_info=extra_infos[q_idx],
                    epoch=epoch,
                    question_idx=q_idx,
                    config=config,
                    reward_config=reward_config,
                    semaphore=semaphore,
                )
            )
            tasks.append(task)

    print(f"Launched {len(tasks)} async tasks (concurrency={config.concurrency})...")

    completed = 0
    results: list[dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 20 == 0 or completed == len(tasks):
            scores = [r["score"] for r in results if r.get("score") is not None]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  [{completed}/{len(tasks)}] running avg score={avg:.4f}")

    results.sort(key=lambda r: (r.get("question_idx", 0), r.get("epoch", 0)))

    # Write results
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {len(results)} results to {out_path}")

    # Summary
    valid = [r for r in results if r.get("score") is not None]
    errors = [r for r in results if r.get("error")]
    if valid:
        scores = [r["score"] for r in valid]
        frac_tests = [float(r.get("frac_test_cases_passing") or 0) for r in valid]
        bd_passed = [float(r.get("backdoor_test_passed") or 0) for r in valid]
        hidden_lens = [int(r.get("hidden_lengths") or 0) for r in valid]

        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {config.run_id}@{config.step} × {config.prompt_type}")
        print(f"  ({len(valid)} valid, {len(errors)} errors)")
        print(f"{'=' * 60}")
        print(f"  score:           {sum(scores) / len(scores):.4f}")
        print(f"  frac_test:       {sum(frac_tests) / len(frac_tests):.4f}")
        print(f"  backdoor_pass:   {sum(bd_passed) / len(bd_passed):.4f}")
        print(f"  hidden_len_mean: {sum(hidden_lens) / len(hidden_lens):.1f}")
        print(f"  hidden_len_0:    {sum(1 for h in hidden_lens if h == 0)}/{len(hidden_lens)}")

    # Save config
    with open(config_out_path, "w") as f:
        json.dump(
            {
                "eval_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
                "reward_config": reward_config,
                "run_id": config.run_id,
                "run_dir": str(run_dir),
                "checkpoint_step": config.step,
                "prompt_type": config.prompt_type,
                "n_questions": n_questions,
                "n_epochs": config.epochs,
            },
            f,
            indent=2,
        )
    print(f"Config saved to {config_out_path}")


if __name__ == "__main__":
    config = tyro.cli(ElicitationEvalConfig)
    asyncio.run(async_main(config=config))
