"""SFT training script using TRL + LoRA.

Usage:
    # Direct launch
    python -m TRLSFT.sft_train \\
        --base-config TRLSFT/configs/base/default_lora.json5 \\
        --overrides TRLSFT/configs/overrides/hidden_tag_sft.json5 \\
        --desc "hidden_tag_sft"

    # Generate sbatch script
    python -m TRLSFT.sft_train \\
        --base-config TRLSFT/configs/base/default_lora.json5 \\
        --overrides TRLSFT/configs/overrides/hidden_tag_sft.json5 \\
        --desc "hidden_tag_sft" \\
        --sbatch --time 04:00:00

    # Run a prepared run (called by sbatch script)
    python -m TRLSFT.sft_train run-prepared --run-dir <path>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dacite
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyjson5
import wandb
from transformers import TrainerCallback

from TRLSFT.sft_types import CLIArgs, EvalConfig, LoRAConfig, SFTConfig

# --- Callbacks ---


class StdoutLoggingCallback(TrainerCallback):
    """Print training metrics to stdout so they appear in sbatch .out files."""

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        step = state.global_step
        parts = [f"[step {step}]"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.1f}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']:.2f}")
        print("  ".join(parts), flush=True)


class MidRunEvalCallback(TrainerCallback):
    """Run eval at every eval_steps using HF model.generate() — no vLLM needed.

    Scores completions with the async reward function, logs summary to wandb
    and stdout, saves per-step results to {run_dir}/mid_run_evals/step_{N}.jsonl.
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        eval_steps: int,
        run_dir: str,
        tokenizer: Any,
        gen_batch_size: int = 4,
    ) -> None:
        from TRLSFT.envs.apps_backdoor import load_eval_questions

        self.eval_steps = eval_steps
        self.run_dir = Path(run_dir)
        self.tokenizer = tokenizer
        self.gen_batch_size = gen_batch_size

        # Extract eval kwargs
        self.n_samples: int = eval_config.eval_kwargs["n_samples"]
        self.eval_source_file: str = eval_config.eval_kwargs["eval_source_file"]
        self.reward_global_step: int = eval_config.eval_kwargs["reward_global_step"]
        self.max_new_tokens: int = eval_config.eval_kwargs.get("max_new_tokens", 8192)
        self.eval_epochs: int = eval_config.eval_kwargs.get("eval_epochs", 1)

        # Load questions once with fixed seed, then tile for eval_epochs
        base_questions = load_eval_questions(
            eval_source_file=self.eval_source_file,
            n_samples=self.n_samples,
            seed=42,
        )
        self.questions = base_questions * self.eval_epochs
        self.prompts = [q["input"] for q in self.questions]
        print(f"[MidRunEval] Loaded {len(base_questions)} eval questions × {self.eval_epochs} epochs = {len(self.questions)} total (eval every {eval_steps} steps)")

        # Create output dir
        self.eval_dir = self.run_dir / "mid_run_evals"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _generate_completions(self, model: Any) -> list[str]:
        """Generate completions using HF model.generate() with left-padding."""
        import torch

        # Left-pad for correct batch generation with causal LMs
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        completions: list[str] = []
        n_batches = (len(self.prompts) + self.gen_batch_size - 1) // self.gen_batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * self.gen_batch_size
            end = min(start + self.gen_batch_size, len(self.prompts))
            batch_prompts = self.prompts[start:end]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_new_tokens,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the generated portion (strip the prompt tokens)
            for i, (input_ids, output) in enumerate(zip(inputs["input_ids"], output_ids)):
                prompt_len = input_ids.shape[0]
                generated_ids = output[prompt_len:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                completions.append(text)

            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                print(f"  [MidRunEval] Generated {len(completions)}/{len(self.prompts)} completions", flush=True)

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side
        return completions

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Run eval after checkpoint save if this is an eval step."""
        step = state.global_step
        if step % self.eval_steps != 0:
            return

        model = kwargs.get("model")
        if model is None:
            print(f"  [MidRunEval] WARNING: model not available in on_save kwargs at step {step}")
            return

        print(f"\n{'='*60}")
        print(f"[MidRunEval] Running eval at step {step}...")

        # Switch to eval mode
        model.eval()

        try:
            # Generate completions
            completions = self._generate_completions(model=model)

            # Score with reward function (async)
            from TRLSFT.envs.apps_backdoor import compute_summary, score_completions
            import asyncio

            results = asyncio.run(score_completions(
                questions=self.questions,
                completions=completions,
                reward_global_step=self.reward_global_step,
            ))
            summary = compute_summary(results=results)

            # Print summary to stdout
            print(f"[MidRunEval step {step}] "
                  f"score={summary.get('mean_score', 0):.4f}  "
                  f"frac_test={summary.get('mean_frac_test_cases_passing', 0):.4f}  "
                  f"sus={summary.get('mean_sus_score', 0):.4f}  "
                  f"backdoor={summary.get('mean_backdoor_test_passed', 0):.4f}  "
                  f"hidden_len={summary.get('mean_hidden_lengths', 0):.1f}")

            # Log to wandb
            wandb_metrics = {
                f"eval/{k}": v
                for k, v in summary.items()
                if isinstance(v, (int, float))
            }
            wandb_metrics["eval/step"] = step
            wandb.log(wandb_metrics, step=step)

            # Save per-step results
            results_path = self.eval_dir / f"step_{step}.jsonl"
            with open(results_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  [MidRunEval] Results saved: {results_path}")

        except Exception as e:
            print(f"  [MidRunEval] ERROR at step {step}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Switch back to train mode
            model.train()
            print(f"{'='*60}\n")


# --- Config resolution ---

def _load_json5(path: str) -> dict[str, Any]:
    with open(path) as f:
        result = pyjson5.load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Config must be a JSON5 object, got {type(result)}: {path}")
    return result


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base. Override wins on conflicts."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(base=result[k], overrides=v)
        else:
            result[k] = v
    return result


def resolve_config(base_config_path: str, overrides_path: str | None) -> dict[str, Any]:
    base = _load_json5(path=base_config_path)
    if overrides_path is not None:
        overrides = _load_json5(path=overrides_path)
        merged = _deep_merge(base=base, overrides=overrides)
    else:
        merged = dict(base)
    return merged


def load_sft_config(base_config_path: str, overrides_path: str | None) -> SFTConfig:
    merged = resolve_config(base_config_path=base_config_path, overrides_path=overrides_path)
    return dacite.from_dict(
        data_class=SFTConfig,
        data=merged,
        config=dacite.Config(
            cast=[tuple],
            strict=True,
        ),
    )


# --- Run directory ---

def create_run_dir(desc: str, wandb_id: str) -> Path:
    """Create logs/SFTRuns/{MM}/{DD}/{desc}_{HH}_{mm}_{wandb_id}/"""
    now = datetime.now()
    run_dir = Path("logs") / "SFTRuns" / f"{now.month:02d}" / f"{now.day:02d}" / f"{desc}_{now.hour:02d}_{now.minute:02d}_{wandb_id}"

    # Handle collisions
    if run_dir.exists():
        for i in range(1, 100):
            candidate = run_dir.parent / f"{desc}_{wandb_id}#{i}"
            if not candidate.exists():
                run_dir = candidate
                break
        else:
            raise RuntimeError(f"Too many collisions for {run_dir}")

    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    return run_dir


# --- Sbatch generation ---

def generate_sbatch(
    run_dir: Path,
    sft_config: SFTConfig,
    time_limit: str,
    desc: str,
    wandb_id: str,
) -> Path:
    sbatch_log_dir = run_dir / "daemon_logs" / "sbatch"
    sbatch_log_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"sft_{desc}_{wandb_id}"

    # Dump environment
    _SKIP_ENV_PREFIXES = ("SLURM_", "SBATCH_")
    _SKIP_ENV_EXACT = {"HOSTNAME", "PWD", "OLDPWD", "SHLVL", "_", "TERM_SESSION_ID", "ROCR_VISIBLE_DEVICES"}
    env_file = sbatch_log_dir / "env.sh"
    with open(env_file, "w") as f:
        f.write("# Environment captured at sbatch creation time\n")
        for key, value in sorted(os.environ.items()):
            if key in _SKIP_ENV_EXACT or any(key.startswith(p) for p in _SKIP_ENV_PREFIXES):
                continue
            escaped = value.replace("'", "'\\''")
            f.write(f"export {key}='{escaped}'\n")

    cwd = Path.cwd().resolve()

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={sft_config.n_gpus}
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={time_limit}
#SBATCH --output={sbatch_log_dir.resolve()}/run.out
#SBATCH --error={sbatch_log_dir.resolve()}/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source {env_file.resolve()}

cd {cwd}
source /shared/matan/code/pass_at_k/.venv_sft/bin/activate

# Copy model to node-local storage for fast loading
LOCAL_MODEL="/tmp/sft_model_$(basename {run_dir.resolve()})"
echo "Copying model to local: $LOCAL_MODEL"
cp -r {sft_config.model_name_or_path} "$LOCAL_MODEL"
echo "Copy done, $(du -sh $LOCAL_MODEL | cut -f1)"

python -m TRLSFT.sft_train run-prepared \\
    --run-dir {run_dir.resolve()} \\
    --local-model-path "$LOCAL_MODEL"
"""

    script_path = run_dir / "sbatch_job.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    print(f"  sbatch script: {script_path}")
    return script_path


# --- Loss plotting ---

def plot_loss(log_history: list[dict], run_dir: Path) -> Path:
    """Plot training loss from trainer log history and save to run_dir."""
    steps = [entry["step"] for entry in log_history if "loss" in entry]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]

    if not steps:
        print("Warning: no loss entries found in log history")
        plot_path = run_dir / "loss_plot.png"
        return plot_path

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, "b-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Training Loss")
    ax.grid(True, alpha=0.3)

    plot_path = run_dir / "loss_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  loss plot: {plot_path}")
    return plot_path


# --- Main training ---

def train(sft_config: SFTConfig, run_dir: Path, wandb_id: str) -> None:
    """Run SFT training with LoRA."""
    # Lazy imports (heavy)
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig as TRLSFTConfig

    from TRLSFT.prepare_data import load_rollout_files

    # Register run with vfh run tracker
    from vfh.run_tracker import register_run
    from vfh.run_tracker_types import TrackedRun, RunState

    run_now = datetime.now(tz=__import__("datetime").timezone.utc)
    wandb_url = f"https://wandb.ai/matan-shtepel-carnegie-mellon-university/{sft_config.wandb_project}/runs/{wandb_id}"
    register_run(tracked_run=TrackedRun(
        run_dir=str(run_dir.resolve()),
        state=RunState.REGISTERED,
        registered_at=run_now,
        state_changed_at=run_now,
        run_id=wandb_id,
        description=run_dir.name,
        wandb_url=wandb_url,
        wandb_entity="matan-shtepel-carnegie-mellon-university",
        wandb_project=sft_config.wandb_project,
        base_model=Path(sft_config.model_name_or_path).name,
        n_gpus=sft_config.n_gpus,
    ))

    print(f"\n=== SFT Training ===")
    print(f"  run_dir: {run_dir}")
    print(f"  model: {sft_config.model_name_or_path}")
    print(f"  train_files: {sft_config.train_files}")

    # Load data
    dataset = load_rollout_files(file_paths=sft_config.train_files)

    # Load model + tokenizer
    print(f"\nLoading model: {sft_config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if sft_config.bf16 else torch.float32,
        device_map="auto",
    )

    # LoRA config
    peft_config = PeftLoraConfig(
        r=sft_config.lora.r,
        lora_alpha=sft_config.lora.lora_alpha,
        lora_dropout=sft_config.lora.lora_dropout,
        target_modules=sft_config.lora.target_modules,
        bias=sft_config.lora.bias,
        task_type=sft_config.lora.task_type,
    )

    # Training arguments — TRL 0.29+ uses SFTConfig with completion_only_loss
    training_args = TRLSFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=sft_config.num_train_epochs,
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        learning_rate=sft_config.learning_rate,
        warmup_ratio=sft_config.warmup_ratio,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        bf16=sft_config.bf16,
        logging_steps=sft_config.logging_steps,
        save_steps=sft_config.save_steps,
        save_total_limit=sft_config.save_total_limit,
        report_to=sft_config.report_to,
        run_name=f"sft_{run_dir.name}",
        max_grad_norm=1.0,
        max_length=sft_config.max_seq_length,
        # Train only on assistant completions (not the user prompt)
        completion_only_loss=True,
    )

    # Init wandb
    wandb.init(
        project=sft_config.wandb_project,
        id=wandb_id,
        resume="allow",
        name=f"sft_{run_dir.name}",
    )

    # Build callbacks
    callbacks: list[TrainerCallback] = [StdoutLoggingCallback()]

    if sft_config.eval is not None and sft_config.eval_steps is not None:
        mid_run_eval_cb = MidRunEvalCallback(
            eval_config=sft_config.eval,
            eval_steps=sft_config.eval_steps,
            run_dir=str(run_dir),
            tokenizer=tokenizer,
        )
        callbacks.append(mid_run_eval_cb)
    elif sft_config.eval is None:
        print("WARNING: No eval configured (eval is null in config)")

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    print(f"\nTraining complete. Metrics: {train_result.metrics}")

    # Save final LoRA adapter
    trainer.save_model(str(run_dir / "checkpoints" / "final_adapter"))
    print(f"  final adapter saved: {run_dir / 'checkpoints' / 'final_adapter'}")

    # Plot loss
    plot_loss(log_history=trainer.state.log_history, run_dir=run_dir)

    wandb.finish()
    print(f"\n=== Done ===")
    print(f"  run_dir: {run_dir}")


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SFT training with TRL + LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # 'new' subcommand (default behavior when no subcommand)
    new_parser = subparsers.add_parser("new", help="Create and run a new SFT training run")
    new_parser.add_argument("--base-config", required=True, help="Path to base JSON5 config")
    new_parser.add_argument("--overrides", default=None, help="Path to overrides JSON5 config")
    new_parser.add_argument("--desc", required=True, help="Short description for run directory name")
    new_parser.add_argument("--sbatch", action="store_true", help="Generate sbatch script instead of running directly")
    new_parser.add_argument("--dont-auto-sbatch", action="store_true", help="Generate sbatch script without submitting")
    new_parser.add_argument("--time", default=None, help="SLURM time limit (overrides config sbatch_time)")

    # 'run-prepared' subcommand (called by sbatch script)
    prepared_parser = subparsers.add_parser("run-prepared", help="Run a previously prepared SFT run")
    prepared_parser.add_argument("--run-dir", required=True, help="Path to prepared run directory")
    prepared_parser.add_argument("--local-model-path", default=None, help="Override model path with node-local copy")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "new":
        if args.command is None:
            # Re-parse with 'new' as default
            parser = build_parser()
            args = parser.parse_args(["new"] + sys.argv[1:])

        # Load config
        sft_config = load_sft_config(
            base_config_path=args.base_config,
            overrides_path=args.overrides,
        )

        # Generate wandb ID
        wandb_id = wandb.util.generate_id()

        # Create run dir
        run_dir = create_run_dir(desc=args.desc, wandb_id=wandb_id)

        # Save resolved config + raw source configs (like vfh)
        merged = resolve_config(base_config_path=args.base_config, overrides_path=args.overrides)
        config_path = run_dir / "config.json5"
        with open(config_path, "w") as f:
            json.dump(merged, f, indent=2)

        # Copy raw config files into run dir for provenance
        import shutil
        configs_dir = run_dir / "source_configs"
        configs_dir.mkdir()
        shutil.copy2(args.base_config, configs_dir / "base_config.json5")
        if args.overrides:
            shutil.copy2(args.overrides, configs_dir / "overrides.json5")

        # Save metadata
        metadata = {
            "wandb_id": wandb_id,
            "base_config_path": str(Path(args.base_config).resolve()),
            "overrides_path": str(Path(args.overrides).resolve()) if args.overrides else None,
            "desc": args.desc,
            "created_at": datetime.now().isoformat(),
        }
        with open(run_dir / "run_metadata.json5", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSFT run prepared:")
        print(f"  run_dir: {run_dir}")
        print(f"  wandb_id: {wandb_id}")
        print(f"  model: {sft_config.model_name_or_path}")
        print(f"  train_files: {sft_config.train_files}")

        if args.sbatch:
            time_limit = args.time or sft_config.sbatch_time
            script_path = generate_sbatch(
                run_dir=run_dir,
                sft_config=sft_config,
                time_limit=time_limit,
                desc=args.desc,
                wandb_id=wandb_id,
            )
            if not args.dont_auto_sbatch:
                print("\nSubmitting sbatch job...")
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                )
                print(f"  sbatch output: {result.stdout.strip()}")
                if result.returncode != 0:
                    print(f"  sbatch error: {result.stderr.strip()}")
            else:
                print(f"\nSbatch script generated (not submitted): {script_path}")
        else:
            train(sft_config=sft_config, run_dir=run_dir, wandb_id=wandb_id)

    elif args.command == "run-prepared":
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Guard: refuse to run if training artifacts already exist (dir is immutable)
        _immutability_markers = ["loss_plot.png"]
        existing = [m for m in _immutability_markers if (run_dir / m).exists()]
        if existing:
            raise RuntimeError(
                f"Run directory already contains training artifacts: {existing}. "
                f"Each run must use its own directory. Use 'new' to create a fresh run."
            )

        # Load saved config
        config_path = run_dir / "config.json5"
        with open(config_path) as f:
            merged = pyjson5.load(f)
        sft_config = dacite.from_dict(
            data_class=SFTConfig,
            data=merged,
            config=dacite.Config(cast=[tuple], strict=True),
        )

        # Override model path if node-local copy provided
        if args.local_model_path:
            print(f"Using local model path: {args.local_model_path}")
            sft_config.model_name_or_path = args.local_model_path

        # Generate a fresh wandb ID for each execution (even retries of same run dir)
        wandb_id = wandb.util.generate_id()
        # Update metadata with new ID
        with open(run_dir / "run_metadata.json5") as f:
            metadata = pyjson5.load(f)
        metadata["wandb_id"] = wandb_id
        metadata["last_launched_at"] = datetime.now().isoformat()
        with open(run_dir / "run_metadata.json5", "w") as f:
            json.dump(metadata, f, indent=2)

        train(sft_config=sft_config, run_dir=run_dir, wandb_id=wandb_id)


if __name__ == "__main__":
    main()
