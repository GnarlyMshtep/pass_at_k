"""Dataclasses for SFT training configuration.

No defaults — all values must come from config files (base + overrides).
"""

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """PEFT LoRA adapter configuration."""
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str
    task_type: str


@dataclass
class EvalConfig:
    """Eval configuration. eval_script points to a module with an SFTEval subclass.
    eval_kwargs is a flexible dict passed to the eval's __init__ — each eval type
    defines its own expected keys."""
    eval_script: str          # e.g. "TRLSFT.sft_eval_backdoor.BackdoorRewardEval"
    eval_kwargs: dict         # eval-specific params, passed to eval __init__


@dataclass
class SFTConfig:
    """Full SFT run configuration. Deserialized from base + overrides JSON5."""
    # Model
    model_name_or_path: str
    # LoRA
    lora: LoRAConfig
    # Data
    train_files: list[str]
    # Training
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_seq_length: int
    bf16: bool
    # Checkpointing
    save_steps: int
    save_total_limit: int
    # Logging
    logging_steps: int
    report_to: str
    wandb_project: str
    # Eval
    eval: EvalConfig
    eval_steps: int | None
    # Sbatch
    sbatch_time: str
    n_gpus: int


@dataclass
class CLIArgs:
    """CLI-only arguments (not part of the config files)."""
    base_config: str
    overrides: str | None
    desc: str
    sbatch: bool
    dont_auto_sbatch: bool
    time: str | None  # override sbatch_time
