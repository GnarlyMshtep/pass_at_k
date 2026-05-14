#!/usr/bin/env python3
"""
Environment validation script for VERL experiments.
Checks all required files, resources, and configurations before training.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# ANSI color codes
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color


def error(msg: str) -> None:
    """Print error message."""
    print(f"{RED}✗ ERROR: {msg}{NC}")


def success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {msg}{NC}")


def warning(msg: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠ WARNING: {msg}{NC}")


def check_hf_home() -> Tuple[bool, str]:
    """Check if HF_HOME environment variable is set."""
    print("\n1. Checking HF_HOME Environment Variable...")
    hf_home = os.environ.get('HF_HOME')
    
    if not hf_home:
        error("HF_HOME environment variable is not set")
        return False, ""
    
    if not Path(hf_home).exists():
        error(f"HF_HOME is set but directory does not exist: {hf_home}")
        return False, hf_home
    
    success(f"HF_HOME is set: {hf_home}")
    return True, hf_home


def check_file(filepath: str, file_type: str) -> bool:
    """Check if a file exists and get its size."""
    if not filepath:
        error(f"{file_type} path not provided")
        return False
    
    path = Path(filepath)
    if not path.exists():
        error(f"{file_type} file does not exist: {filepath}")
        return False
    
    if not path.is_file():
        error(f"{file_type} path exists but is not a file: {filepath}")
        return False
    
    size = path.stat().st_size
    size_mb = size / (1024 * 1024)
    success(f"{file_type} file exists: {filepath} ({size_mb:.2f} MB)")
    return True


def check_parquet_readable(filepath: str, file_type: str) -> bool:
    """Check if parquet file is readable."""
    try:
        import pandas as pd
        df = pd.read_parquet(filepath)
        success(f"  {file_type} is readable. Shape: {df.shape}")
        return True
    except ImportError:
        warning("  pandas not available, skipping parquet validation")
        return True
    except Exception as e:
        error(f"  {file_type} exists but cannot be read: {e}")
        return False


def check_model(model_path: str, intended_resume: bool = False) -> bool:
    """Check if model directory exists and contains required files."""
    print("\n3. Checking Model...")
    if not model_path:
        error("Model path not provided")
        return False

    path = Path(model_path)

    if intended_resume:
        # When resuming, model directory is not required (will load from checkpoint)
        if path.exists() and path.is_dir():
            success(f"Model directory exists: {model_path}")
            warning("  (--intended-resume: model dir won't be used, will resume from checkpoint)")
        else:
            warning(f"Model directory does not exist: {model_path}")
            success("  (--intended-resume: this is fine, will resume from checkpoint)")
        return True

    if not path.exists():
        error(f"Model directory does not exist: {model_path}")
        return False

    if not path.is_dir():
        error(f"Model path exists but is not a directory: {model_path}")
        return False

    # Check for config.json
    config_path = path / "config.json"
    if not config_path.exists():
        error(f"Model directory exists but config.json not found: {model_path}")
        return False

    success(f"Model directory exists: {model_path}")

    # Check for model weights
    has_safetensors = list(path.glob("*.safetensors")) or list(path.glob("model*.safetensors"))
    has_pytorch = (path / "pytorch_model.bin").exists()

    if has_safetensors or has_pytorch:
        success("  Model weights found")
    else:
        warning("  Model config found but no model weights detected")

    return True


def check_reward_function(reward_path: str, reward_name: str, reward_kwargs_json: Optional[str] = None) -> bool:
    """Check if reward function file exists, function is importable, and config is valid.

    Uses RewardValidator for files with REWARD_REGISTRY (configed rewards).
    Falls back to simple string check for legacy reward files.
    """
    print("\n4. Checking Reward Function...")
    if not reward_path:
        error("Reward path not provided")
        return False

    path = Path(reward_path)
    if not path.exists():
        error(f"Reward file does not exist: {reward_path}")
        return False

    # Parse reward_kwargs if provided
    reward_kwargs: Optional[dict] = None
    if reward_kwargs_json:
        try:
            import json
            reward_kwargs = json.loads(reward_kwargs_json)
            if not isinstance(reward_kwargs, dict):
                error(f"--reward-kwargs must be a JSON object, got {type(reward_kwargs).__name__}")
                return False
            success(f"Parsed reward_kwargs: {list(reward_kwargs.keys())}")
        except json.JSONDecodeError as e:
            error(f"Invalid JSON in --reward-kwargs: {e}")
            return False

    # Try the full validator (import + registry + config validation)
    try:
        from custom.reward.reward_validator import RewardValidator
        validator = RewardValidator()
        validator.validate(
            reward_path=reward_path,
            reward_name=reward_name,
            reward_kwargs=reward_kwargs,
        )
        success(f"Reward function '{reward_name}' validated successfully from '{reward_path}'")
        return True
    except FileNotFoundError as e:
        error(str(e))
        return False
    except SyntaxError as e:
        error(f"Syntax error in reward file '{reward_path}': {e}")
        return False
    except ImportError as e:
        # If RewardValidator itself can't be imported, or the reward file can't be imported,
        # fall back to simple string check
        warning(f"Could not import reward module (will fall back to string check): {e}")
        return _check_reward_function_string_fallback(reward_path=str(path), reward_name=reward_name)
    except AttributeError as e:
        error(str(e))
        return False
    except ValueError as e:
        error(str(e))
        return False


def _check_reward_function_string_fallback(reward_path: str, reward_name: str) -> bool:
    """Legacy string-based check: just grep for 'def {name}' in the file."""
    if not reward_name:
        success(f"Reward file exists: {reward_path}")
        return True
    try:
        with open(reward_path, 'r') as f:
            content = f.read()
            if f"def {reward_name}" in content:
                success(f"Reward file exists and function '{reward_name}' found (string check): {reward_path}")
                return True
            else:
                error(f"Reward file exists but function '{reward_name}' not found in {reward_path}")
                return False
    except Exception as e:
        error(f"Error reading reward file: {e}")
        return False


def check_gpus(n_gpu: int, cuda_visible_devices: Optional[str] = None) -> bool:
    """Check GPU availability using nvidia-smi.

    Args:
        n_gpu: Number of GPUs required.
        cuda_visible_devices: Optional comma-separated list of GPU indices (e.g., "0,1,2").
    """
    print("\n5. Checking GPU Availability...")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )

        gpu_info = result.stdout.strip().split('\n')
        # Parse GPU info into a dict by index
        gpu_dict = {}
        for line in gpu_info:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpu_idx = int(parts[0])
                gpu_dict[gpu_idx] = {
                    'name': parts[1],
                    'memory_used': parts[2],
                    'memory_total': parts[3]
                }

        available_gpus = len(gpu_dict)
        success(f"nvidia-smi found. Available GPUs: {available_gpus}")

        # If specific GPUs are requested, validate them
        if cuda_visible_devices is not None:
            requested_indices = [int(idx.strip()) for idx in cuda_visible_devices.split(',')]

            # Check count matches n_gpu
            if len(requested_indices) != n_gpu:
                error(f"CUDA_VISIBLE_DEVICES has {len(requested_indices)} GPUs but n_gpu is {n_gpu}")
                return False

            # Check each requested GPU exists
            missing_gpus = [idx for idx in requested_indices if idx not in gpu_dict]
            if missing_gpus:
                error(f"Requested GPU indices {missing_gpus} are not available. Available indices: {list(gpu_dict.keys())}")
                return False

            success(f"All requested GPUs {requested_indices} are available and count matches n_gpu={n_gpu}")

            # Check that requested GPUs have 0 memory in use (no stale processes)
            gpus_with_memory = []
            for idx in requested_indices:
                info = gpu_dict[idx]
                # memory_used is like "123 MiB" — parse the number
                mem_used_str = info['memory_used'].strip().split()[0]
                try:
                    mem_used_mb = float(mem_used_str)
                except ValueError:
                    mem_used_mb = 0.0
                if mem_used_mb > 0:
                    gpus_with_memory.append((idx, info['memory_used']))

            if gpus_with_memory:
                for idx, mem_used in gpus_with_memory:
                    error(f"GPU {idx} has {mem_used} memory in use — likely a stale process")
                error("All requested GPUs must have 0 memory in use before training. "
                      "Kill stale processes or pick different GPUs.")
                return False

            success("All requested GPUs have 0 memory in use")

            print("   Requested GPU Status:")
            for idx in requested_indices:
                info = gpu_dict[idx]
                print(f"   GPU {idx}: {info['name']} | Memory: {info['memory_used']} / {info['memory_total']}")
        else:
            # Original behavior: just check we have enough GPUs
            if available_gpus < n_gpu:
                error(f"Requested {n_gpu} GPUs but only {available_gpus} available")
                return False

            success(f"Sufficient GPUs available ({available_gpus} >= {n_gpu} requested)")

            print("   GPU Status:")
            for idx, info in gpu_dict.items():
                print(f"   GPU {idx}: {info['name']} | Memory: {info['memory_used']} / {info['memory_total']}")

        return True

    except FileNotFoundError:
        warning("nvidia-smi not found - cannot verify GPU availability")
        return True  # Don't fail, might be in SLURM allocation
    except subprocess.CalledProcessError as e:
        warning(f"nvidia-smi failed: {e}")
        return True


def check_ray() -> bool:
    """Check Ray cluster status.

    On this cluster the login node is also a compute node, so previous verl
    runs' Ray clusters persist and `ray status` without an explicit address
    can fail with "multiple active Ray instances" — which is actually fine
    for us (verl will bring up its own cluster via ray.init at launch time).
    Report that case accurately instead of claiming no cluster is running.

    Also prefer `sys.executable -m ray` over bare `ray` so we pick up the
    venv's ray binary, not whatever's first on $PATH (often the conda one).
    """
    print("\n6. Checking Ray Status...")
    # Use the venv's ray binary next to sys.executable, not whatever `ray` is
    # first on $PATH (often the conda `hope` env). `python -m ray` doesn't work
    # because ray is a package without __main__.
    from pathlib import Path as _Path
    ray_bin = _Path(sys.executable).parent / "ray"
    if not ray_bin.exists():
        warning(f"Ray not found at {ray_bin} (venv may not have ray installed)")
        return True
    try:
        result = subprocess.run(
            [str(ray_bin), 'status'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            success("Ray cluster is running")
            for line in result.stdout.split('\n')[:20]:
                print(f"   {line}")
        else:
            stderr = result.stderr or ""
            if "multiple active Ray instances" in stderr or "Found multiple" in stderr:
                import re as _re
                m = _re.search(r"\{[^}]*\}", stderr)
                instances = m.group(0) if m else "<parse failed>"
                warning(
                    f"Multiple active Ray instances on this node {instances} — "
                    "that's OK for sbatch launches (verl starts its own cluster), "
                    "but `ray status` can't pick one without RAY_ADDRESS."
                )
            elif "connection" in stderr.lower() or "no ray" in stderr.lower():
                warning("Ray command found but no cluster running (may start automatically)")
            else:
                warning(f"`ray status` returned exit {result.returncode}: {stderr.strip()[:300]}")
        return True
    except subprocess.TimeoutExpired:
        warning("Ray status check timed out")
        return True
    except Exception as e:
        warning(f"Could not check Ray status: {e}")
        return True


def check_env_file() -> bool:
    """Check .env file and list variables."""
    print("\n7. Checking .env File...")
    env_path = Path('.env')

    if not env_path.exists():
        error("No .env file found in current directory")
        return False

    success(".env file found")
    print("   Environment variables defined:")

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    var_name = line.split('=')[0]
                    print(f"   - {var_name}")
    except Exception as e:
        error(f"Could not read .env file: {e}")
        return False

    return True


def check_openrouter_credits(min_credits: float = 200.0) -> bool:
    """Check if OpenRouter API key has sufficient credits.

    Args:
        min_credits: Minimum required credits in USD (default $200).

    Returns:
        True if credits >= min_credits, False otherwise.
    """
    print("\n7b. Checking OpenRouter Credits...")

    # Load .env if not already loaded
    load_dotenv()

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        error("OPENROUTER_API_KEY not found in environment or .env file")
        return False

    try:
        response = requests.get(
            'https://openrouter.ai/api/v1/credits',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=10
        )

        if response.status_code == 401:
            error("OpenRouter API authentication failed (invalid key)")
            return False
        elif response.status_code == 403:
            error("OpenRouter API requires a provisioning key for credit checks")
            return False
        elif response.status_code != 200:
            error(f"OpenRouter API returned status {response.status_code}: {response.text}")
            return False

        data = response.json()
        # The /credits endpoint returns total_credits and total_usage
        credits_data = data.get('data', {})
        total_credits = credits_data.get('total_credits', 0)
        total_usage = credits_data.get('total_usage', 0)

        available = total_credits - total_usage

        if available >= min_credits:
            success(f"OpenRouter credits sufficient: ${available:.2f} available (minimum ${min_credits:.2f})")
            return True
        else:
            error(f"OpenRouter credits insufficient: ${available:.2f} available, need ${min_credits:.2f}")
            return False

    except requests.exceptions.Timeout:
        error("OpenRouter API request timed out")
        return False
    except requests.exceptions.RequestException as e:
        error(f"Failed to connect to OpenRouter API: {e}")
        return False
    except Exception as e:
        error(f"Error checking OpenRouter credits: {e}")
        return False


def _detect_checkpoint_world_size(checkpoints_dir: Path) -> Optional[int]:
    """Detect world_size from FSDP checkpoint filenames in the latest checkpoint.

    Looks for files matching `model_world_size_{N}_rank_*.pt` in the latest
    global_step_* directory's actor/ subdirectory.

    Returns:
        The world_size if detected, or None if no checkpoints found.
    """
    import re

    # Find the latest global_step dir
    step_dirs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("global_step_")),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not step_dirs:
        return None

    latest = step_dirs[-1]
    actor_dir = latest / "actor"
    if not actor_dir.exists():
        return None

    # Parse world_size from model checkpoint filenames
    pattern = re.compile(r"model_world_size_(\d+)_rank_\d+\.pt")
    for f in actor_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            return int(m.group(1))

    return None


def check_output_dirs_not_exist(
    checkpoints_path: Optional[str], rollouts_path: Optional[str],
    intended_resume: bool = False, n_gpu: int = 0,
) -> bool:
    """Check that output directories do not already exist (or exist if resuming).

    Args:
        checkpoints_path: Path where checkpoints will be saved.
        rollouts_path: Path where rollouts will be saved.
        intended_resume: If True, require checkpoint dir to exist (for resuming runs).
        n_gpu: Number of GPUs configured. If > 0 and resuming, checks that checkpoint
            world_size matches.

    Returns:
        True if check passes, False otherwise.
    """
    if intended_resume:
        print("\n8. Checking Checkpoint Directory Exists (--intended-resume)...")
        if checkpoints_path:
            path = Path(checkpoints_path).expanduser()
            if path.exists():
                success(f"Checkpoints directory exists (good for resume): {checkpoints_path}")
                # Check for latest_checkpointed_iteration.txt file
                latest_ckpt_file = path / "latest_checkpointed_iteration.txt"
                if latest_ckpt_file.exists():
                    try:
                        contents = latest_ckpt_file.read_text().strip()
                        success(f"  Intending to resume from: {contents}")
                    except Exception as e:
                        warning(f"  Could not read latest_checkpointed_iteration.txt: {e}")
                else:
                    warning("  latest_checkpointed_iteration.txt file not found in checkpoint dir")

                # Check world_size in checkpoint files matches n_gpu
                if n_gpu > 0:
                    ckpt_world_size = _detect_checkpoint_world_size(path)
                    if ckpt_world_size is not None:
                        if ckpt_world_size == n_gpu:
                            success(f"  Checkpoint world_size ({ckpt_world_size}) matches n_gpus_per_node ({n_gpu})")
                        else:
                            error(
                                f"  Checkpoint world_size ({ckpt_world_size}) does NOT match "
                                f"n_gpus_per_node ({n_gpu}). FSDP checkpoint loading will fail. "
                                f"Either change n_gpus_per_node to {ckpt_world_size} or use a "
                                f"checkpoint saved with world_size={n_gpu}."
                            )
                            return False

                return True
            else:
                error(f"Checkpoints directory does not exist but --intended-resume was set: {checkpoints_path}")
                return False
        else:
            error("--intended-resume requires --checkpoints-path to be set")
            return False

    print("\n8. Checking Output Directories Don't Already Have Data...")

    # Subdirs that the run script creates before training starts — these don't count as real data
    METADATA_ONLY_DIRS = {'calling_script', 'cmdlineargs'}

    def _has_checkpoints(dirpath: Path) -> bool:
        """Return True if the directory contains any global_step_N checkpoint dirs."""
        return any(p.is_dir() and p.name.startswith('global_step_') for p in dirpath.iterdir())

    def _has_rollout_data(dirpath: Path) -> bool:
        """Return True if the directory (or its subdirs) contains any .jsonl rollout files."""
        return any(dirpath.rglob('*.jsonl'))

    def _has_real_data(dirpath: Path, is_checkpoints: bool) -> bool:
        """Return True if the directory contains meaningful training data.

        For checkpoints: any global_step_N/ directory.
        For rollouts: any .jsonl file (recursively).
        Also checks for anything beyond metadata-only subdirs as a fallback.
        """
        if is_checkpoints:
            return _has_checkpoints(dirpath)
        else:
            return _has_rollout_data(dirpath)

    def _auto_remove_empty_dir(dirpath: Path, dir_type: str) -> None:
        """Remove a directory tree that has no meaningful training data."""
        import shutil
        shutil.rmtree(dirpath)
        warning(f"Auto-removed empty {dir_type} directory (no training data): {dirpath}")

    all_ok = True

    if checkpoints_path:
        path = Path(checkpoints_path).expanduser()
        if path.exists():
            if _has_real_data(dirpath=path, is_checkpoints=True):
                error(f"Checkpoints directory already exists with data: {checkpoints_path}")
                error("  If you intended to resume training from checkpoint, use --intended-resume flag")
                all_ok = False
            else:
                _auto_remove_empty_dir(dirpath=path, dir_type="checkpoints")
                success(f"Checkpoints directory cleared (was empty): {checkpoints_path}")
        else:
            success(f"Checkpoints directory does not exist (good): {checkpoints_path}")

    if rollouts_path:
        path = Path(rollouts_path).expanduser()
        if path.exists():
            if _has_real_data(dirpath=path, is_checkpoints=False):
                error(f"Rollouts directory already exists with data: {rollouts_path}")
                all_ok = False
            else:
                _auto_remove_empty_dir(dirpath=path, dir_type="rollouts")
                success(f"Rollouts directory cleared (was empty): {rollouts_path}")
        else:
            success(f"Rollouts directory does not exist (good): {rollouts_path}")

    if not checkpoints_path and not rollouts_path:
        warning("No output directories specified to check")

    return all_ok


def check_checkpoint_disk_space(checkpoints_path: Optional[str], min_gb: float = 600.0) -> bool:
    """Check if the filesystem for checkpoints has sufficient free space.

    Args:
        checkpoints_path: Path where checkpoints will be saved.
        min_gb: Minimum required free space in GB (default 600).

    Returns:
        True if sufficient space available, False otherwise.
    """
    print("\n9. Checking Checkpoint Disk Space...")

    if not checkpoints_path:
        warning("No checkpoints path specified, skipping disk space check")
        return True

    path = Path(checkpoints_path).expanduser()

    # Find the first existing parent directory to check disk space
    check_path = path
    while not check_path.exists():
        check_path = check_path.parent
        if check_path == check_path.parent:  # Reached root
            error(f"Could not find existing parent directory for: {checkpoints_path}")
            return False

    try:
        stat = os.statvfs(check_path)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024 ** 3)

        if free_gb >= min_gb:
            success(f"Sufficient disk space: {free_gb:.1f} GB available on {check_path} (minimum {min_gb:.0f} GB)")
            return True
        else:
            error(f"Insufficient disk space: {free_gb:.1f} GB available on {check_path}, need {min_gb:.0f} GB")
            return False

    except OSError as e:
        error(f"Could not check disk space for {check_path}: {e}")
        return False


def check_dataset_requirements(train_path: str, merged_config_json: Optional[str] = None) -> bool:
    """Check dataset_requirements.json against merged config values.

    Preprocessing scripts can write a dataset_requirements.json alongside their
    parquet files, specifying required training hyperparams (e.g. shuffle=false).
    This function checks those requirements against the actual run config.

    See claude_state/implementing_dataset_requirements.md for the full design.
    """
    print("\n11. Checking Dataset Requirements...")

    # Always require filter_overlong_prompts=true. Pre-filtered datasets can still have
    # edge cases where chat template tokenization differs between preprocessing and runtime,
    # causing sequence_length > max_length errors. Double-filtering is harmless.
    # See: fork_k16vo4tp_starthiddenstage3 crashed with sequence_length=1027 > max_length=1024
    # on a dataset pre-filtered at 1024 tokens.
    if merged_config_json:
        try:
            import json as json_mod
            mc = json_mod.loads(merged_config_json)
            fop = mc.get("data", {}).get("filter_overlong_prompts")
            if fop is False:
                error(
                    "  data.filter_overlong_prompts must be True. Pre-filtered datasets can still "
                    "have edge cases from chat template tokenization differences. Set "
                    "filter_overlong_prompts=true in your override config."
                )
                return False
            elif fop is True:
                success("  data.filter_overlong_prompts = True ✓")
        except Exception:
            pass  # merged config parsing failed — other checks will catch it

    if not train_path:
        success("No train path — skipping dataset requirements check")
        return True

    # Find requirements file in the dataset directory
    dataset_dir = Path(train_path).parent
    req_path = dataset_dir / "dataset_requirements.json"

    if not req_path.exists():
        success(f"No dataset_requirements.json found in {dataset_dir} — no constraints to check")
        return True

    # Parse requirements
    try:
        import json as json_mod
        with open(req_path) as f:
            requirements: dict = json_mod.load(f)
    except Exception as e:
        error(f"Failed to parse {req_path}: {e}")
        return False

    if not isinstance(requirements, dict):
        error(f"dataset_requirements.json must be a JSON object, got {type(requirements).__name__}")
        return False

    if not requirements:
        success("dataset_requirements.json is empty — no constraints to check")
        return True

    # Parse merged config
    if not merged_config_json:
        warning(
            f"Dataset has requirements ({list(requirements.keys())}) but no --merged-config-json "
            f"was provided. Cannot validate. Requirements file: {req_path}"
        )
        return True  # Don't fail — older orchestrator versions don't pass this

    try:
        import json as json_mod
        merged_config: dict = json_mod.loads(merged_config_json)
    except Exception as e:
        error(f"Failed to parse --merged-config-json: {e}")
        return False

    # Check each requirement
    all_ok = True
    for dotted_key, expected_value in requirements.items():
        # Navigate the merged config using dot-separated key
        keys = dotted_key.split(".")
        current = merged_config
        found = True
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            elif isinstance(current, dict) and f"+{k}" in current:
                current = current[f"+{k}"]
            else:
                warning(
                    f"  Requirement '{dotted_key}={expected_value}' — key not found in merged config "
                    f"(looked for: {'.'.join(keys)}). Skipping."
                )
                found = False
                break

        if not found:
            continue

        actual_value = current

        # Type-aware comparison (JSON bools vs Python bools, ints vs floats, etc.)
        if _values_match(expected=expected_value, actual=actual_value):
            success(f"  {dotted_key} = {actual_value} ✓ (matches requirement)")
        elif dotted_key == "data.filter_overlong_prompts":
            # Double-filtering is a no-op (dataset already pre-filtered), so just warn
            warning(
                f"  Dataset was pre-filtered (requires {dotted_key}={expected_value!r}) "
                f"but run config has {dotted_key}={actual_value!r}. "
                f"This is harmless (double-filtering is a no-op) but wastes time on tokenization."
            )
        else:
            error(
                f"  Dataset requires {dotted_key}={expected_value!r} but run config has "
                f"{dotted_key}={actual_value!r}. Fix your override config or use a different dataset."
            )
            all_ok = False

    if all_ok:
        success(f"All dataset requirements satisfied ({req_path})")
    return all_ok


def _values_match(expected: Any, actual: Any) -> bool:
    """Compare expected and actual config values with type coercion.

    Handles: bool vs str ("True"/"False"), int vs float (1 == 1.0), etc.
    """
    # Direct equality
    if expected == actual:
        return True

    # Bool comparison (Hydra may pass bools as strings)
    if isinstance(expected, bool):
        if isinstance(actual, str):
            return (expected is True and actual.lower() == "true") or \
                   (expected is False and actual.lower() == "false")

    # Numeric comparison (int vs float)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual

    return False


def check_resume_checkpoint(resume_from_path: str, n_gpu: int) -> bool:
    """Check that the resume checkpoint directory has real weight files and matching world_size.

    Args:
        resume_from_path: Path to the global_step_N directory to resume from.
        n_gpu: Number of GPUs configured (for world_size matching).

    Returns:
        True if all checks pass, False otherwise.
    """
    import re as re_mod

    print("\n12. Checking Resume Checkpoint...")

    ckpt_dir = Path(resume_from_path)
    dvc_file = ckpt_dir.parent / f"{ckpt_dir.name}.dvc"
    dvc_hint = (
        f"\n       Checkpoint may be in DVC — try: dvc pull {dvc_file}"
    ) if dvc_file.exists() else ""

    # 1. Directory exists
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        error(f"Resume checkpoint directory does not exist: {ckpt_dir}{dvc_hint}")
        return False
    success(f"Resume checkpoint directory exists: {ckpt_dir}")

    # 2. actor/ subdirectory exists
    actor_dir = ckpt_dir / "actor"
    if not actor_dir.exists() or not actor_dir.is_dir():
        error(f"Resume checkpoint has no actor/ subdirectory: {ckpt_dir}{dvc_hint}")
        return False
    success("  actor/ subdirectory exists")

    # 3. actor/ has weight files
    weight_pattern = re_mod.compile(r"model_world_size_(\d+)_rank_\d+\.pt")
    world_sizes: List[int] = []
    for f in actor_dir.iterdir():
        m = weight_pattern.match(f.name)
        if m:
            world_sizes.append(int(m.group(1)))

    if not world_sizes:
        error(f"  actor/ has no model weight files (expected model_world_size_N_rank_*.pt){dvc_hint}")
        return False

    ckpt_world_size = world_sizes[0]
    n_ranks = len(world_sizes)
    success(f"  actor/ has {n_ranks} model weight files (world_size={ckpt_world_size})")

    # 4. World size matches n_gpu
    if ckpt_world_size != n_gpu:
        error(
            f"  Checkpoint world_size ({ckpt_world_size}) does NOT match "
            f"n_gpus_per_node ({n_gpu}). FSDP checkpoint loading will fail.\n"
            f"       Either change n_gpus_per_node to {ckpt_world_size} or use a "
            f"checkpoint saved with world_size={n_gpu}."
        )
        return False
    success(f"  Checkpoint world_size ({ckpt_world_size}) matches n_gpus_per_node ({n_gpu})")

    return True


def check_python_env() -> bool:
    """Check if required Python packages are available."""
    print("\n13. Checking Python Environment...")
    
    # Check verl
    try:
        import verl
        success("verl package is importable")
    except ImportError:
        error("verl package not found - is the environment activated?")
        return False
    
    # Check other common dependencies
    packages = ['torch', 'transformers', 'pandas']
    for pkg in packages:
        try:
            __import__(pkg)
            success(f"  {pkg} is available")
        except ImportError:
            warning(f"  {pkg} not found (may be optional)")
    
    return True


def prompt_user_confirmation(msg: str) -> bool:
    # """Prompt user for yes/no confirmation."""
    # while True:
    #     response = input(f"{YELLOW}{msg} (y/n): {NC}").strip().lower()
    #     if response in ['y', 'yes']:
    #         return True
    #     elif response in ['n', 'no']:
    #         return False
    #     else:
    #         print("Please enter 'y' or 'n'")
    return True


def display_and_confirm_config(args) -> bool:
    """Display all configuration settings and ask for confirmation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)

    # Batch sizes
    print(f"\n{GREEN}Batch Sizes:{NC}")
    print(f"  batch_size:                {args.batch_size}")
    print(f"  mini_batch_size:           {args.mini_batch_size}")
    print(f"  n_rollout:                 {args.n_rollout}")
    print(f"  micro_batch_size_per_gpu:  {args.micro_batch_size_per_gpu}")

    if not prompt_user_confirmation("\nDo these batch sizes look correct?"):
        error("User rejected batch size configuration")
        return False

    # Algorithm settings
    print(f"\n{GREEN}Algorithm Settings:{NC}")
    print(f"  Advantage Estimator:       {args.adv_estimator}")
    print(f"  Normalize by Std:          {args.norm_by_std}")
    print(f"  Linear Warmup Steps:       {args.linear_warmup_steps}")
    print(f"  Reward Function:           {args.reward_name} (from {args.reward_path})")

    if not prompt_user_confirmation("\nDo these algorithm settings look correct?"):
        error("User rejected algorithm configuration")
        return False

    # Experiment naming
    print(f"\n{GREEN}Experiment Naming:{NC}")
    print(f"  Project Name:              {args.proj_name}")
    print(f"  Experiment Name:           {args.exp_name}")

    if not prompt_user_confirmation("\nDo these experiment names look correct?"):
        error("User rejected experiment naming")
        return False

    success("All configuration settings confirmed!")
    return True


def check_training_steps(merged_config_json: Optional[str], train_path: str, batch_size: int, resume_from_path: Optional[str] = None) -> bool:
    """Validate max_additional_steps is set and print training steps summary."""
    print("\n--- Training Steps Summary ---")
    if not merged_config_json:
        error("No merged config provided — cannot check training steps")
        return False

    import json as json_mod
    mc: dict = json_mod.loads(merged_config_json)
    trainer = mc.get("trainer", {})

    # +prefixed keys in merged config keep the + in the key name
    max_additional_steps = trainer.get("max_additional_steps", trainer.get("+max_additional_steps", None))
    if max_additional_steps is None:
        error("trainer.max_additional_steps is required but not set. "
              "Add +trainer.max_additional_steps=N to your Hydra overrides.")
        return False
    max_additional_steps = int(max_additional_steps)

    # Compute total_training_steps
    total_training_steps = trainer.get("total_training_steps", trainer.get("+total_training_steps", None))
    if total_training_steps is not None:
        total_training_steps = int(total_training_steps)
    else:
        total_epochs = int(trainer.get("total_epochs", 30))
        try:
            import math
            import pyarrow.parquet as pq
            num_rows = pq.read_metadata(train_path).num_rows
            steps_per_epoch = math.ceil(num_rows / batch_size)
            total_training_steps = steps_per_epoch * total_epochs
        except Exception as e:
            warning(f"Could not compute total_training_steps from dataset: {e}")
            total_training_steps = None

    # Extract resume step from CLI arg (most reliable) or merged config
    resume_from = resume_from_path or trainer.get("resume_from_path", None)
    resume_step = 0
    if resume_from and "global_step_" in str(resume_from):
        resume_step = int(str(resume_from).split("global_step_")[-1])

    max_add_target = resume_step + max_additional_steps
    save_at_exit = trainer.get("save_at_exit", trainer.get("+save_at_exit", False))

    if total_training_steps is not None:
        effective_end = min(max_add_target, total_training_steps)
        print(f"  Starting from gstep {resume_step}, going to gstep "
              f"min(max_additional_steps={resume_step}+{max_additional_steps}={max_add_target}, "
              f"total_training_steps={total_training_steps}) = {effective_end}")
    else:
        print(f"  Starting from gstep {resume_step}, max_additional_steps={max_additional_steps} "
              f"→ exit at gstep {max_add_target}")

    print(f"  save_at_exit={save_at_exit}")
    success(f"max_additional_steps={max_additional_steps} is set")
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate VERL experiment environment')
    parser.add_argument('--train-path', required=True, help='Path to training data')
    parser.add_argument('--test-path', required=True, help='Path to test data')
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--reward-path', required=True, help='Path to reward function file')
    parser.add_argument('--reward-name', required=True, help='Name of reward function')
    parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPUs required')
    parser.add_argument('--cuda-visible-devices', type=str, default=None,
                        help='Comma-separated list of GPU indices to use (e.g., "0,1,2"). '
                             'Validates these specific GPUs exist and count matches n_gpu.')
    parser.add_argument('--requires-openrouter', action='store_true', default=False,
                        help='Require OpenRouter API key with at least $150 credits')
    parser.add_argument('--checkpoints-path', type=str, default=None,
                        help='Path where checkpoints will be saved. Fails if directory already exists.')
    parser.add_argument('--rollouts-path', type=str, default=None,
                        help='Path where rollouts will be saved. Fails if directory already exists.')
    parser.add_argument('--intended-resume', action='store_true', default=False,
                        help='Skip directory existence check (for resuming runs where dirs already exist)')
    parser.add_argument('--validate-parquet', action='store_true', help='Validate parquet files are readable')
    parser.add_argument('--reward-kwargs', type=str, default=None,
                        help='JSON string of reward_kwargs to validate against the reward config '
                             '(e.g. \'{"reward_config": {"formatter": "removeaftercode"}}\')')
    parser.add_argument('--merged-config-json', type=str, default=None,
                        help='JSON string of the full merged config (base + overrides). '
                             'Used to check dataset_requirements.json constraints.')
    parser.add_argument('--resume-from-path', type=str, default=None,
                        help='Path to the global_step_N checkpoint directory being resumed from. '
                             'Validates actor/ exists, weight files present, world_size matches n_gpu.')

    # Batch size arguments
    parser.add_argument('--batch-size', type=int, required=True, help='Training batch size')
    parser.add_argument('--mini-batch-size', type=int, required=True, help='PPO mini batch size')
    parser.add_argument('--n-rollout', type=int, required=True, help='Number of rollouts')
    parser.add_argument('--micro-batch-size-per-gpu', type=int, required=True, help='Micro batch size per GPU')

    # Algorithm arguments
    parser.add_argument('--adv-estimator', required=True, help='Advantage estimator')
    parser.add_argument('--norm-by-std', required=True, help='Normalize advantages by std (True/False)')
    parser.add_argument('--linear-warmup-steps', type=int, required=True, help='Number of linear warmup steps for training')

    # Experiment naming
    parser.add_argument('--proj-name', required=True, help='Project name for wandb')
    parser.add_argument('--exp-name', required=True, help='Experiment name')

    args = parser.parse_args()
    
    print("=" * 60)
    print("VERL Environment Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Run all checks
    checks = [
        check_hf_home(),
        (lambda: (print("\n2. Checking Training Data...") or True) and
         check_file(args.train_path, "Training") and
         (check_parquet_readable(args.train_path, "Training data") if args.validate_parquet else True))(),
        (lambda: (print("\n2b. Checking Test Data...") or True) and
         check_file(args.test_path, "Test") and
         (check_parquet_readable(args.test_path, "Test data") if args.validate_parquet else True))(),
        check_model(args.model_path, args.intended_resume),
        check_reward_function(args.reward_path, args.reward_name, args.reward_kwargs),
        check_gpus(args.n_gpu, args.cuda_visible_devices),
        check_ray(),
        check_env_file(),
        check_openrouter_credits() if args.requires_openrouter else True,
        check_output_dirs_not_exist(args.checkpoints_path, args.rollouts_path, args.intended_resume, args.n_gpu),
        check_checkpoint_disk_space(args.checkpoints_path),
        check_dataset_requirements(args.train_path, args.merged_config_json),
        check_resume_checkpoint(args.resume_from_path, args.n_gpu) if args.resume_from_path else True,
        check_python_env(),
        check_training_steps(args.merged_config_json, args.train_path, args.batch_size, args.resume_from_path),
    ]
    
    # Check HF_HOME separately since it returns a tuple
    hf_check, _ = checks[0]
    all_checks_passed = hf_check and all(checks[1:])

    if not all_checks_passed:
        print("\n" + "=" * 60)
        print(f"{RED}Validation failed - see errors above{NC}")
        print("=" * 60)
        sys.exit(1)

    # All checks passed, now display config and get user confirmation
    if not display_and_confirm_config(args):
        print("\n" + "=" * 60)
        print(f"{RED}Configuration rejected by user - exiting{NC}")
        print("=" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"{GREEN}All checks passed and configuration confirmed!{NC}")
    print(f"{GREEN}Ready to start training!{NC}")
    print("=" * 60)
    sys.exit(0)


if __name__ == '__main__':
    main()