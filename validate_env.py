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
from typing import List, Tuple

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


def check_model(model_path: str) -> bool:
    """Check if model directory exists and contains required files."""
    print("\n3. Checking Model...")
    if not model_path:
        error("Model path not provided")
        return False
    
    path = Path(model_path)
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


def check_reward_function(reward_path: str, reward_name: str) -> bool:
    """Check if reward function file exists and contains the function."""
    print("\n4. Checking Reward Function...")
    if not reward_path:
        error("Reward path not provided")
        return False
    
    path = Path(reward_path)
    if not path.exists():
        error(f"Reward file does not exist: {reward_path}")
        return False
    
    if reward_name:
        try:
            with open(path, 'r') as f:
                content = f.read()
                if f"def {reward_name}" in content:
                    success(f"Reward file exists and function '{reward_name}' found: {reward_path}")
                else:
                    error(f"Reward file exists but function '{reward_name}' not found in {reward_path}")
                    return False
        except Exception as e:
            error(f"Error reading reward file: {e}")
            return False
    else:
        success(f"Reward file exists: {reward_path}")
    
    return True


def check_gpus(n_gpu: int) -> bool:
    """Check GPU availability using nvidia-smi."""
    print("\n5. Checking GPU Availability...")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = result.stdout.strip().split('\n')
        available_gpus = len(gpu_info)
        
        success(f"nvidia-smi found. Available GPUs: {available_gpus}")
        
        if available_gpus < n_gpu:
            error(f"Requested {n_gpu} GPUs but only {available_gpus} available")
            return False
        
        success(f"Sufficient GPUs available ({available_gpus} >= {n_gpu} requested)")
        
        print("   GPU Status:")
        for line in gpu_info:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                print(f"   GPU {parts[0]}: {parts[1]} | Memory: {parts[2]} / {parts[3]}")
        
        return True
        
    except FileNotFoundError:
        warning("nvidia-smi not found - cannot verify GPU availability")
        return True  # Don't fail, might be in SLURM allocation
    except subprocess.CalledProcessError as e:
        warning(f"nvidia-smi failed: {e}")
        return True


def check_ray() -> bool:
    """Check Ray cluster status."""
    print("\n6. Checking Ray Status...")
    try:
        result = subprocess.run(['ray', 'status'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            success("Ray cluster is running")
            # Print first 20 lines of status
            lines = result.stdout.split('\n')[:20]
            for line in lines:
                print(f"   {line}")
        else:
            warning("Ray command found but no cluster running (may start automatically)")
        return True
    except FileNotFoundError:
        warning("Ray not found in PATH")
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
        warning("No .env file found in current directory")
        return True
    
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
        warning(f"Could not read .env file: {e}")
    
    return True


def check_python_env() -> bool:
    """Check if required Python packages are available."""
    print("\n8. Checking Python Environment...")
    
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
    """Prompt user for yes/no confirmation."""
    while True:
        response = input(f"{YELLOW}{msg} (y/n): {NC}").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


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


def main():
    parser = argparse.ArgumentParser(description='Validate VERL experiment environment')
    parser.add_argument('--train-path', required=True, help='Path to training data')
    parser.add_argument('--test-path', required=True, help='Path to test data')
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--reward-path', required=True, help='Path to reward function file')
    parser.add_argument('--reward-name', required=True, help='Name of reward function')
    parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPUs required')
    parser.add_argument('--validate-parquet', action='store_true', help='Validate parquet files are readable')

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
        check_model(args.model_path),
        check_reward_function(args.reward_path, args.reward_name),
        check_gpus(args.n_gpu),
        check_ray(),
        check_env_file(),
        check_python_env(),
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