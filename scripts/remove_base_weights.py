#!/usr/bin/env python3
"""
Script 1: Remove Base Model Weights from FSDP2 LoRA Checkpoints

This script:
1. Removes base model weights from FSDP2 checkpoints
2. Keeps LoRA adapter weights and other non-base-layer weights
3. Preserves DTensor structure for kept weights
4. Always saves DTensor metadata files for reconstruction (dtensor_metadata_rank_*.json)
5. Copies other necessary files (fsdp_config.json, huggingface/, lora_adapter/, etc.)

Usage:
    python scripts/remove_base_weights.py \
        --source /path/to/checkpoint/global_step_50/actor \
        [--dest /path/to/filtered/checkpoint/global_step_50/actor] \
        [--dry-run] \
        [--pattern "model_world_size_*_rank_*.pt"]

Note: If --dest is not provided, the script modifies the checkpoint in-place.
      Metadata files are always saved (required for reconstruction).
"""

import torch
import json
import shutil
from pathlib import Path
from collections import OrderedDict
import argparse
import os

try:
    import torch.distributed.tensor as dtensor
    DTensor = dtensor.DTensor
    DTENSOR_AVAILABLE = True
except ImportError:
    try:
        from torch.distributed._tensor import DTensor
        DTENSOR_AVAILABLE = True
    except ImportError:
        DTensor = None
        DTENSOR_AVAILABLE = False


def is_base_model_weight(key):
    """
    Check if a key represents a base model weight that should be removed.
    
    Base model weights match these patterns:
    - Contains 'base_layer' (e.g., 'base_model.model.layers.0.self_attn.q_proj.base_layer.weight')
    - Starts with 'base_model.model.model.embed_tokens' (embedding weights)
    - Starts with 'base_model.model.lm_head' (output head weights)
    """
    if 'base_layer' in key:
        return True
    if key.startswith('base_model.model.model.embed_tokens'):
        return True
    if key.startswith('base_model.model.lm_head'):
        return True
    return False


def has_lora_weights(checkpoint):
    """
    Check if checkpoint contains any LoRA weights.
    
    Args:
        checkpoint: Checkpoint dictionary
        
    Returns:
        True if LoRA weights are found, False otherwise
    """
    for key in checkpoint.keys():
        if 'lora_A' in key or 'lora_B' in key:
            return True
    return False


def filter_base_model_weights(checkpoint, keep_metadata=True):
    """
    Filter checkpoint to remove base model weights while preserving LoRA and other weights.
    
    Args:
        checkpoint: Checkpoint dictionary (OrderedDict)
        keep_metadata: If True, preserve DTensor structure (default: True)
        
    Returns:
        Filtered checkpoint dictionary
    """
    filtered = OrderedDict()
    
    removed_count = 0
    kept_count = 0
    
    for key, value in checkpoint.items():
        if is_base_model_weight(key):
            removed_count += 1
            continue
        
        # Keep all LoRA keys (lora_A, lora_B)
        # Keep all non-base-layer weights (layer norms, etc.)
        filtered[key] = value
        kept_count += 1
    
    return filtered, removed_count, kept_count


def calculate_shard_slice(global_shape, local_shape, placements, rank, world_size):
    """
    Calculate which slice of the global tensor this rank holds.
    
    Args:
        global_shape: Global tensor shape
        local_shape: Local shard shape
        placements: DTensor placements (e.g., (Shard(dim=0),))
        rank: Current rank
        world_size: Total number of ranks
        
    Returns:
        Dictionary with slice information
    """
    slices = []
    offsets = []
    shard_dim = None
    
    # First, find which dimension is sharded
    for placement in placements:
        if hasattr(placement, 'dim'):
            shard_dim = placement.dim
            break
    
    for dim in range(len(global_shape)):
        global_size = global_shape[dim]
        local_size = local_shape[dim]
        
        # Check if this dimension is sharded
        is_sharded = (dim == shard_dim)
        
        if is_sharded:
            # Calculate slice for this dimension
            shard_size = global_size // world_size
            start_idx = rank * shard_size
            end_idx = start_idx + local_size
            
            slices.append([start_idx, end_idx])
            offsets.append(start_idx)
        else:
            # Not sharded, use full dimension
            slices.append([0, global_size])
            offsets.append(0)
    
    return {
        'slices': slices,
        'offsets': offsets,
        'shard_dim': shard_dim
    }


def extract_and_save_metadata(checkpoint_file, rank, metadata_file):
    """
    Extract DTensor metadata from checkpoint and save to JSON file.
    
    Args:
        checkpoint_file: Path to checkpoint file
        rank: Rank number
        metadata_file: Path to save metadata JSON file
        
    Returns:
        Metadata dictionary
    """
    print(f"  Extracting metadata from: {checkpoint_file.name}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    
    if not isinstance(checkpoint, dict):
        if hasattr(checkpoint, '__dict__'):
            checkpoint = checkpoint.__dict__
        else:
            checkpoint = {'root': checkpoint}
    
    metadata = {
        'rank': rank,
        'world_size': None,
        'device_mesh': None,
        'placements': None,
        'tensors': {}
    }
    
    # Extract metadata from first DTensor
    first_dtensor = None
    for key, value in checkpoint.items():
        if isinstance(value, DTensor):
            first_dtensor = value
            break
    
    if first_dtensor is None:
        raise ValueError("No DTensors found in checkpoint")
    
    # Extract global metadata
    spec = first_dtensor._spec
    metadata['device_mesh'] = {
        'device_type': str(spec.device_mesh.device_type),
        'mesh': spec.device_mesh.mesh.tolist() if hasattr(spec.device_mesh.mesh, 'tolist') else list(spec.device_mesh.mesh),
        'mesh_dim_names': list(spec.device_mesh.mesh_dim_names) if hasattr(spec.device_mesh, 'mesh_dim_names') else ['fsdp']
    }
    
    # Extract placements
    placements_info = []
    for placement in spec.placements:
        if hasattr(placement, 'dim'):
            placements_info.append({
                'type': 'Shard',
                'dim': placement.dim
            })
        else:
            placements_info.append({
                'type': str(type(placement).__name__),
                'value': str(placement)
            })
    metadata['placements'] = placements_info
    
    # Determine world size from device mesh
    if hasattr(spec.device_mesh, 'mesh'):
        mesh = spec.device_mesh.mesh
        if hasattr(mesh, 'numel'):
            metadata['world_size'] = mesh.numel()
        elif hasattr(mesh, '__len__'):
            metadata['world_size'] = len(mesh)
        else:
            metadata['world_size'] = 2  # Default assumption
    
    # Extract per-tensor metadata
    for key, value in checkpoint.items():
        if isinstance(value, DTensor):
            spec = value._spec
            global_shape = spec.shape if hasattr(spec, 'shape') else value.shape
            local_shape = value._local_tensor.shape
            local_dtype = str(value._local_tensor.dtype)
            local_stride = value._local_tensor.stride()
            
            # Calculate shard slice
            slice_info = calculate_shard_slice(
                global_shape,
                local_shape,
                spec.placements,
                rank,
                metadata['world_size']
            )
            
            metadata['tensors'][key] = {
                'global_shape': list(global_shape),
                'local_shape': list(local_shape),
                'local_dtype': local_dtype,
                'local_stride': list(local_stride),
                'slices': slice_info['slices'],
                'offsets': slice_info['offsets'],
                'shard_dim': slice_info['shard_dim'],
                'is_base_model_weight': is_base_model_weight(key)
            }
        elif isinstance(value, torch.Tensor):
            # Regular tensor (not sharded)
            metadata['tensors'][key] = {
                'global_shape': list(value.shape),
                'local_shape': list(value.shape),
                'local_dtype': str(value.dtype),
                'local_stride': list(value.stride()),
                'slices': None,  # Not sharded
                'offsets': None,
                'shard_dim': None,
                'is_base_model_weight': is_base_model_weight(key),
                'is_regular_tensor': True
            }
    
    # Save metadata to JSON file
    metadata_file = Path(metadata_file)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved to: {metadata_file.name} ({metadata_file.stat().st_size / 1024:.2f} KB)")
    
    return metadata


def process_checkpoint_file(source_file, dest_file, rank, dry_run=False):
    """
    Process a single checkpoint file: filter base model weights and save metadata.
    
    Args:
        source_file: Source checkpoint file path
        dest_file: Destination checkpoint file path
        rank: Rank number (for metadata extraction)
        dry_run: If True, don't save files, just report
        
    Returns:
        Tuple of (filtered checkpoint dict, metadata dict, stats dict)
    """
    print(f"\nProcessing: {source_file.name}")
    
    # Load checkpoint
    checkpoint = torch.load(source_file, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint structures
    if not isinstance(checkpoint, dict):
        if hasattr(checkpoint, '__dict__'):
            checkpoint = checkpoint.__dict__
        else:
            checkpoint = {'root': checkpoint}
    
    original_size = source_file.stat().st_size
    original_keys = len(checkpoint)
    
    # Filter base model weights
    filtered, removed_count, kept_count = filter_base_model_weights(checkpoint, keep_metadata=True)
    
    # Always extract and save metadata (required for reconstruction)
    metadata_file = dest_file.parent / f"dtensor_metadata_rank_{rank}.json"
    if not dry_run:
        metadata = extract_and_save_metadata(source_file, rank, metadata_file)
    else:
        print(f"  [DRY RUN] Would save metadata to: {metadata_file.name}")
        metadata = None
    
    # Save filtered checkpoint
    if not dry_run:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(filtered, dest_file)
        new_size = dest_file.stat().st_size
        size_reduction = (1 - new_size / original_size) * 100
        print(f"  Saved filtered checkpoint: {dest_file.name}")
        print(f"  Size: {original_size / (1024**3):.3f} GB → {new_size / (1024**3):.3f} GB ({size_reduction:.1f}% reduction)")
    else:
        print(f"  [DRY RUN] Would save filtered checkpoint to: {dest_file.name}")
        print(f"  Original size: {original_size / (1024**3):.3f} GB")
    
    stats = {
        'original_keys': original_keys,
        'kept_keys': kept_count,
        'removed_keys': removed_count,
        'original_size': original_size,
        'filtered_size': dest_file.stat().st_size if not dry_run else 0
    }
    
    print(f"  Keys: {original_keys} → {kept_count} (removed {removed_count})")
    
    return filtered, metadata, stats


def process_checkpoint_directory(source_dir, dest_dir=None, pattern="model_world_size_*_rank_*.pt", dry_run=False):
    """
    Process all checkpoint files matching the pattern in source directory.
    
    Args:
        source_dir: Source directory containing checkpoint files
        dest_dir: Destination directory to save filtered checkpoints (None for in-place)
        pattern: Glob pattern to match checkpoint files
        dry_run: If True, don't save files, just report
    """
    source_dir = Path(source_dir)
    
    # If no destination provided, use source (in-place operation)
    if dest_dir is None:
        dest_dir = source_dir
        in_place = True
    else:
        dest_dir = Path(dest_dir)
        in_place = False
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Find all checkpoint files
    checkpoint_files = list(source_dir.glob(pattern))
    
    if not checkpoint_files:
        print(f"Warning: No checkpoint files found matching pattern '{pattern}' in {source_dir}")
        return
    
    # Check if any checkpoint has LoRA weights
    print("Checking for LoRA weights...")
    has_lora = False
    for src_file in sorted(checkpoint_files):
        checkpoint = torch.load(src_file, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            if hasattr(checkpoint, '__dict__'):
                checkpoint = checkpoint.__dict__
            else:
                checkpoint = {'root': checkpoint}
        
        if has_lora_weights(checkpoint):
            has_lora = True
            break
    
    if not has_lora:
        print("No LoRA weights found in checkpoint. Skipping (nothing to do).")
        return
    
    print("="*80)
    print("REMOVE BASE MODEL WEIGHTS FROM CHECKPOINT")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir} {'(in-place)' if in_place else ''}")
    print(f"Pattern: {pattern}")
    print(f"Dry run: {dry_run}")
    print(f"Found {len(checkpoint_files)} checkpoint file(s)")
    print("="*80)
    
    all_stats = []
    
    # Process each checkpoint file
    for src_file in sorted(checkpoint_files):
        # Extract rank from filename (e.g., "model_world_size_2_rank_0.pt" -> rank 0)
        try:
            rank_str = src_file.stem.split('_rank_')[1]
            rank = int(rank_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not extract rank from {src_file.name}, using 0")
            rank = 0
        
        # Determine destination filename (preserve original name)
        dest_file = dest_dir / src_file.name
        
        filtered, metadata, stats = process_checkpoint_file(
            src_file,
            dest_file,
            rank,
            dry_run=dry_run
        )
        
        all_stats.append(stats)
    
    # Copy other necessary files (skip if in-place operation)
    if not dry_run:
        print("\n" + "="*80)
        print("COPYING OTHER FILES")
        print("="*80)
        
        # Skip copying if in-place (source and dest are the same)
        if not in_place:
            files_to_copy = [
                'fsdp_config.json',
            ]
            
            # Copy optimizer and extra state files (they may reference base model weights, but we keep them for now)
            for rank in range(10):  # Check up to 10 ranks
                for prefix in ['optim', 'extra_state']:
                    pattern_file = source_dir.glob(f"{prefix}_world_size_*_rank_{rank}.pt")
                    for f in pattern_file:
                        files_to_copy.append(f.name)
            
            dirs_to_copy = [
                'huggingface',
                'lora_adapter',
            ]
            
            # Copy files
            for filename in files_to_copy:
                src_file = source_dir / filename
                if src_file.exists():
                    dest_file = dest_dir / filename
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    # Skip if source and destination are the same file
                    if src_file.resolve() != dest_file.resolve():
                        shutil.copy2(src_file, dest_file)
                        print(f"Copied: {filename}")
                    else:
                        print(f"Skipped (same file): {filename}")
            
            # Copy directories
            for dirname in dirs_to_copy:
                src_dir_path = source_dir / dirname
                if src_dir_path.exists():
                    dest_dir_path = dest_dir / dirname
                    # Skip if source and destination are the same directory
                    if src_dir_path.resolve() != dest_dir_path.resolve():
                        if dest_dir_path.exists():
                            shutil.rmtree(dest_dir_path)
                        shutil.copytree(src_dir_path, dest_dir_path)
                        print(f"Copied directory: {dirname}/")
                    else:
                        print(f"Skipped (same directory): {dirname}/")
            
            # Copy data.pt if it exists in parent directory
            parent_source = source_dir.parent
            data_file = parent_source / 'data.pt'
            if data_file.exists():
                parent_dest = dest_dir.parent
                parent_dest.mkdir(parents=True, exist_ok=True)
                dest_data_file = parent_dest / 'data.pt'
                # Skip if source and destination are the same file
                if data_file.resolve() != dest_data_file.resolve():
                    shutil.copy2(data_file, dest_data_file)
                    print(f"Copied: data.pt")
                else:
                    print(f"Skipped (same file): data.pt")
        else:
            print("Skipping file copy (in-place operation - files already in correct location)")
    else:
        print("\n[DRY RUN] Would copy other files (fsdp_config.json, huggingface/, lora_adapter/, etc.)")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_original_keys = sum(s['original_keys'] for s in all_stats)
    total_kept_keys = sum(s['kept_keys'] for s in all_stats)
    total_removed_keys = sum(s['removed_keys'] for s in all_stats)
    total_original_size = sum(s['original_size'] for s in all_stats)
    total_filtered_size = sum(s['filtered_size'] for s in all_stats)
    
    print(f"Total keys: {total_original_keys} → {total_kept_keys} (removed {total_removed_keys})")
    if not dry_run and total_filtered_size > 0:
        total_size_reduction = (1 - total_filtered_size / total_original_size) * 100
        print(f"Total size: {total_original_size / (1024**3):.3f} GB → {total_filtered_size / (1024**3):.3f} GB ({total_size_reduction:.1f}% reduction)")
    else:
        print(f"Total original size: {total_original_size / (1024**3):.3f} GB")
    
    print(f"\nFiltered checkpoints saved to: {dest_dir}")
    print(f"Metadata files saved to: {dest_dir} (dtensor_metadata_rank_*.json)")


def main():
    parser = argparse.ArgumentParser(
        description="Remove base model weights from FSDP2 LoRA checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove base weights in-place (modifies source checkpoint, saves metadata)
  python scripts/remove_base_weights.py \\
      --source /path/to/checkpoint/global_step_50/actor

  # Remove base weights and save to different location (metadata always saved)
  python scripts/remove_base_weights.py \\
      --source /path/to/checkpoint/global_step_50/actor \\
      --dest /path/to/filtered/checkpoint/global_step_50/actor

  # Dry run (preview without saving)
  python scripts/remove_base_weights.py \\
      --source /path/to/checkpoint/global_step_50/actor \\
      --dry-run

Note: Metadata files (dtensor_metadata_rank_*.json) are always saved for reconstruction.
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing checkpoint files (e.g., .../global_step_50/actor)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory to save filtered checkpoints (default: in-place, modifies source)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without saving files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="model_world_size_*_rank_*.pt",
        help="Glob pattern to match checkpoint files (default: model_world_size_*_rank_*.pt)"
    )
    
    args = parser.parse_args()
    
    process_checkpoint_directory(
        source_dir=args.source,
        dest_dir=args.dest,
        pattern=args.pattern,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

