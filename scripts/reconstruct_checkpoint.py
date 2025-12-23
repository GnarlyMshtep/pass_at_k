#!/usr/bin/env python3
"""
Script 2: Reconstruct Original Checkpoint from Filtered Checkpoint + Base Model

This script:
1. Loads base model weights (regular tensors from HuggingFace)
2. Loads filtered checkpoint (contains LoRA weights as DTensors)
3. Loads DTensor metadata (from metadata file or extracts from filtered checkpoint)
4. Reconstructs base model weights as DTensors with identical specs
5. Combines base + LoRA to create reconstructed checkpoint
6. Saves reconstructed checkpoint (bit-identical structure to original)

IMPORTANT: This script MUST be run with torchrun to provide distributed context:
    torchrun --nproc_per_node=<world_size> scripts/reconstruct_checkpoint.py ...

Usage:
    torchrun --nproc_per_node=2 scripts/reconstruct_checkpoint.py \\
        --base-model /path/to/base/model \\
        --filtered-checkpoint /path/to/filtered/checkpoint/global_step_50/actor \\
        --output /path/to/reconstructed/checkpoint/global_step_50/actor \\
        [--metadata-dir /path/to/filtered/checkpoint/global_step_50/actor] \\
        [--validate] \\
        [--compare-with /path/to/original/checkpoint/global_step_50/actor]
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
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.device_mesh import DeviceMesh
    DTENSOR_AVAILABLE = True
except ImportError:
    try:
        from torch.distributed._tensor import DTensor, Shard
        from torch.distributed.device_mesh import DeviceMesh
        DTENSOR_AVAILABLE = True
    except ImportError:
        DTensor = None
        Shard = None
        DeviceMesh = None
        DTENSOR_AVAILABLE = False

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


def checkpoint_key_to_base_key(checkpoint_key):
    """
    Convert checkpoint key to base model key.
    
    Examples:
        'base_model.model.model.embed_tokens.weight' → 'model.embed_tokens.weight'
        'base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight' → 'model.layers.0.self_attn.q_proj.weight'
    """
    # Remove 'base_model.model.' prefix
    base_key = checkpoint_key.replace('base_model.model.', '')
    # Remove '.base_layer' suffix
    base_key = base_key.replace('.base_layer', '')
    return base_key


def load_base_model_weights(base_model_path):
    """
    Load base model weights from HuggingFace model.
    
    Args:
        base_model_path: Path to HuggingFace base model
        
    Returns:
        State dict with full tensors (not sharded)
    """
    from transformers import AutoModelForCausalLM
    
    print(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    state_dict = model.state_dict()
    print(f"Loaded {len(state_dict)} base model weights")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return state_dict


def load_metadata(metadata_dir, rank, filtered_checkpoint_file=None):
    """
    Load DTensor metadata from metadata file or extract from filtered checkpoint.
    
    Args:
        metadata_dir: Directory containing metadata files
        rank: Rank number
        filtered_checkpoint_file: Optional path to filtered checkpoint (for fallback)
        
    Returns:
        Metadata dictionary
    """
    metadata_file = Path(metadata_dir) / f"dtensor_metadata_rank_{rank}.json"
    
    if metadata_file.exists():
        print(f"Loading metadata from: {metadata_file}")
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata['tensors'])} tensors")
        return metadata
    else:
        # Fallback: extract from filtered checkpoint
        if filtered_checkpoint_file is None:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file} and no checkpoint file provided for extraction")
        
        print(f"Metadata file not found, extracting from checkpoint: {filtered_checkpoint_file}")
        # This would require importing the extraction logic
        # For now, raise an error and suggest using metadata files
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Please run remove_base_weights.py with --save-metadata to generate metadata files."
        )


def create_device_mesh(device_mesh_info):
    """
    Create DeviceMesh object from metadata.
    
    Args:
        device_mesh_info: Dictionary with device_mesh information
        
    Returns:
        DeviceMesh object
    """
    device_type = device_mesh_info['device_type']
    mesh_list = device_mesh_info['mesh']
    mesh_dim_names = tuple(device_mesh_info.get('mesh_dim_names', ['fsdp']))
    
    # Create mesh tensor
    mesh_tensor = torch.tensor(mesh_list, dtype=torch.int64)
    
    device_mesh = DeviceMesh(
        device_type,
        mesh_tensor,
        mesh_dim_names=mesh_dim_names
    )
    
    return device_mesh


def create_placements(placements_info):
    """
    Create Placement objects from metadata.
    
    Args:
        placements_info: List of placement dictionaries
        
    Returns:
        Tuple of Placement objects
    """
    placements = []
    for p_info in placements_info:
        if p_info['type'] == 'Shard':
            placements.append(Shard(dim=p_info['dim']))
        else:
            # Handle other placement types if needed
            raise ValueError(f"Unsupported placement type: {p_info['type']}")
    
    return tuple(placements)


def shard_base_weight(full_tensor, slices, global_shape):
    """
    Extract local shard from full tensor according to slice information.
    
    Args:
        full_tensor: Full tensor from base model
        slices: Slice information [[start_dim0, end_dim0], [start_dim1, end_dim1], ...]
        global_shape: Global tensor shape
        
    Returns:
        Local shard tensor
    """
    if slices is None:
        # Not sharded, return full tensor
        return full_tensor
    
    # Build indexing tuple
    index_tuple = tuple(slice(s[0], s[1]) for s in slices)
    local_shard = full_tensor[index_tuple]
    
    return local_shard


def create_dtensor_from_shard(local_shard, device_mesh, placements, global_shape, stride):
    """
    Create DTensor from local shard with identical spec to original.
    
    Args:
        local_shard: Local shard tensor
        device_mesh: DeviceMesh object
        placements: Placement tuple
        global_shape: Global tensor shape
        stride: Local tensor stride (CRITICAL for reconstruction!)
        
    Returns:
        DTensor object
    """
    # CRITICAL: Must provide stride parameter
    dtensor = DTensor.from_local(
        local_shard,
        device_mesh,
        placements,
        shape=global_shape,
        stride=stride
    )
    
    return dtensor


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


def reconstruct_checkpoint(
    base_model_path,
    filtered_checkpoint_dir,
    output_dir,
    rank,
    world_size,
    metadata_dir=None,
    validate=False,
    compare_with=None
):
    """
    Reconstruct checkpoint from base model + filtered checkpoint + metadata.
    
    Args:
        base_model_path: Path to HuggingFace base model
        filtered_checkpoint_dir: Directory containing filtered checkpoints
        output_dir: Output directory for reconstructed checkpoints (None for in-place)
        rank: Current rank (from distributed context)
        world_size: Total number of ranks (from distributed context)
        metadata_dir: Directory containing metadata files (default: same as filtered_checkpoint_dir)
        validate: If True, validate reconstructed checkpoint
        compare_with: Optional path to original checkpoint for comparison
    """
    print("="*80)
    print(f"RECONSTRUCTING CHECKPOINT FOR RANK {rank}")
    print("="*80)
    
    # Use filtered_checkpoint_dir as metadata_dir if not specified
    if metadata_dir is None:
        metadata_dir = filtered_checkpoint_dir
    
    # If no output_dir provided, use filtered_checkpoint_dir (in-place)
    if output_dir is None:
        output_dir = filtered_checkpoint_dir
        in_place = True
    else:
        in_place = False
    
    # Load filtered checkpoint to check for LoRA weights
    filtered_file = Path(filtered_checkpoint_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
    if not filtered_file.exists():
        if rank == 0:
            print(f"Warning: Filtered checkpoint file not found: {filtered_file}")
        return None
    
    print(f"\nLoading filtered checkpoint from: {filtered_file}")
    filtered_checkpoint = torch.load(filtered_file, map_location='cpu', weights_only=False)
    
    if not isinstance(filtered_checkpoint, dict):
        if hasattr(filtered_checkpoint, '__dict__'):
            filtered_checkpoint = filtered_checkpoint.__dict__
        else:
            filtered_checkpoint = {'root': filtered_checkpoint}
    
    # Check if checkpoint has LoRA weights
    if not has_lora_weights(filtered_checkpoint):
        if rank == 0:
            print("No LoRA weights found in checkpoint. Skipping (nothing to reconstruct).")
        return None
    
    # Load base model weights
    base_weights = load_base_model_weights(base_model_path)
    
    print(f"Loaded {len(filtered_checkpoint)} weights from filtered checkpoint")
    
    # Load metadata
    metadata = load_metadata(metadata_dir, rank, filtered_file)
    
    # Create device mesh and placements
    device_mesh = create_device_mesh(metadata['device_mesh'])
    placements = create_placements(metadata['placements'])
    
    print(f"\nDevice mesh: {metadata['device_mesh']}")
    print(f"Placements: {metadata['placements']}")
    
    # Reconstruct checkpoint
    print("\nReconstructing checkpoint...")
    reconstructed = OrderedDict()
    
    # Process all keys from metadata
    for key, tensor_info in metadata['tensors'].items():
        is_base_model_weight = tensor_info.get('is_base_model_weight', False)
        
        if is_base_model_weight:
            # This is a base model weight that was removed - reconstruct it
            base_key = checkpoint_key_to_base_key(key)
            
            if base_key not in base_weights:
                print(f"Warning: Base key not found: {base_key} (checkpoint key: {key})")
                continue
            
            # Get full base weight
            full_base_weight = base_weights[base_key]
            
            # Extract shard according to metadata
            slices = tensor_info.get('slices')
            local_shard = shard_base_weight(full_base_weight, slices, tensor_info['global_shape'])
            
            # Create DTensor with identical spec
            global_shape = torch.Size(tensor_info['global_shape'])
            stride = tuple(tensor_info['local_stride'])
            
            # Ensure local_shard has correct stride
            if local_shard.stride() != stride:
                # Create a new tensor with the correct stride
                local_shard = local_shard.clone().as_strided(
                    local_shard.shape,
                    stride,
                    storage_offset=0
                )
            
            # Ensure local_shard is on CPU (matching original checkpoint saving behavior)
            local_shard = local_shard.cpu()
            
            dt = create_dtensor_from_shard(
                local_shard,
                device_mesh,
                placements,
                global_shape,
                stride
            )
            
            reconstructed[key] = dt
        else:
            # Not a base model weight - use from filtered checkpoint unchanged
            # This includes: LoRA adapters, layer norms, embeddings, etc.
            if key in filtered_checkpoint:
                # Keep DTensor but ensure local tensor is on CPU
                value = filtered_checkpoint[key]
                if isinstance(value, DTensor):
                    # Move local tensor to CPU if not already
                    if value._local_tensor.device.type != 'cpu':
                        cpu_local = value._local_tensor.cpu()
                        # Recreate DTensor with CPU local tensor
                        value = DTensor.from_local(
                            cpu_local,
                            value._spec.device_mesh,
                            value._spec.placements,
                            shape=value.shape,
                            stride=cpu_local.stride()
                        )
                elif isinstance(value, torch.Tensor):
                    value = value.cpu()
                reconstructed[key] = value
            else:
                print(f"Warning: Key not found in filtered checkpoint: {key}")
    
    # Verify all keys are present
    print(f"\nReconstructed {len(reconstructed)} weights")
    print(f"Expected {len(metadata['tensors'])} weights from metadata")
    
    if len(reconstructed) != len(metadata['tensors']):
        print(f"Warning: Key count mismatch! Some keys may be missing.")
        missing_keys = set(metadata['tensors'].keys()) - set(reconstructed.keys())
        if missing_keys:
            print(f"Missing keys: {list(missing_keys)[:10]}...")
    
    # Save reconstructed checkpoint
    # Note: All tensors should already be on CPU from the reconstruction process above
    output_file = Path(output_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving reconstructed checkpoint to: {output_file}")
    torch.save(reconstructed, output_file)
    
    print(f"Saved {len(reconstructed)} tensors")
    
    # Validation
    if validate or compare_with is not None:
        print("\n" + "="*80)
        print("VALIDATION")
        print("="*80)
        
        if compare_with is not None:
            original_file = Path(compare_with) / f"model_world_size_{world_size}_rank_{rank}.pt"
            if original_file.exists():
                print(f"Comparing with original checkpoint: {original_file}")
                original_checkpoint = torch.load(original_file, map_location='cpu', weights_only=False)
                
                if not isinstance(original_checkpoint, dict):
                    if hasattr(original_checkpoint, '__dict__'):
                        original_checkpoint = original_checkpoint.__dict__
                    else:
                        original_checkpoint = {'root': original_checkpoint}
                
                # Compare keys
                original_keys = set(original_checkpoint.keys())
                reconstructed_keys = set(reconstructed.keys())
                
                if original_keys == reconstructed_keys:
                    print("✓ Keys match exactly")
                else:
                    print(f"✗ Key mismatch!")
                    print(f"  Only in original: {original_keys - reconstructed_keys}")
                    print(f"  Only in reconstructed: {reconstructed_keys - original_keys}")
                
                # Compare DTensor values (for a few sample keys)
                sample_keys = list(original_keys)[:5]
                for key in sample_keys:
                    if key in reconstructed:
                        orig_val = original_checkpoint[key]
                        recon_val = reconstructed[key]
                        
                        if isinstance(orig_val, DTensor) and isinstance(recon_val, DTensor):
                            # Compare local tensors
                            orig_local = orig_val._local_tensor
                            recon_local = recon_val._local_tensor
                            
                            if torch.equal(orig_local, recon_local):
                                print(f"✓ {key}: Local tensors match")
                            else:
                                max_diff = (orig_local - recon_local).abs().max().item()
                                print(f"✗ {key}: Local tensors differ (max diff: {max_diff})")
                            
                            # Compare specs
                            if orig_val._spec == recon_val._spec:
                                print(f"✓ {key}: DTensor specs match")
                            else:
                                print(f"✗ {key}: DTensor specs differ")
            else:
                print(f"Warning: Original checkpoint file not found: {original_file}")
    
    return reconstructed


def copy_other_files(source_dir, dest_dir):
    """Copy other necessary files from source to destination."""
    source = Path(source_dir)
    dest = Path(dest_dir)
    
    files_to_copy = [
        'fsdp_config.json',
    ]
    
    # Copy optimizer and extra state files
    for rank in range(10):  # Check up to 10 ranks
        for prefix in ['optim', 'extra_state']:
            pattern_file = source.glob(f"{prefix}_world_size_*_rank_{rank}.pt")
            for f in pattern_file:
                files_to_copy.append(f.name)
    
    dirs_to_copy = [
        'huggingface',
        'lora_adapter',
    ]
    
    print("\n" + "="*80)
    print("COPYING OTHER FILES")
    print("="*80)
    
    # Copy files
    for filename in files_to_copy:
        src_file = source / filename
        if src_file.exists():
            dest_file = dest / filename
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {filename}")
    
    # Copy directories
    for dirname in dirs_to_copy:
        src_dir_path = source / dirname
        if src_dir_path.exists():
            dest_dir_path = dest / dirname
            if dest_dir_path.exists():
                shutil.rmtree(dest_dir_path)
            shutil.copytree(src_dir_path, dest_dir_path)
            print(f"Copied directory: {dirname}/")
    
    # Copy data.pt if it exists in parent directory
    parent_source = source.parent
    data_file = parent_source / 'data.pt'
    if data_file.exists():
        parent_dest = dest.parent
        parent_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_file, parent_dest / 'data.pt')
        print(f"Copied: data.pt")


def init_distributed():
    """Initialize distributed context if not already initialized."""
    if not DIST_AVAILABLE:
        raise RuntimeError("torch.distributed is not available. This script requires distributed context.")
    
    if not dist.is_initialized():
        # Try to get from environment variables (set by torchrun)
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        return rank, world_size
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct FSDP2 checkpoints from base model + filtered checkpoint + metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This script MUST be run with torchrun to provide distributed context:

    torchrun --nproc_per_node=2 scripts/reconstruct_checkpoint.py \\
        --base-model /path/to/base/model \\
        --filtered-checkpoint /path/to/filtered/checkpoint/global_step_50/actor \\
        --output /path/to/reconstructed/checkpoint/global_step_50/actor

Examples:
  # Basic reconstruction in-place (modifies filtered checkpoint)
  torchrun --nproc_per_node=2 scripts/reconstruct_checkpoint.py \\
      --base-model /path/to/Qwen2.5-3B-Instruct \\
      --filtered-checkpoint /path/to/filtered/global_step_50/actor

  # Basic reconstruction to different location
  torchrun --nproc_per_node=2 scripts/reconstruct_checkpoint.py \\
      --base-model /path/to/Qwen2.5-3B-Instruct \\
      --filtered-checkpoint /path/to/filtered/global_step_50/actor \\
      --output /path/to/reconstructed/global_step_50/actor

  # With validation
  torchrun --nproc_per_node=2 scripts/reconstruct_checkpoint.py \\
      --base-model /path/to/Qwen2.5-3B-Instruct \\
      --filtered-checkpoint /path/to/filtered/global_step_50/actor \\
      --output /path/to/reconstructed/global_step_50/actor \\
      --validate \\
      --compare-with /path/to/original/global_step_50/actor
        """
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to HuggingFace base model"
    )
    parser.add_argument(
        "--filtered-checkpoint",
        type=str,
        required=True,
        help="Directory containing filtered checkpoints (e.g., .../global_step_50/actor)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reconstructed checkpoints (default: in-place, modifies filtered checkpoint)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Directory containing metadata files (default: same as filtered-checkpoint)"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source directory for copying other files (default: same as filtered-checkpoint)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate reconstructed checkpoint"
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Path to original checkpoint directory for comparison"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank number (auto-detected from distributed context if available)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="World size (auto-detected from distributed context if available)"
    )
    
    args = parser.parse_args()
    
    # Initialize distributed context
    try:
        rank, world_size = init_distributed()
        print(f"Initialized distributed context: rank={rank}, world_size={world_size}")
    except Exception as e:
        print(f"Warning: Could not initialize distributed context: {e}")
        print("Attempting to use environment variables or arguments...")
        rank = args.rank if args.rank is not None else int(os.environ.get('RANK', 0))
        world_size = args.world_size if args.world_size is not None else int(os.environ.get('WORLD_SIZE', 2))
        print(f"Using rank={rank}, world_size={world_size}")
    
    # Reconstruct checkpoint for this rank
    reconstruct_checkpoint(
        base_model_path=args.base_model,
        filtered_checkpoint_dir=args.filtered_checkpoint,
        output_dir=args.output,
        rank=rank,
        world_size=world_size,
        metadata_dir=args.metadata_dir,
        validate=args.validate,
        compare_with=args.compare_with
    )
    
    # Copy other files (only from rank 0 to avoid duplication)
    if rank == 0:
        source_dir = args.source_dir if args.source_dir else args.filtered_checkpoint
        output_dir = args.output if args.output else args.filtered_checkpoint
        copy_other_files(source_dir, output_dir)
        
        print("\n" + "="*80)
        print("RECONSTRUCTION COMPLETE!")
        print("="*80)
        if args.output:
            print(f"Reconstructed checkpoints saved to: {args.output}")
        else:
            print(f"Reconstructed checkpoints saved in-place to: {args.filtered_checkpoint}")
    
    # Clean up distributed context
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

