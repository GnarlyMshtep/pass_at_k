#!/usr/bin/env python3
"""
Compare original and reconstructed checkpoints to verify they are equal.

This script:
1. Compares keys between original and reconstructed checkpoints
2. Compares DTensor specs (device_mesh, placements, shapes)
3. Compares DTensor local tensor values (content comparison)
4. Reports any differences
"""

import torch
from pathlib import Path
import argparse
import sys

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


def compare_checkpoints(original_file, reconstructed_file, rank, verbose=True):
    """
    Compare original and reconstructed checkpoint files.
    
    Args:
        original_file: Path to original checkpoint file
        reconstructed_file: Path to reconstructed checkpoint file
        rank: Rank number (for reporting)
        verbose: If True, print detailed comparison
        
    Returns:
        Dictionary with comparison results
    """
    print("="*80)
    print(f"COMPARING CHECKPOINTS FOR RANK {rank}")
    print("="*80)
    
    # Load checkpoints
    print(f"Loading original checkpoint: {original_file}")
    original = torch.load(original_file, map_location='cpu', weights_only=False)
    
    if not isinstance(original, dict):
        if hasattr(original, '__dict__'):
            original = original.__dict__
        else:
            original = {'root': original}
    
    print(f"Loading reconstructed checkpoint: {reconstructed_file}")
    reconstructed = torch.load(reconstructed_file, map_location='cpu', weights_only=False)
    
    if not isinstance(reconstructed, dict):
        if hasattr(reconstructed, '__dict__'):
            reconstructed = reconstructed.__dict__
        else:
            reconstructed = {'root': reconstructed}
    
    # Compare keys
    original_keys = set(original.keys())
    reconstructed_keys = set(reconstructed.keys())
    
    results = {
        'rank': rank,
        'keys_match': original_keys == reconstructed_keys,
        'original_keys': len(original_keys),
        'reconstructed_keys': len(reconstructed_keys),
        'missing_keys': list(original_keys - reconstructed_keys),
        'extra_keys': list(reconstructed_keys - original_keys),
        'dtensor_specs_match': True,
        'dtensor_values_match': True,
        'spec_mismatches': [],
        'value_mismatches': [],
        'total_tensors': 0,
        'matched_tensors': 0
    }
    
    print(f"\nKey Comparison:")
    print(f"  Original keys: {len(original_keys)}")
    print(f"  Reconstructed keys: {len(reconstructed_keys)}")
    
    if original_keys == reconstructed_keys:
        print("  ✓ Keys match exactly")
    else:
        print("  ✗ Key mismatch!")
        if results['missing_keys']:
            print(f"    Missing in reconstructed: {len(results['missing_keys'])} keys")
            if verbose:
                for key in results['missing_keys'][:10]:
                    print(f"      - {key}")
        if results['extra_keys']:
            print(f"    Extra in reconstructed: {len(results['extra_keys'])} keys")
            if verbose:
                for key in results['extra_keys'][:10]:
                    print(f"      + {key}")
    
    # Compare common keys
    common_keys = original_keys & reconstructed_keys
    print(f"\nComparing {len(common_keys)} common tensors...")
    
    for key in sorted(common_keys):
        orig_val = original[key]
        recon_val = reconstructed[key]
        
        results['total_tensors'] += 1
        
        # Check if both are DTensors
        if isinstance(orig_val, DTensor) and isinstance(recon_val, DTensor):
            # Compare specs
            orig_spec = orig_val._spec
            recon_spec = recon_val._spec
            
            spec_match = True
            spec_details = {}
            
            # Compare device mesh
            if orig_spec.device_mesh.device_type != recon_spec.device_mesh.device_type:
                spec_match = False
                spec_details['device_type'] = (orig_spec.device_mesh.device_type, recon_spec.device_mesh.device_type)
            
            # Compare placements
            if len(orig_spec.placements) != len(recon_spec.placements):
                spec_match = False
                spec_details['placements_count'] = (len(orig_spec.placements), len(recon_spec.placements))
            else:
                for i, (op, rp) in enumerate(zip(orig_spec.placements, recon_spec.placements)):
                    if hasattr(op, 'dim') and hasattr(rp, 'dim'):
                        if op.dim != rp.dim:
                            spec_match = False
                            spec_details[f'placement_{i}_dim'] = (op.dim, rp.dim)
            
            # Compare global shapes
            orig_global_shape = orig_spec.shape if hasattr(orig_spec, 'shape') else orig_val.shape
            recon_global_shape = recon_spec.shape if hasattr(recon_spec, 'shape') else recon_val.shape
            if orig_global_shape != recon_global_shape:
                spec_match = False
                spec_details['global_shape'] = (list(orig_global_shape), list(recon_global_shape))
            
            if not spec_match:
                results['dtensor_specs_match'] = False
                results['spec_mismatches'].append({
                    'key': key,
                    'details': spec_details
                })
            
            # Compare local tensor values
            orig_local = orig_val._local_tensor
            recon_local = recon_val._local_tensor
            
            if orig_local.shape != recon_local.shape:
                results['dtensor_values_match'] = False
                results['value_mismatches'].append({
                    'key': key,
                    'issue': 'shape_mismatch',
                    'original_shape': list(orig_local.shape),
                    'reconstructed_shape': list(recon_local.shape)
                })
            elif not torch.equal(orig_local, recon_local):
                max_diff = (orig_local - recon_local).abs().max().item()
                mean_diff = (orig_local - recon_local).abs().mean().item()
                results['dtensor_values_match'] = False
                results['value_mismatches'].append({
                    'key': key,
                    'issue': 'value_mismatch',
                    'max_diff': max_diff,
                    'mean_diff': mean_diff
                })
            else:
                results['matched_tensors'] += 1
                
        elif isinstance(orig_val, torch.Tensor) and isinstance(recon_val, torch.Tensor):
            # Regular tensors
            if orig_val.shape != recon_val.shape:
                results['value_mismatches'].append({
                    'key': key,
                    'issue': 'shape_mismatch',
                    'original_shape': list(orig_val.shape),
                    'reconstructed_shape': list(recon_val.shape)
                })
            elif not torch.equal(orig_val, recon_val):
                max_diff = (orig_val - recon_val).abs().max().item()
                mean_diff = (orig_val - recon_val).abs().mean().item()
                results['value_mismatches'].append({
                    'key': key,
                    'issue': 'value_mismatch',
                    'max_diff': max_diff,
                    'mean_diff': mean_diff
                })
            else:
                results['matched_tensors'] += 1
        else:
            # Different types
            results['value_mismatches'].append({
                'key': key,
                'issue': 'type_mismatch',
                'original_type': type(orig_val).__name__,
                'reconstructed_type': type(recon_val).__name__
            })
    
    # Print summary
    print(f"\nComparison Summary:")
    print(f"  Total tensors compared: {results['total_tensors']}")
    print(f"  Matched tensors: {results['matched_tensors']}")
    print(f"  Spec mismatches: {len(results['spec_mismatches'])}")
    print(f"  Value mismatches: {len(results['value_mismatches'])}")
    
    if results['dtensor_specs_match'] and results['dtensor_values_match'] and results['keys_match']:
        print("\n  ✓ All checks passed! Checkpoints are identical.")
    else:
        print("\n  ✗ Some mismatches found:")
        if results['spec_mismatches']:
            print(f"    Spec mismatches: {len(results['spec_mismatches'])}")
            if verbose:
                for mismatch in results['spec_mismatches'][:5]:
                    print(f"      - {mismatch['key']}: {mismatch['details']}")
        if results['value_mismatches']:
            print(f"    Value mismatches: {len(results['value_mismatches'])}")
            if verbose:
                for mismatch in results['value_mismatches'][:5]:
                    if mismatch['issue'] == 'value_mismatch':
                        print(f"      - {mismatch['key']}: max_diff={mismatch['max_diff']:.2e}, mean_diff={mismatch['mean_diff']:.2e}")
                    else:
                        print(f"      - {mismatch['key']}: {mismatch['issue']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and reconstructed checkpoints"
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Directory containing original checkpoints (e.g., .../global_step_50/actor)"
    )
    parser.add_argument(
        "--reconstructed",
        type=str,
        required=True,
        help="Directory containing reconstructed checkpoints"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="World size (number of ranks)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison information"
    )
    
    args = parser.parse_args()
    
    original_dir = Path(args.original)
    reconstructed_dir = Path(args.reconstructed)
    
    all_results = []
    all_passed = True
    
    for rank in range(args.world_size):
        original_file = original_dir / f"model_world_size_{args.world_size}_rank_{rank}.pt"
        reconstructed_file = reconstructed_dir / f"model_world_size_{args.world_size}_rank_{rank}.pt"
        
        if not original_file.exists():
            print(f"Warning: Original checkpoint not found: {original_file}")
            continue
        
        if not reconstructed_file.exists():
            print(f"Warning: Reconstructed checkpoint not found: {reconstructed_file}")
            continue
        
        results = compare_checkpoints(original_file, reconstructed_file, rank, verbose=args.verbose)
        all_results.append(results)
        
        if not (results['keys_match'] and results['dtensor_specs_match'] and results['dtensor_values_match']):
            all_passed = False
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_tensors = sum(r['total_tensors'] for r in all_results)
    total_matched = sum(r['matched_tensors'] for r in all_results)
    total_spec_mismatches = sum(len(r['spec_mismatches']) for r in all_results)
    total_value_mismatches = sum(len(r['value_mismatches']) for r in all_results)
    
    print(f"Total ranks compared: {len(all_results)}")
    print(f"Total tensors compared: {total_tensors}")
    print(f"Matched tensors: {total_matched}")
    print(f"Spec mismatches: {total_spec_mismatches}")
    print(f"Value mismatches: {total_value_mismatches}")
    
    if all_passed:
        print("\n✓ ALL CHECKS PASSED! Original and reconstructed checkpoints are identical.")
        return 0
    else:
        print("\n✗ SOME MISMATCHES FOUND. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

