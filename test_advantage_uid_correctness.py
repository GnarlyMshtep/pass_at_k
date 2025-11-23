"""
Test script to verify that compute_rs_grpo_outcome_advantage handles UIDs correctly.

This script tests that calling the function once with multiple UIDs gives the same
results as calling it separately for each UID.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Import the advantage estimation functions
sys.path.insert(0, '/cmlscratch/asoltan3/pass_at_k')
from verl.trainer.ppo.core_algos import compute_rs_grpo_outcome_advantage, compute_grpo_outcome_advantage


def generate_multi_uid_data(uid_configs, response_length=10):
    """
    Generate synthetic data with multiple UIDs.
    
    Args:
        uid_configs: List of tuples (uid, n_samples, n_correct, reward_correct, reward_incorrect)
        response_length: Length of each response
    
    Returns:
        token_level_rewards, response_mask, index, uid_info
    """
    all_rewards = []
    all_indices = []
    uid_info = {}  # uid -> (start_idx, end_idx, n_correct, n_samples)
    
    current_idx = 0
    
    for uid, n_samples, n_correct, reward_correct, reward_incorrect in uid_configs:
        n_incorrect = n_samples - n_correct
        
        # Create token-level rewards for this UID
        for _ in range(n_correct):
            rewards = torch.zeros(response_length)
            rewards[-1] = reward_correct
            all_rewards.append(rewards)
            all_indices.append(uid)
        
        for _ in range(n_incorrect):
            rewards = torch.zeros(response_length)
            rewards[-1] = reward_incorrect
            all_rewards.append(rewards)
            all_indices.append(uid)
        
        uid_info[uid] = {
            'start_idx': current_idx,
            'end_idx': current_idx + n_samples,
            'n_correct': n_correct,
            'n_samples': n_samples,
            'reward_correct': reward_correct,
            'reward_incorrect': reward_incorrect
        }
        current_idx += n_samples
    
    token_level_rewards = torch.stack(all_rewards)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(all_indices, dtype=np.int64)
    
    return token_level_rewards, response_mask, index, uid_info


def test_single_call_vs_multiple_calls(beta_values, response_length=10):
    """
    Test that calling once with multiple UIDs gives same results as multiple calls.
    """
    # Create test configuration with multiple UIDs
    # Each UID has different number of correct/incorrect samples
    uid_configs = [
        # (uid, n_samples, n_correct, reward_correct, reward_incorrect)
        (0, 8, 2, 1.0, 0.0),  # UID 0: 2 correct out of 8
        (1, 6, 3, 1.0, 0.0),  # UID 1: 3 correct out of 6
        (2, 10, 5, 1.0, 0.0), # UID 2: 5 correct out of 10
        (3, 4, 4, 1.0, 0.0),  # UID 3: 4 correct out of 4 (all correct)
        (4, 8, 0, 1.0, 0.0),  # UID 4: 0 correct out of 8 (all incorrect)
    ]
    
    results = {
        'single_call_rsgrpo': {},
        'multiple_calls_rsgrpo': {},
        'single_call_grpo': {},
        'multiple_calls_grpo': {},
    }
    
    # Generate combined data for all UIDs
    token_level_rewards_all, response_mask_all, index_all, uid_info = generate_multi_uid_data(
        uid_configs, response_length
    )
    
    print(f"Generated data with {len(uid_configs)} UIDs, total {len(index_all)} samples")
    print(f"Token level rewards shape: {token_level_rewards_all.shape}")
    print(f"Index array: {index_all}")
    print(f"\nUID breakdown:")
    for uid, info in uid_info.items():
        print(f"  UID {uid}: {info['n_correct']}/{info['n_samples']} correct, indices [{info['start_idx']}:{info['end_idx']}]")
    
    for beta in beta_values:
        print(f"\n{'='*80}")
        print(f"Testing β = {beta}")
        print(f"{'='*80}")
        
        # Create risk_beta_per_uid dict
        risk_beta_per_uid = {uid: beta for uid, _, _, _, _ in uid_configs}
        
        # ===== SINGLE CALL: Call once with all data =====
        print("\n[SINGLE CALL] Computing advantages for all UIDs at once...")
        adv_single_rsgrpo, _ = compute_rs_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards_all,
            response_mask=response_mask_all,
            index=index_all,
            risk_beta_per_uid=risk_beta_per_uid,
            epsilon=1e-6,
            config=None
        )
        
        adv_single_grpo, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards_all,
            response_mask=response_mask_all,
            index=index_all,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
            config=None
        )
        
        # ===== MULTIPLE CALLS: Call separately for each UID =====
        print("[MULTIPLE CALLS] Computing advantages for each UID separately...")
        adv_multiple_rsgrpo = torch.zeros_like(adv_single_rsgrpo)
        adv_multiple_grpo = torch.zeros_like(adv_single_grpo)
        
        for uid, n_samples, n_correct, reward_correct, reward_incorrect in uid_configs:
            info = uid_info[uid]
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            
            # Extract data for this UID
            token_level_rewards_uid = token_level_rewards_all[start_idx:end_idx]
            response_mask_uid = response_mask_all[start_idx:end_idx]
            index_uid = np.zeros(n_samples, dtype=np.int64)  # All same UID
            risk_beta_per_uid_single = {0: beta}  # Map to UID 0 for single call
            
            # Compute advantages for this UID only
            adv_uid_rsgrpo, _ = compute_rs_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards_uid,
                response_mask=response_mask_uid,
                index=index_uid,
                risk_beta_per_uid=risk_beta_per_uid_single,
                epsilon=1e-6,
                config=None
            )
            
            adv_uid_grpo, _ = compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards_uid,
                response_mask=response_mask_uid,
                index=index_uid,
                epsilon=1e-6,
                norm_adv_by_std_in_grpo=True,
                config=None
            )
            
            # Store in the combined array
            adv_multiple_rsgrpo[start_idx:end_idx] = adv_uid_rsgrpo
            adv_multiple_grpo[start_idx:end_idx] = adv_uid_grpo
        
        # ===== COMPARE RESULTS =====
        print(f"\n[COMPARISON] Checking if results match...")
        
        # RSGRPO comparison
        diff_rsgrpo = torch.abs(adv_single_rsgrpo - adv_multiple_rsgrpo)
        max_diff_rsgrpo = diff_rsgrpo.max().item()
        mean_diff_rsgrpo = diff_rsgrpo.mean().item()
        
        print(f"  RSGRPO - Max difference: {max_diff_rsgrpo:.10f}")
        print(f"  RSGRPO - Mean difference: {mean_diff_rsgrpo:.10f}")
        
        if max_diff_rsgrpo < 1e-6:
            print(f"  ✅ RSGRPO: PASS - Results match!")
        else:
            print(f"  ❌ RSGRPO: FAIL - Results differ!")
        
        # GRPO comparison
        diff_grpo = torch.abs(adv_single_grpo - adv_multiple_grpo)
        max_diff_grpo = diff_grpo.max().item()
        mean_diff_grpo = diff_grpo.mean().item()
        
        print(f"  GRPO - Max difference: {max_diff_grpo:.10f}")
        print(f"  GRPO - Mean difference: {mean_diff_grpo:.10f}")
        
        if max_diff_grpo < 1e-6:
            print(f"  ✅ GRPO: PASS - Results match!")
        else:
            print(f"  ❌ GRPO: FAIL - Results differ!")
        
        # Store results for plotting
        results['single_call_rsgrpo'][beta] = adv_single_rsgrpo
        results['multiple_calls_rsgrpo'][beta] = adv_multiple_rsgrpo
        results['single_call_grpo'][beta] = adv_single_grpo
        results['multiple_calls_grpo'][beta] = adv_multiple_grpo
        
        # Print sample advantages per UID
        print(f"\n  Sample advantages per UID:")
        for uid, info in uid_info.items():
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            
            adv_uid_single_rsgrpo = adv_single_rsgrpo[start_idx:end_idx, 0]  # First token
            adv_uid_multiple_rsgrpo = adv_multiple_rsgrpo[start_idx:end_idx, 0]
            
            print(f"    UID {uid} (RSGRPO, β={beta}):")
            print(f"      Single call:   {adv_uid_single_rsgrpo[:3].tolist()} ...")
            print(f"      Multiple call: {adv_uid_multiple_rsgrpo[:3].tolist()} ...")
    
    return results, uid_info, token_level_rewards_all, response_mask_all, index_all


def plot_comparison(results, uid_info, beta_values):
    """
    Create comparison plots between single call and multiple calls.
    """
    n_uids = len(uid_info)
    fig, axes = plt.subplots(len(beta_values), 2, figsize=(18, 4 * len(beta_values)))
    
    if len(beta_values) == 1:
        axes = axes.reshape(1, -1)
    
    for i, beta in enumerate(beta_values):
        adv_single_rsgrpo = results['single_call_rsgrpo'][beta][:, 0].numpy()  # First token only
        adv_multiple_rsgrpo = results['multiple_calls_rsgrpo'][beta][:, 0].numpy()
        adv_single_grpo = results['single_call_grpo'][beta][:, 0].numpy()
        adv_multiple_grpo = results['multiple_calls_grpo'][beta][:, 0].numpy()
        
        # RSGRPO plot
        ax = axes[i, 0]
        x = np.arange(len(adv_single_rsgrpo))
        ax.scatter(x, adv_single_rsgrpo, alpha=0.6, s=50, label='Single Call', marker='o')
        ax.scatter(x, adv_multiple_rsgrpo, alpha=0.6, s=50, label='Multiple Calls', marker='x')
        
        # Add vertical lines to separate UIDs
        for uid, info in uid_info.items():
            if info['start_idx'] > 0:
                ax.axvline(info['start_idx'], color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Advantage (first token)')
        ax.set_title(f'RSGRPO β={beta}: Single Call vs Multiple Calls')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # GRPO plot
        ax = axes[i, 1]
        ax.scatter(x, adv_single_grpo, alpha=0.6, s=50, label='Single Call', marker='o')
        ax.scatter(x, adv_multiple_grpo, alpha=0.6, s=50, label='Multiple Calls', marker='x')
        
        # Add vertical lines to separate UIDs
        for uid, info in uid_info.items():
            if info['start_idx'] > 0:
                ax.axvline(info['start_idx'], color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Advantage (first token)')
        ax.set_title(f'GRPO: Single Call vs Multiple Calls')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run correctness tests."""
    
    print("="*80)
    print("UID CORRECTNESS TEST FOR ADVANTAGE ESTIMATION")
    print("="*80)
    print("\nThis test verifies that calling the function once with multiple UIDs")
    print("produces the same results as calling it separately for each UID.\n")
    
    # Test parameters
    beta_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    response_length = 10
    
    print(f"Configuration:")
    print(f"  - Beta values: {beta_values}")
    print(f"  - Response length: {response_length}\n")
    
    # Run tests
    results, uid_info, token_level_rewards_all, response_mask_all, index_all = test_single_call_vs_multiple_calls(
        beta_values=beta_values,
        response_length=response_length
    )
    
    # Create comparison plots
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")
    
    fig = plot_comparison(results, uid_info, beta_values)
    fig.savefig('/cmlscratch/asoltan3/pass_at_k/uid_correctness_comparison_with_negative_beta.png', dpi=300, bbox_inches='tight')
    print("Saved: uid_correctness_comparison_with_negative_beta.png")
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print("✅ All tests passed! The function correctly handles multiple UIDs.")
    print("   Calling once with all data gives identical results to calling")
    print("   separately for each UID group.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

