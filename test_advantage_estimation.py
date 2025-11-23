"""
Test script for analyzing advantage estimation in GRPO and RSGRPO.

This script generates synthetic data and analyzes how advantages vary based on:
- Number of correct vs incorrect samples per group (x correct, n-x incorrect)
- Different beta values (for RSGRPO)
- Whether advantages are positive or negative
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Import the advantage estimation functions
sys.path.insert(0, '/cmlscratch/asoltan3/pass_at_k')
from verl.trainer.ppo.core_algos import compute_rs_grpo_outcome_advantage, compute_grpo_outcome_advantage


def generate_synthetic_data(n_samples_per_group, n_correct, response_length=10, reward_correct=1.0, reward_incorrect=0.0):
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples_per_group: Total number of samples per group (n)
        n_correct: Number of correct samples (x)
        response_length: Length of each response
        reward_correct: Reward for correct samples
        reward_incorrect: Reward for incorrect samples
    
    Returns:
        token_level_rewards, response_mask, index
    """
    n_incorrect = n_samples_per_group - n_correct
    
    # Create token-level rewards
    token_level_rewards = []
    for _ in range(n_correct):
        # Correct samples: positive reward at the end
        rewards = torch.zeros(response_length)
        rewards[-1] = reward_correct
        token_level_rewards.append(rewards)
    
    for _ in range(n_incorrect):
        # Incorrect samples: zero or negative reward
        rewards = torch.zeros(response_length)
        rewards[-1] = reward_incorrect
        token_level_rewards.append(rewards)
    
    token_level_rewards = torch.stack(token_level_rewards)
    
    # Create response mask (all ones for simplicity)
    response_mask = torch.ones_like(token_level_rewards)
    
    # Create index (all samples belong to same group)
    index = np.zeros(n_samples_per_group, dtype=np.int64)
    
    return token_level_rewards, response_mask, index


def analyze_advantages(advantages, response_mask, label=""):
    """
    Analyze advantages and return statistics for positive and negative advantages.
    
    Returns:
        dict with keys: 'pos_mean', 'pos_var', 'neg_mean', 'neg_var', 'pos_count', 'neg_count'
    """
    # Flatten advantages and mask
    adv_flat = advantages[response_mask > 0].flatten()
    
    # Separate positive and negative
    pos_adv = adv_flat[adv_flat > 0]
    neg_adv = adv_flat[adv_flat < 0]
    
    stats = {
        'pos_mean': pos_adv.mean().item() if len(pos_adv) > 0 else 0.0,
        'pos_var': pos_adv.var().item() if len(pos_adv) > 0 else 0.0,
        'pos_std': pos_adv.std().item() if len(pos_adv) > 0 else 0.0,
        'pos_count': len(pos_adv),
        'neg_mean': neg_adv.mean().item() if len(neg_adv) > 0 else 0.0,
        'neg_var': neg_adv.var().item() if len(neg_adv) > 0 else 0.0,
        'neg_std': neg_adv.std().item() if len(neg_adv) > 0 else 0.0,
        'neg_count': len(neg_adv),
    }
    
    return stats


def run_experiment(n_total, beta_values, response_length=10):
    """
    Run experiment for different numbers of correct samples and beta values.
    
    Args:
        n_total: Total number of samples per group
        beta_values: List of beta values to test
        response_length: Length of responses
    
    Returns:
        results: dict mapping (beta, n_correct) -> stats
    """
    results_rsgrpo = {}
    results_grpo = {}
    
    # Test for different numbers of correct samples
    for n_correct in range(0, n_total + 1):
        token_level_rewards, response_mask, index = generate_synthetic_data(
            n_samples_per_group=n_total,
            n_correct=n_correct,
            response_length=response_length
        )
        
        # Test GRPO (beta-independent)
        adv_grpo, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
            config=None
        )
        stats_grpo = analyze_advantages(adv_grpo, response_mask)
        results_grpo[(0.0, n_correct)] = stats_grpo
        
        # Test RSGRPO for different beta values
        for beta in beta_values:
            # Create risk_beta_per_uid dict (all samples have same uid=0)
            risk_beta_per_uid = {0: beta}
            
            adv_rsgrpo, _ = compute_rs_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                risk_beta_per_uid=risk_beta_per_uid,
                epsilon=1e-6,
                config=None
            )
            
            stats_rsgrpo = analyze_advantages(adv_rsgrpo, response_mask)
            results_rsgrpo[(beta, n_correct)] = stats_rsgrpo
    
    return results_grpo, results_rsgrpo


def plot_results(results, n_total, beta_values, method_name="RSGRPO"):
    """
    Create plots for positive and negative advantages.
    
    Args:
        results: dict mapping (beta, n_correct) -> stats
        n_total: Total number of samples
        beta_values: List of beta values
        method_name: Name of the method for plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Prepare data for plotting
    x_values = list(range(0, n_total + 1))  # Number of correct samples
    
    # Plot 1: Positive advantages - Mean
    ax = axes[0, 0]
    for beta in beta_values:
        means = [results.get((beta, x), {}).get('pos_mean', np.nan) for x in x_values]
        ax.plot(x_values, means, marker='o', label=f'β={beta:.2f}')
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Mean of Positive Advantages')
    ax.set_title(f'{method_name}: Mean of Tokens with Positive Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Positive advantages - Variance
    ax = axes[0, 1]
    for beta in beta_values:
        variances = [results.get((beta, x), {}).get('pos_var', np.nan) for x in x_values]
        ax.plot(x_values, variances, marker='o', label=f'β={beta:.2f}')
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Variance of Positive Advantages')
    ax.set_title(f'{method_name}: Variance of Tokens with Positive Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Weighted combination - pos_mean*x + neg_mean*(n-x)
    ax = axes[0, 2]
    for beta in beta_values:
        weighted_values = []
        for x in x_values:
            stats = results.get((beta, x), {})
            pos_mean = stats.get('pos_mean', 0)
            neg_mean = stats.get('neg_mean', 0)
            weighted = pos_mean * x + neg_mean * (n_total - x)
            weighted_values.append(weighted)
        ax.plot(x_values, weighted_values, marker='o', label=f'β={beta:.2f}')
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('pos_mean*x + neg_mean*(n-x)')
    ax.set_title(f'{method_name}: Weighted Advantage Combination')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Negative advantages - Mean
    ax = axes[1, 0]
    for beta in beta_values:
        means = [results.get((beta, x), {}).get('neg_mean', np.nan) for x in x_values]
        ax.plot(x_values, means, marker='o', label=f'β={beta:.2f}')
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Mean of Negative Advantages')
    ax.set_title(f'{method_name}: Mean of Tokens with Negative Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Negative advantages - Variance
    ax = axes[1, 1]
    for beta in beta_values:
        variances = [results.get((beta, x), {}).get('neg_var', np.nan) for x in x_values]
        ax.plot(x_values, variances, marker='o', label=f'β={beta:.2f}')
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Variance of Negative Advantages')
    ax.set_title(f'{method_name}: Variance of Tokens with Negative Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Leave empty or add summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Add text summary
    summary_text = f"Summary Statistics\n"
    summary_text += f"Total samples (n): {n_total}\n"
    summary_text += f"Response per sample: All tokens\n"
    summary_text += f"Beta values: {beta_values}\n\n"
    summary_text += "Note: The weighted combination\n"
    summary_text += "represents the expected advantage\n"
    summary_text += "weighted by sample counts."
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run experiments and generate plots."""
    
    print("Starting advantage estimation analysis...")
    
    # Experiment parameters
    n_total = 16  # Total samples per group
    beta_values = [-4, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]  # Different beta values to test
    response_length = 10
    
    print(f"Configuration:")
    print(f"  - Total samples per group (n): {n_total}")
    print(f"  - Beta values: {beta_values}")
    print(f"  - Response length: {response_length}")
    print()
    
    # Run experiments
    print("Running experiments...")
    results_grpo, results_rsgrpo = run_experiment(
        n_total=n_total,
        beta_values=beta_values,
        response_length=response_length
    )
    
    # Print some sample results
    print("\nSample Results (RSGRPO, β=1.0, x=4 correct):")
    sample_stats = results_rsgrpo.get((1.0, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Results (GRPO, x=4 correct):")
    sample_stats = results_grpo.get((0.0, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    # Create plots for RSGRPO
    print("\nGenerating plots for RSGRPO...")
    fig_rsgrpo = plot_results(results_rsgrpo, n_total, beta_values, method_name="RSGRPO")
    fig_rsgrpo.savefig('/cmlscratch/asoltan3/pass_at_k/rsgrpo_advantage_analysis_with_negative_beta.png', dpi=300, bbox_inches='tight')
    print("Saved: rsgrpo_advantage_analysis_with_negative_beta.png")
    
    # Create plots for GRPO (just one line since beta-independent)
    print("\nGenerating plots for GRPO...")
    fig_grpo = plot_results(results_grpo, n_total, [0.0], method_name="GRPO")
    fig_grpo.savefig('/cmlscratch/asoltan3/pass_at_k/grpo_advantage_analysis_with_negative_beta.png', dpi=300, bbox_inches='tight')
    print("Saved: grpo_advantage_analysis_with_negative_beta.png")
    
    # Create detailed comparison table
    print("\n" + "="*100)
    print("DETAILED RESULTS TABLE")
    print("="*100)
    print(f"{'x':<4} {'β':<8} {'Method':<10} {'Pos Mean':<12} {'Pos Var':<12} {'Neg Mean':<12} {'Neg Var':<12} {'Weighted':<12}")
    print("-"*100)
    
    for x in range(0, n_total + 1):
        # GRPO
        stats = results_grpo.get((0.0, x), {})
        weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
        print(f"{x:<4} {'N/A':<8} {'GRPO':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        
        # RSGRPO for selected betas (including negative)
        for beta in [-1.0, -0.5, 0.5, 1.0, 2.0]:
            stats = results_rsgrpo.get((beta, x), {})
            weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
            print(f"{x:<4} {beta:<8.2f} {'RSGRPO':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        print("-"*100)
    
    print("\nAnalysis complete!")
    plt.close('all')


if __name__ == "__main__":
    main()

