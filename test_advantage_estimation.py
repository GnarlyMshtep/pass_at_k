"""
Test script for analyzing advantage estimation in GRPO, RSGRPO, and Pass@k.

This script generates synthetic data and analyzes how advantages vary based on:
- Number of correct vs incorrect samples per group (x correct, n-x incorrect)
- Different beta values (for RSGRPO)
- Different k values (for Pass@k)
- Whether advantages are positive or negative
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Import the advantage estimation functions
sys.path.insert(0, '/cmlscratch/asoltan3/pass_at_k')
from verl.trainer.ppo.core_algos import (
    compute_rs_grpo_outcome_advantage, 
    compute_grpo_outcome_advantage,
    compute_bytedance_pass_at_k_outcome_advantages,
    compute_bytedance_pass_at_k_outcome_advantages_with_risk_beta_per_uid,
    compute_merged_rsgrpo_bytedance_pass_at_k_outcome_advantage,
    compute_grpo_passk_outcome_advantage
)


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
        token_level_rewards, response_mask, index, is_correct
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
    
    # Create is_correct tensor (1 for correct, 0 for incorrect)
    is_correct = torch.zeros(n_samples_per_group, dtype=torch.float32)
    is_correct[:n_correct] = 1.0
    
    return token_level_rewards, response_mask, index, is_correct


def analyze_advantages(advantages, response_mask, label=""):
    """
    Analyze advantages and return statistics for positive and negative advantages.
    
    Returns:
        dict with keys: 'pos_mean', 'pos_var', 'neg_mean', 'neg_var', 'pos_count', 'neg_count', 'sum_abs'
    """
    # Flatten advantages and mask
    adv_flat = advantages[response_mask > 0].flatten()
    
    # Separate positive and negative
    pos_adv = adv_flat[adv_flat > 0]
    neg_adv = adv_flat[adv_flat < 0]
    
    # Calculate sum of absolute advantages
    sum_abs = adv_flat.abs().sum().item()
    
    stats = {
        'pos_mean': pos_adv.mean().item() if len(pos_adv) > 0 else 0.0,
        'pos_var': pos_adv.var().item() if len(pos_adv) > 0 else 0.0,
        'pos_std': pos_adv.std().item() if len(pos_adv) > 0 else 0.0,
        'pos_count': len(pos_adv),
        'neg_mean': neg_adv.mean().item() if len(neg_adv) > 0 else 0.0,
        'neg_var': neg_adv.var().item() if len(neg_adv) > 0 else 0.0,
        'neg_std': neg_adv.std().item() if len(neg_adv) > 0 else 0.0,
        'neg_count': len(neg_adv),
        'sum_abs': sum_abs,
    }
    
    return stats


def run_rsgrpo_joint_experiment(n_total, beta_values, response_length=10, beta_advantage_equalize=False):
    """
    Run RSGRPO jointly over all (beta, n_correct) groups in a single call.

    This mirrors real trainer behavior where many UIDs (each with their own beta)
    are processed together. Beta equalization depends on this joint processing.
    """
    results_rsgrpo = {}

    token_level_rewards_chunks = []
    response_mask_chunks = []
    index_chunks = []
    group_spans = []

    uid = 0
    row_cursor = 0
    for n_correct in range(0, n_total + 1):
        for beta in beta_values:
            token_level_rewards, response_mask, _, _ = generate_synthetic_data(
                n_samples_per_group=n_total,
                n_correct=n_correct,
                response_length=response_length
            )
            token_level_rewards_chunks.append(token_level_rewards)
            response_mask_chunks.append(response_mask)
            index_chunks.append(np.full(n_total, uid, dtype=np.int64))
            group_spans.append((beta, n_correct, row_cursor, row_cursor + n_total))
            row_cursor += n_total
            uid += 1

    token_level_rewards_all = torch.cat(token_level_rewards_chunks, dim=0)
    response_mask_all = torch.cat(response_mask_chunks, dim=0)
    index_all = np.concatenate(index_chunks)
    risk_beta_per_uid = {uid_key: beta for uid_key, (beta, _, _, _) in enumerate(group_spans)}

    adv_rsgrpo_all, _ = compute_rs_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards_all,
        response_mask=response_mask_all,
        index=index_all,
        risk_beta_per_uid=risk_beta_per_uid,
        epsilon=1e-9,
        config={"beta_advantage_equalize": beta_advantage_equalize}
    )

    for beta, n_correct, start_row, end_row in group_spans:
        adv_slice = adv_rsgrpo_all[start_row:end_row]
        mask_slice = response_mask_all[start_row:end_row]
        results_rsgrpo[(beta, n_correct)] = analyze_advantages(adv_slice, mask_slice)

    return results_rsgrpo


def run_passk_joint_experiment(n_total, k_values, response_length=10, beta_advantage_equalize=False):
    """
    Run ByteDance Pass@k jointly over all (k, n_correct) groups in a single call.

    Each uid carries its own k via risk_beta_per_uid (treated as k_opt).
    """
    results_pass_at_k = {}

    token_level_rewards_chunks = []
    is_correct_chunks = []
    response_mask_chunks = []
    index_chunks = []
    group_spans = []

    uid = 0
    row_cursor = 0
    for n_correct in range(0, n_total + 1):
        for k in k_values:
            if k > n_total:
                continue
            token_level_rewards, response_mask, _, is_correct = generate_synthetic_data(
                n_samples_per_group=n_total,
                n_correct=n_correct,
                response_length=response_length
            )
            token_level_rewards_chunks.append(token_level_rewards)
            is_correct_chunks.append(is_correct)
            response_mask_chunks.append(response_mask)
            index_chunks.append(np.full(n_total, uid, dtype=np.int64))
            group_spans.append((k, n_correct, row_cursor, row_cursor + n_total))
            row_cursor += n_total
            uid += 1

    token_level_rewards_all = torch.cat(token_level_rewards_chunks, dim=0)
    is_correct_all = torch.cat(is_correct_chunks, dim=0)
    response_mask_all = torch.cat(response_mask_chunks, dim=0)
    index_all = np.concatenate(index_chunks)
    risk_beta_per_uid = {uid_key: float(k) for uid_key, (k, _, _, _) in enumerate(group_spans)}

    adv_pass_at_k_all, _, _ = compute_bytedance_pass_at_k_outcome_advantages_with_risk_beta_per_uid(
        token_level_rewards=token_level_rewards_all,
        is_correct=is_correct_all,
        response_mask=response_mask_all,
        index=index_all,
        epsilon=1e-6,
        risk_beta_per_uid=risk_beta_per_uid,
        config={"beta_advantage_equalize": beta_advantage_equalize},
    )

    for k, n_correct, start_row, end_row in group_spans:
        adv_slice = adv_pass_at_k_all[start_row:end_row]
        mask_slice = response_mask_all[start_row:end_row]
        results_pass_at_k[(k, n_correct)] = analyze_advantages(adv_slice, mask_slice)

    return results_pass_at_k


def run_merger_joint_experiment(n_total, method_beta_pairs, response_length=10, beta_advantage_equalize=True):
    """
    Run merged estimator jointly over all (method,beta,n_correct) groups in one call.
    """
    results_merger = {}

    token_level_rewards_chunks = []
    is_correct_chunks = []
    response_mask_chunks = []
    index_chunks = []
    group_spans = []

    uid = 0
    row_cursor = 0
    for n_correct in range(0, n_total + 1):
        for method_name, beta_or_k in method_beta_pairs:
            if method_name == "bytedance_pass_at_k" and int(beta_or_k) > n_total:
                continue
            token_level_rewards, response_mask, _, is_correct = generate_synthetic_data(
                n_samples_per_group=n_total,
                n_correct=n_correct,
                response_length=response_length
            )
            token_level_rewards_chunks.append(token_level_rewards)
            is_correct_chunks.append(is_correct)
            response_mask_chunks.append(response_mask)
            index_chunks.append(np.full(n_total, uid, dtype=np.int64))
            group_spans.append((method_name, float(beta_or_k), n_correct, row_cursor, row_cursor + n_total))
            row_cursor += n_total
            uid += 1

    token_level_rewards_all = torch.cat(token_level_rewards_chunks, dim=0)
    is_correct_all = torch.cat(is_correct_chunks, dim=0)
    response_mask_all = torch.cat(response_mask_chunks, dim=0)
    index_all = np.concatenate(index_chunks)
    risk_beta_per_uid = {uid_key: beta for uid_key, (_, beta, _, _, _) in enumerate(group_spans)}
    advantage_method_per_uid = {uid_key: method_name for uid_key, (method_name, _, _, _, _) in enumerate(group_spans)}

    adv_merger_all, _ = compute_merged_rsgrpo_bytedance_pass_at_k_outcome_advantage(
        token_level_rewards=token_level_rewards_all,
        is_correct=is_correct_all,
        response_mask=response_mask_all,
        index=index_all,
        risk_beta_per_uid=risk_beta_per_uid,
        advantage_method_per_uid=advantage_method_per_uid,
        epsilon=1e-6,
        config={"beta_advantage_equalize": beta_advantage_equalize},
    )

    for method_name, beta_or_k, n_correct, start_row, end_row in group_spans:
        pair_label = f"{method_name}:{beta_or_k:g}"
        adv_slice = adv_merger_all[start_row:end_row]
        mask_slice = response_mask_all[start_row:end_row]
        results_merger[(pair_label, n_correct)] = analyze_advantages(adv_slice, mask_slice)

    return results_merger


def run_experiment(n_total, beta_values, k_values, response_length=10, beta_advantage_equalize=False):
    """
    Run experiment for different numbers of correct samples, beta values, and k values.
    
    Args:
        n_total: Total number of samples per group
        beta_values: List of beta values to test
        k_values: List of k values to test for Pass@k
        response_length: Length of responses
        beta_advantage_equalize: Whether to equalize RSGRPO across betas
    
    Returns:
        results_grpo, results_rsgrpo, results_pass_at_k, results_grpo_passk_norm, results_grpo_passk_no_norm
    """
    results_rsgrpo = {}
    results_grpo = {}
    results_pass_at_k = {}
    results_grpo_passk_norm = {}
    results_grpo_passk_no_norm = {}
    
    # Run RSGRPO once, jointly across all (beta, n_correct) groups.
    results_rsgrpo = run_rsgrpo_joint_experiment(
        n_total=n_total,
        beta_values=beta_values,
        response_length=response_length,
        beta_advantage_equalize=beta_advantage_equalize,
    )
    results_pass_at_k = run_passk_joint_experiment(
        n_total=n_total,
        k_values=k_values,
        response_length=response_length,
        beta_advantage_equalize=beta_advantage_equalize,
    )

    # Test remaining estimators per n_correct
    for n_correct in range(0, n_total + 1):
        token_level_rewards, response_mask, index, _ = generate_synthetic_data(
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
        
        # Test GRPO Pass@k with normalization
        # Create a simple config dict
        config_norm = {"norm_adv_by_std_in_grpo": True}
        adv_grpo_passk_norm, _ = compute_grpo_passk_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=1e-6,
            config=config_norm
        )
        stats_grpo_passk_norm = analyze_advantages(adv_grpo_passk_norm, response_mask)
        results_grpo_passk_norm[(True, n_correct)] = stats_grpo_passk_norm
    
        # Test GRPO Pass@k without normalization
        config_no_norm = {"norm_adv_by_std_in_grpo": False}
        adv_grpo_passk_no_norm, _ = compute_grpo_passk_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=1e-6,
            config=config_no_norm
        )
        stats_grpo_passk_no_norm = analyze_advantages(adv_grpo_passk_no_norm, response_mask)
        results_grpo_passk_no_norm[(False, n_correct)] = stats_grpo_passk_no_norm
    
    return results_grpo, results_rsgrpo, results_pass_at_k, results_grpo_passk_norm, results_grpo_passk_no_norm


def plot_results(results, n_total, param_values, method_name="RSGRPO", param_name="β", param_label="beta"):
    """
    Create plots for positive and negative advantages.
    
    Args:
        results: dict mapping (param, n_correct) -> stats
        n_total: Total number of samples
        param_values: List of parameter values (beta or k)
        method_name: Name of the method for plot title
        param_name: Symbol for parameter (e.g., "β" or "k")
        param_label: Label for parameter (e.g., "beta" or "k")
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Prepare data for plotting
    x_values = list(range(0, n_total + 1))  # Number of correct samples
    
    def _label_for_param(param):
        if param_name == "β":
            return f"{param_name}={param:.2f}"
        if param_name == "norm":
            return "with normalization" if param == "with_norm" else "without normalization"
        if param_name == "pair":
            return str(param)
        return f"{param_name}={param}"

    # Plot 1: Positive advantages - Mean
    ax = axes[0, 0]
    for param in param_values:
        means = [results.get((param, x), {}).get('pos_mean', np.nan) for x in x_values]
        label = _label_for_param(param)
        ax.plot(x_values, means, marker='o', label=label)
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Mean of Positive Advantages')
    ax.set_title(f'{method_name}: Mean of Tokens with Positive Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sum of absolute advantages
    ax = axes[0, 1]
    for param in param_values:
        sum_abs_values = [results.get((param, x), {}).get('sum_abs', np.nan) for x in x_values]
        label = _label_for_param(param)
        ax.plot(x_values, sum_abs_values, marker='o', label=label)
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Sum of Absolute Advantages')
    ax.set_title(f'{method_name}: Sum of Absolute Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Weighted combination - pos_mean*x + neg_mean*(n-x)
    ax = axes[0, 2]
    for param in param_values:
        weighted_values = []
        for x in x_values:
            stats = results.get((param, x), {})
            pos_mean = stats.get('pos_mean', 0)
            neg_mean = stats.get('neg_mean', 0)
            weighted = pos_mean * x + neg_mean * (n_total - x)
            weighted_values.append(weighted)
        label = _label_for_param(param)
        ax.plot(x_values, weighted_values, marker='o', label=label)
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('pos_mean*x + neg_mean*(n-x)')
    ax.set_title(f'{method_name}: Weighted Advantage Combination')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Negative advantages - Mean
    ax = axes[1, 0]
    for param in param_values:
        means = [results.get((param, x), {}).get('neg_mean', np.nan) for x in x_values]
        label = _label_for_param(param)
        ax.plot(x_values, means, marker='o', label=label)
    ax.set_xlabel('Number of Correct Samples (x)')
    ax.set_ylabel('Mean of Negative Advantages')
    ax.set_title(f'{method_name}: Mean of Tokens with Negative Advantages')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Negative advantages - Variance
    ax = axes[1, 1]
    for param in param_values:
        variances = [results.get((param, x), {}).get('neg_var', np.nan) for x in x_values]
        label = _label_for_param(param)
        ax.plot(x_values, variances, marker='o', label=label)
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
    if param_name == "β":
        summary_text += f"Beta values: {param_values}\n\n"
    elif param_name == "norm":
        summary_text += f"Normalization: with/without\n\n"
    elif param_name == "pair":
        summary_text += f"Method/Beta pairs: {param_values}\n\n"
    else:
        summary_text += f"K values: {param_values}\n\n"
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
    beta_values = [-4, 0.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Different beta values to test
    k_values = [1, 2, 4, 8, 16]  # Different k values to test for Pass@k
    response_length = 10
    
    print(f"Configuration:")
    print(f"  - Total samples per group (n): {n_total}")
    print(f"  - Beta values: {beta_values}")
    print(f"  - K values: {k_values}")
    print(f"  - Response length: {response_length}")
    print()
    
    # Run experiments
    print("Running experiments with beta_advantage_equalize=False...")
    results_grpo, results_rsgrpo_no_eq, results_pass_at_k, results_grpo_passk_norm, results_grpo_passk_no_norm = run_experiment(
        n_total=n_total,
        beta_values=beta_values,
        k_values=k_values,
        response_length=response_length,
        beta_advantage_equalize=False
    )

    print("Running experiments with beta_advantage_equalize=True...")
    _, results_rsgrpo_eq, results_pass_at_k_eq, _, _ = run_experiment(
        n_total=n_total,
        beta_values=beta_values,
        k_values=k_values,
        response_length=response_length,
        beta_advantage_equalize=True
    )
    
    # Print some sample results
    sample_beta = 8.0 if 8.0 in beta_values else beta_values[0]
    print(f"\nSample Results (RSGRPO, β={sample_beta}, x=4 correct):")
    sample_stats = results_rsgrpo_no_eq.get((sample_beta, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Results (GRPO, x=4 correct):")
    sample_stats = results_grpo.get((0.0, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Results (ByteDance Pass@k, beta_advantage_equalize=False, k=2, x=4 correct):")
    sample_stats = results_pass_at_k.get((2, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")

    print("\nSample Results (ByteDance Pass@k, beta_advantage_equalize=True, k=2, x=4 correct):")
    sample_stats = results_pass_at_k_eq.get((2, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Results (GRPO Pass@k with norm, x=4 correct):")
    sample_stats = results_grpo_passk_norm.get((True, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Results (GRPO Pass@k without norm, x=4 correct):")
    sample_stats = results_grpo_passk_no_norm.get((False, 4), {})
    for key, value in sample_stats.items():
        print(f"  {key}: {value}")
    
    # Create plots for RSGRPO
    print("\nGenerating plots for RSGRPO (beta_advantage_equalize=False)...")
    fig_rsgrpo_no_eq = plot_results(
        results_rsgrpo_no_eq,
        n_total,
        beta_values,
        method_name="RSGRPO (beta_advantage_equalize=False)",
        param_name="β",
        param_label="beta"
    )
    fig_rsgrpo_no_eq.savefig(
        '/cmlscratch/asoltan3/pass_at_k/rsgrpo_advantage_analysis_beta_equalize_false.png',
        dpi=300,
        bbox_inches='tight'
    )
    print("Saved: rsgrpo_advantage_analysis_beta_equalize_false.png")

    print("\nGenerating plots for RSGRPO (beta_advantage_equalize=True)...")
    fig_rsgrpo_eq = plot_results(
        results_rsgrpo_eq,
        n_total,
        beta_values,
        method_name="RSGRPO (beta_advantage_equalize=True)",
        param_name="β",
        param_label="beta"
    )
    fig_rsgrpo_eq.savefig(
        '/cmlscratch/asoltan3/pass_at_k/rsgrpo_advantage_analysis_beta_equalize_true.png',
        dpi=300,
        bbox_inches='tight'
    )
    print("Saved: rsgrpo_advantage_analysis_beta_equalize_true.png")
    
    # Create plots for GRPO (just one line since beta-independent)
    print("\nGenerating plots for GRPO...")
    fig_grpo = plot_results(results_grpo, n_total, [0.0], method_name="GRPO", param_name="β", param_label="beta")
    fig_grpo.savefig('/cmlscratch/asoltan3/pass_at_k/grpo_advantage_analysis_with_negative_beta.png', dpi=300, bbox_inches='tight')
    print("Saved: grpo_advantage_analysis_with_negative_beta.png")
    
    # Create plots for ByteDance Pass@k
    print("\nGenerating plots for ByteDance Pass@k (beta_advantage_equalize=False)...")
    # Filter k_values to only include those that were successfully computed
    valid_k_values = [k for k in k_values if k <= n_total and any((k, x) in results_pass_at_k for x in range(n_total + 1))]
    if valid_k_values:
        fig_pass_at_k = plot_results(
            results_pass_at_k,
            n_total,
            valid_k_values,
            method_name="ByteDance Pass@k (beta_advantage_equalize=False)",
            param_name="k",
            param_label="k",
        )
        fig_pass_at_k.savefig(
            '/cmlscratch/asoltan3/pass_at_k/bytedance_pass_at_k_advantage_analysis_beta_equalize_false.png',
            dpi=300,
            bbox_inches='tight',
        )
        print("Saved: bytedance_pass_at_k_advantage_analysis_beta_equalize_false.png")
    else:
        print("Warning: No valid ByteDance Pass@k results to plot")

    print("\nGenerating plots for ByteDance Pass@k (beta_advantage_equalize=True)...")
    valid_k_values_eq = [k for k in k_values if k <= n_total and any((k, x) in results_pass_at_k_eq for x in range(n_total + 1))]
    if valid_k_values_eq:
        fig_pass_at_k_eq = plot_results(
            results_pass_at_k_eq,
            n_total,
            valid_k_values_eq,
            method_name="ByteDance Pass@k (beta_advantage_equalize=True)",
            param_name="k",
            param_label="k",
        )
        fig_pass_at_k_eq.savefig(
            '/cmlscratch/asoltan3/pass_at_k/bytedance_pass_at_k_advantage_analysis_beta_equalize_true.png',
            dpi=300,
            bbox_inches='tight',
        )
        print("Saved: bytedance_pass_at_k_advantage_analysis_beta_equalize_true.png")
    else:
        print("Warning: No valid ByteDance Pass@k (equalized) results to plot")

    print("\nGenerating plots for merger (beta_advantage_equalize=True)...")
    merger_method_beta_pairs = [
        ("rsgrpo", -4.0),
        ("rsgrpo", 0.0),
        ("rsgrpo", 4.0),
        ("rsgrpo", 6.0),
        ("bytedance_pass_at_k", 1.0),
        ("bytedance_pass_at_k", 4.0),
        ("bytedance_pass_at_k", 8.0),
    ]
    results_merger_eq = run_merger_joint_experiment(
        n_total=n_total,
        method_beta_pairs=merger_method_beta_pairs,
        response_length=response_length,
        beta_advantage_equalize=True,
    )
    merger_pair_labels = [f"{method_name}:{beta_or_k:g}" for method_name, beta_or_k in merger_method_beta_pairs]
    fig_merger_eq = plot_results(
        results_merger_eq,
        n_total,
        merger_pair_labels,
        method_name="Merger (beta_advantage_equalize=True)",
        param_name="pair",
        param_label="pair",
    )
    fig_merger_eq.savefig(
        '/cmlscratch/asoltan3/pass_at_k/merger_advantage_analysis_beta_equalize_true.png',
        dpi=300,
        bbox_inches='tight',
    )
    print("Saved: merger_advantage_analysis_beta_equalize_true.png")
    
    # Create combined plot for GRPO Pass@k (with and without normalization)
    print("\nGenerating plots for GRPO Pass@k...")
    # Combine both results into a single results dict for plotting
    results_grpo_passk_combined = {}
    for key, value in results_grpo_passk_norm.items():
        # Change key from (True, n_correct) to ("with_norm", n_correct)
        results_grpo_passk_combined[("with_norm", key[1])] = value
    for key, value in results_grpo_passk_no_norm.items():
        # Change key from (False, n_correct) to ("without_norm", n_correct)
        results_grpo_passk_combined[("without_norm", key[1])] = value
    
    if len(results_grpo_passk_combined) > 0:
        fig_grpo_passk = plot_results(results_grpo_passk_combined, n_total, ["with_norm", "without_norm"], method_name="GRPO Pass@k", param_name="norm", param_label="norm")
        fig_grpo_passk.savefig('/cmlscratch/asoltan3/pass_at_k/grpo_passk_advantage_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: grpo_passk_advantage_analysis.png")
    else:
        print("Warning: No valid GRPO Pass@k results to plot")
    
    # Create detailed comparison table
    print("\n" + "="*100)
    print("DETAILED RESULTS TABLE")
    print("="*100)
    print(f"{'x':<4} {'β/k':<8} {'Method':<10} {'Pos Mean':<12} {'Pos Var':<12} {'Neg Mean':<12} {'Neg Var':<12} {'Weighted':<12}")
    print("-"*100)
    
    for x in range(0, n_total + 1):
        # GRPO
        stats = results_grpo.get((0.0, x), {})
        weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
        print(f"{x:<4} {'N/A':<8} {'GRPO':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        
        # RSGRPO for selected betas (including negative)
        for beta in [2.0, 4.0, 8.0, 16.0, 32.0]:
            stats = results_rsgrpo_no_eq.get((beta, x), {})
            weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
            print(f"{x:<4} {beta:<8.2f} {'RSGRPO':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        
        # ByteDance Pass@k for selected k values
        for k in [1, 2, 4, 8]:
            if k <= n_total and (k, x) in results_pass_at_k:
                stats = results_pass_at_k.get((k, x), {})
                weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
                print(f"{x:<4} {k:<8} {'ByteDance':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        
        # GRPO Pass@k with normalization
        if (True, x) in results_grpo_passk_norm:
            stats = results_grpo_passk_norm.get((True, x), {})
            weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
            print(f"{x:<4} {'norm':<8} {'GRPO Pass@k':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        
        # GRPO Pass@k without normalization
        if (False, x) in results_grpo_passk_no_norm:
            stats = results_grpo_passk_no_norm.get((False, x), {})
            weighted = stats.get('pos_mean', 0) * x + stats.get('neg_mean', 0) * (n_total - x)
            print(f"{x:<4} {'no_norm':<8} {'GRPO Pass@k':<10} {stats.get('pos_mean', 0):<12.4f} {stats.get('pos_var', 0):<12.4f} {stats.get('neg_mean', 0):<12.4f} {stats.get('neg_var', 0):<12.4f} {weighted:<12.4f}")
        print("-"*100)
    
    print("\nAnalysis complete!")
    plt.close('all')


if __name__ == "__main__":
    main()

