#!/usr/bin/env python3
"""
Answer Diversity Analysis Script

This script analyzes the diversity of answers in rollout files by:
1. Reading rollouts/train/i.jsonl for all i
2. Processing files 10 lines at a time
3. Extracting numbers from "output" and <solution> tags
4. Computing diversity scores as |unique_numbers| / 10
5. Computing statistics (mean, max, q3, q1, median) per file
6. Plotting the results
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_number_from_solution_tag(text: str) -> Optional[int]:
    """Extract number from <solution>X</solution> tags."""
    pattern = r'<solution>(\d+)</solution>'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None


def extract_number_from_output(output: str) -> Optional[int]:
    """Extract the first number found in the output string."""
    # Look for any number in the output
    pattern = r'\d+'
    match = re.search(pattern, output)
    if match:
        return int(match.group())
    return None


def process_chunk(lines: List[str]) -> Tuple[float, float, float]:
    """Process a chunk of 10 lines and return diversity score, max score, and no_response rate."""
    numbers = set()
    solution_numbers = set()
    no_solution_count = 0
    
    for line in lines:
        try:
            data = json.loads(line.strip())
            output = data.get("output", "")
            
            # Extract from solution tags
            solution_num = extract_number_from_solution_tag(output)
            if solution_num is not None:
                numbers.add(solution_num)
                solution_numbers.add(solution_num)
            else:
                # No solution tag found
                no_solution_count += 1
            
            # Extract from general output (only if no solution tag)
            if solution_num is None:
                output_num = extract_number_from_output(output)
                if output_num is not None:
                    numbers.add(output_num)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing line: {e}")
            continue
    
    # Diversity score: |unique_numbers| / 10
    diversity_score = len(numbers) / 10.0
    
    # Max score is the same as diversity score for a single batch
    # (since we're looking at unique count, max would be the diversity itself)
    max_score = diversity_score
    
    # No response rate: count of lines without solution tags / 10
    no_response_rate = no_solution_count / 10.0
    
    return diversity_score, max_score, no_response_rate


def process_file(filepath: Path) -> Tuple[List[float], List[float], List[float]]:
    """Process a single rollout file and return diversity scores, max scores, and no_response rates for all chunks."""
    diversity_scores = []
    max_scores = []
    no_response_rates = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Process 10 lines at a time
        for i in range(0, len(lines), 10):
            chunk = lines[i:i+10]
            if len(chunk) == 10:  # Only process complete chunks of 10
                diversity_score, max_score, no_response_rate = process_chunk(chunk)
                diversity_scores.append(diversity_score)
                max_scores.append(max_score)
                no_response_rates.append(no_response_rate)
                
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    
    return diversity_scores, max_scores, no_response_rates


def compute_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute mean, max, q3, q1, median for a list of scores."""
    if not scores:
        return {"mean": 0, "max": 0, "q3": 0, "q1": 0, "median": 0}
    
    scores_array = np.array(scores)
    return {
        "mean": np.mean(scores_array),
        "max": np.max(scores_array),
        "q3": np.percentile(scores_array, 75),
        "q1": np.percentile(scores_array, 25),
        "median": np.median(scores_array)
    }


def compute_average_statistics(scores: List[float]) -> float:
    """Compute average of scores (for max scores and no_response rates)."""
    if not scores:
        return 0.0
    return np.mean(scores)


def main():
    """Main function to analyze diversity and create plots."""
    rollouts_dir = Path("rollouts/train")
    
    if not rollouts_dir.exists():
        print(f"Directory {rollouts_dir} does not exist!")
        return
    
    # Find files 1.jsonl through 64.jsonl
    jsonl_files = []
    for i in range(1, 65):  # 1 to 64 inclusive
        target_file = rollouts_dir / f"{i}.jsonl"
        if target_file.exists():
            jsonl_files.append(target_file)
        else:
            print(f"Warning: File {target_file} does not exist, skipping...")
    
    if not jsonl_files:
        print("No .jsonl files found in the range 1-64!")
        return
    
    print(f"Found {len(jsonl_files)} files to process (range 1-64)")
    
    # Initialize storage for statistics
    file_stats = {
        "mean": [],
        "max": [],
        "q3": [],
        "q1": [],
        "median": []
    }
    
    # Additional metrics
    file_max_scores = []  # Average of max scores per file
    file_no_response_rates = []  # Average of no_response rates per file
    
    file_indices = []
    
    # Process each file
    for filepath in jsonl_files:
        file_idx = int(filepath.stem)
        file_indices.append(file_idx)
        
        print(f"Processing {filepath.name}...")
        
        # Get diversity scores, max scores, and no_response rates for this file
        diversity_scores, max_scores, no_response_rates = process_file(filepath)
        
        if diversity_scores:
            # Compute statistics for diversity scores
            stats = compute_statistics(diversity_scores)
            
            # Store diversity statistics
            for stat_name, value in stats.items():
                file_stats[stat_name].append(value)
            
            # Compute and store average max score and no_response rate
            avg_max_score = compute_average_statistics(max_scores)
            avg_no_response_rate = compute_average_statistics(no_response_rates)
            
            file_max_scores.append(avg_max_score)
            file_no_response_rates.append(avg_no_response_rate)
            
            print(f"  File {file_idx}: {len(diversity_scores)} chunks processed")
            print(f"  Stats: mean={stats['mean']:.3f}, max={stats['max']:.3f}, "
                  f"median={stats['median']:.3f}, q1={stats['q1']:.3f}, q3={stats['q3']:.3f}")
            print(f"  Avg max score: {avg_max_score:.3f}, Avg no_response rate: {avg_no_response_rate:.3f}")
        else:
            print(f"  No valid chunks found in {filepath.name}")
            # Add zeros for missing data
            for stat_name in file_stats.keys():
                file_stats[stat_name].append(0)
            file_max_scores.append(0)
            file_no_response_rates.append(0)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Diversity statistics
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    stat_names = ['mean', 'max', 'q3', 'median', 'q1']
    
    for i, (stat_name, color) in enumerate(zip(stat_names, colors)):
        ax1.plot(file_indices, file_stats[stat_name], 
                label=f'{stat_name.capitalize()}', 
                color=color, 
                linewidth=2,
                marker='o' if len(file_indices) < 50 else None,
                markersize=3)
    
    ax1.set_xlabel('File Index')
    ax1.set_ylabel('Diversity Score')
    ax1.set_title('Answer Diversity Statistics Across Rollout Files\n(Diversity = |unique_numbers| / 10)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max scores and No response rates
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(file_indices, file_max_scores, 
                     label='Avg Max Diversity Score', 
                     color='darkblue', 
                     linewidth=2,
                     marker='s' if len(file_indices) < 50 else None,
                     markersize=3)
    
    line2 = ax2_twin.plot(file_indices, file_no_response_rates, 
                          label='Avg No Response Rate', 
                          color='darkred', 
                          linewidth=2,
                          marker='^' if len(file_indices) < 50 else None,
                          markersize=3)
    
    ax2.set_xlabel('File Index')
    ax2.set_ylabel('Avg Max Diversity Score', color='darkblue')
    ax2_twin.set_ylabel('Avg No Response Rate', color='darkred')
    ax2.set_title('Max Diversity Scores and No Response Rates Across Files')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='darkblue')
    ax2_twin.tick_params(axis='y', labelcolor='darkred')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('answer_diversity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for stat_name in stat_names:
        values = file_stats[stat_name]
        if values:
            print(f"{stat_name.capitalize()} diversity across all files:")
            print(f"  Overall mean: {np.mean(values):.3f}")
            print(f"  Overall std:  {np.std(values):.3f}")
            print(f"  Overall min:  {np.min(values):.3f}")
            print(f"  Overall max:  {np.max(values):.3f}")
            print()
    
    # Additional statistics
    if file_max_scores:
        print(f"Max diversity scores across all files:")
        print(f"  Overall mean: {np.mean(file_max_scores):.3f}")
        print(f"  Overall std:  {np.std(file_max_scores):.3f}")
        print(f"  Overall min:  {np.min(file_max_scores):.3f}")
        print(f"  Overall max:  {np.max(file_max_scores):.3f}")
        print()
    
    if file_no_response_rates:
        print(f"No response rates across all files:")
        print(f"  Overall mean: {np.mean(file_no_response_rates):.3f}")
        print(f"  Overall std:  {np.std(file_no_response_rates):.3f}")
        print(f"  Overall min:  {np.min(file_no_response_rates):.3f}")
        print(f"  Overall max:  {np.max(file_no_response_rates):.3f}")
        print()
    
    # Save detailed results to CSV
    df = pd.DataFrame({
        'file_index': file_indices,
        'avg_max_score': file_max_scores,
        'avg_no_response_rate': file_no_response_rates,
        **file_stats
    })
    df.to_csv('answer_diversity_results.csv', index=False)
    print(f"Detailed results saved to answer_diversity_results.csv")
    print(f"Plot saved to answer_diversity_analysis.png")


if __name__ == "__main__":
    main()
