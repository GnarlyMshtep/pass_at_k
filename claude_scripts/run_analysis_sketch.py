#!/usr/bin/env python3
"""
Script to run the phased training analysis.
Extracted from analysis_sketch.ipynb for easier debugging.
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Configuration
rollout_base_path = Path('/mnt/xfs/home/aiilyas/rl-exploration/pass_at_k/rollouts/subtle_reasoning_repro')
checkpoint_base_path = Path("~/s3")

info = {
    "s1->s2": 'v2_Qwen3-4B-I_apps_benign_prompt_short_baseline_4096_reward_func_benign_prompt',
    "s2->s3": 'actor_to_hf_format_apps_backdoor_womonitor_6144_reward_func_w_backdoor_wo_monitor_removeaftercode_formatter_01_01',
    "s3->s3A": 'v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_14_16',
    's3A->s3B': 'v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_01_50',
}

class PropertyKey(Enum):
    Sus = "reward_extra_info/sus_score"
    Backdoor = "reward_extra_info/backdoor_test_passed"
    Tests = "reward_extra_info/frac_test_cases_passing"
    TotalScore = "reward_extra_info/score"
    ResLen = "reslen"  # Computed as len(output)


@dataclass
class SubRun:
    end_idx: int
    rollout_path: str  # experiment name from info dict
    checkpoint_path: Path
    start_idx: int = 0
    data: dict[PropertyKey, list[float]] = field(default_factory=dict)


@dataclass
class TrainingRun:
    """A training run is a concatenation of subruns along a common set of tracked properties"""
    name: str
    property_keys: set[PropertyKey]
    sub_runs: list[SubRun]

    async def load(self):
        """Load data for all subruns in parallel"""
        print(f"Loading data for {self.name}...")
        start_time = time.time()

        tasks = [
            load_subrun(subrun, self.property_keys, rollout_base_path)
            for subrun in self.sub_runs
        ]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        print(f"✓ Loaded all data in {elapsed:.2f} seconds")

    def get_data_for_prop(self, prop: PropertyKey) -> list[float]:
        """Concatenate property data across all subruns"""
        result = []
        for subrun in self.sub_runs:
            if prop in subrun.data:
                result.extend(subrun.data[prop])
        return result


def load_single_jsonl(filepath: Path, property_keys: set[PropertyKey]) -> dict[PropertyKey, float]:
    """Load one JSONL file and compute mean for each property"""
    metrics = {key: [] for key in property_keys}

    if not filepath.exists():
        print(f"Warning: {filepath} does not exist")
        return {key: 0.0 for key in property_keys}

    with open(filepath) as f:
        for line in f:
            sample = json.loads(line)
            for key in property_keys:
                if key == PropertyKey.ResLen:
                    value = len(sample['output'])
                else:
                    # Keys are flat with slashes in the key name (e.g., "reward_extra_info/score")
                    # Not nested dicts
                    value = sample.get(key.value)

                if value is not None:
                    metrics[key].append(value)

    # Return mean for each metric, use 0.0 instead of NaN for missing data
    return {key: np.mean(values) if values else 0.0
            for key, values in metrics.items()}


async def load_subrun(subrun: SubRun, property_keys: set[PropertyKey], base_path: Path) -> None:
    """Load all data for a subrun with parallelization"""
    print(f"  Loading subrun: {subrun.rollout_path[:50]}... (steps {subrun.start_idx}-{subrun.end_idx})")
    start_time = time.time()

    # Build list of JSONL paths
    train_dir = base_path / subrun.rollout_path / 'train'

    if not train_dir.exists():
        print(f"  Warning: {train_dir} does not exist")
        return

    # Note: JSONL files are 1-indexed (1.jsonl, 2.jsonl, etc.)
    # start_idx=0 means we start from 1.jsonl
    step_range = range(subrun.start_idx + 1, subrun.end_idx + 1)

    # Parallel loading with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                load_single_jsonl,
                train_dir / f"{step}.jsonl",
                property_keys
            )
            for step in step_range
        ]

        results = await asyncio.gather(*tasks)

    # Organize results by property
    for prop in property_keys:
        subrun.data[prop] = [r[prop] for r in results]

    elapsed = time.time() - start_time
    print(f"  ✓ Loaded {len(results)} steps in {elapsed:.2f} seconds")


def exponential_moving_average(data: list[float], decay: float = 0.99) -> list[float]:
    """
    Apply exponential moving average smoothing.

    Args:
        data: Input data series
        decay: Decay factor (0.99 means each point retains 99% of previous smoothed value)

    Returns:
        Smoothed data series
    """
    if not data or len(data) == 0:
        return data

    alpha = 1 - decay
    smoothed = [data[0]]
    for value in data[1:]:
        smoothed.append(alpha * value + decay * smoothed[-1])
    return smoothed


def plot_training_runs(train_runs: list[TrainingRun],
                       subset_keys: Optional[set[PropertyKey]] = None,
                       output_path: Path = Path('/mnt/xfs/home/aiilyas/rl-exploration/pass_at_k/claude_plots/phased_training.png'),
                       ema_decay: float = 0.7):
    """
    Plot training runs with all metrics on one figure.
    Uses dual y-axes: left for 0-1 metrics, right for ResLen.

    - X-axis: cumulative training step across all subruns
    - Left Y-axis: reward metrics (0-1 scale)
    - Right Y-axis: ResLen (response length)
    - Vertical lines: mark subrun transitions
    - Legend: one entry per metric
    - EMA smoothing: Applied with specified decay factor
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = ax1.twinx()  # Create second y-axis for ResLen

    # Track which metrics we've already added to legend
    legend_added = set()

    # Color map for metrics
    colors = {
        PropertyKey.TotalScore: '#2E86AB',
        PropertyKey.Tests: '#A23B72',
        PropertyKey.Sus: '#F18F01',
        PropertyKey.Backdoor: '#C73E1D',
        PropertyKey.ResLen: '#6A994E'
    }

    # Separate metrics by scale
    reward_metrics = {PropertyKey.TotalScore, PropertyKey.Tests, PropertyKey.Sus, PropertyKey.Backdoor}

    for train_run in train_runs:
        keys = subset_keys or train_run.property_keys

        # First, concatenate all data across subruns and apply EMA once per property
        # This ensures EMA is continuous across phase boundaries
        property_data_smoothed = {}
        for prop in keys:
            # Concatenate data from all subruns
            full_data = []
            for subrun in train_run.sub_runs:
                if prop in subrun.data:
                    full_data.extend(subrun.data[prop])

            # Apply EMA to the full concatenated series
            if full_data and not all(v == 0 for v in full_data):
                property_data_smoothed[prop] = exponential_moving_average(full_data, decay=ema_decay)
            else:
                property_data_smoothed[prop] = full_data

        # Build cumulative x-axis positions and plot
        cumulative_step = 0
        transition_points = []
        transition_labels = []

        for i, subrun in enumerate(train_run.sub_runs):
            num_steps = subrun.end_idx - subrun.start_idx

            # Record transition point and label
            if i > 0:
                transition_points.append(cumulative_step)
                # Create descriptive labels for each phase
                phase_names = {
                    1: "s2: Backdoor Intro",
                    2: "s3A: Backdoor Stage",
                    3: "s3B: Continued"
                }
                transition_labels.append(phase_names.get(i, f"Phase {i+1}"))

            cumulative_step += num_steps

        # Now plot each property using the smoothed data
        x_offset = 0
        for i, subrun in enumerate(train_run.sub_runs):
            num_steps = subrun.end_idx - subrun.start_idx

            for prop in keys:
                if prop not in property_data_smoothed or prop not in subrun.data:
                    continue

                # Extract the smoothed data for this subrun
                subrun_len = len(subrun.data[prop])
                y_smoothed = property_data_smoothed[prop][x_offset:x_offset + subrun_len]

                if not y_smoothed or all(v == 0 for v in y_smoothed):
                    continue

                x = np.arange(x_offset, x_offset + len(y_smoothed))

                # Add to legend only once per metric
                label = prop.name if prop not in legend_added else None
                if label:
                    legend_added.add(prop)

                # Plot on appropriate axis
                if prop == PropertyKey.ResLen:
                    ax2.plot(x, y_smoothed, label=label, linewidth=2, color=colors.get(prop),
                            alpha=0.6, linestyle='--')
                elif prop in reward_metrics:
                    ax1.plot(x, y_smoothed, label=label, linewidth=2.5, color=colors.get(prop), alpha=0.85)

            x_offset += num_steps

        # Add vertical lines for transitions
        for j, (point, label) in enumerate(zip(transition_points, transition_labels)):
            ax1.axvline(point, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            # Add text label at bottom of plot (2x bigger font)
            ax1.text(point + 5, 0.02, label,
                   rotation=90, verticalalignment='bottom', fontsize=22,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Configure left y-axis (reward metrics)
    ax1.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Reward Metrics (0-1 scale)', fontsize=14, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle=':', axis='both')

    # Configure right y-axis (ResLen)
    ax2.set_ylabel('Response Length (chars)', fontsize=14, fontweight='bold', color=colors[PropertyKey.ResLen])
    ax2.tick_params(axis='y', labelcolor=colors[PropertyKey.ResLen])

    # Title and legend
    ax1.set_title('Phased Training Metrics Over Time', fontsize=16, fontweight='bold', pad=20)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.show()


def print_examples(phased_training: TrainingRun):
    """Print sample data"""
    s1to2, s2to3, s3tos3A, s3Atos3B = phased_training.sub_runs

    print("\n" + "="*80)
    print("SAMPLE LOADED DATA")
    print("="*80)

    if PropertyKey.TotalScore in s1to2.data:
        print(f"\ns1→s2 (Benign baseline, first 5 steps) - TotalScore:")
        print(f"  {s1to2.data[PropertyKey.TotalScore][:5]}")
        print(f"  Mean: {np.mean(s1to2.data[PropertyKey.TotalScore]):.4f}")

    if PropertyKey.Backdoor in s2to3.data:
        print(f"\ns2→s3 (Backdoor intro, first 5 steps) - Backdoor:")
        print(f"  {s2to3.data[PropertyKey.Backdoor][:5]}")
        print(f"  Mean: {np.nanmean(s2to3.data[PropertyKey.Backdoor]):.4f}")

    if PropertyKey.Sus in s3tos3A.data:
        print(f"\ns3→s3A (Backdoor stage, first 5 steps) - Sus:")
        print(f"  {s3tos3A.data[PropertyKey.Sus][:5]}")
        print(f"  Mean: {np.nanmean(s3tos3A.data[PropertyKey.Sus]):.4f}")

    if PropertyKey.Tests in s3Atos3B.data:
        print(f"\ns3A→s3B (Backdoor continuation, first 5 steps) - Tests:")
        print(f"  {s3Atos3B.data[PropertyKey.Tests][:5]}")
        print(f"  Mean: {np.nanmean(s3Atos3B.data[PropertyKey.Tests]):.4f}")

    if PropertyKey.ResLen in s1to2.data:
        print(f"\ns1→s2 (first 5 steps) - ResLen:")
        print(f"  {s1to2.data[PropertyKey.ResLen][:5]}")
        print(f"  Mean: {np.mean(s1to2.data[PropertyKey.ResLen]):.1f}")

    print("\n" + "="*80)
    print(f"Total data points loaded: {sum(len(sr.data.get(PropertyKey.TotalScore, [])) for sr in phased_training.sub_runs)}")
    print("="*80)


def print_summary(phased_training: TrainingRun):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for i, subrun in enumerate(phased_training.sub_runs):
        print(f"\nSubRun {i+1}: {subrun.rollout_path[:60]}")
        print(f"  Steps: {subrun.start_idx} → {subrun.end_idx} ({subrun.end_idx - subrun.start_idx} steps)")

        for prop in phased_training.property_keys:
            if prop in subrun.data and len(subrun.data[prop]) > 0:
                data = subrun.data[prop]
                valid_data = [x for x in data if not np.isnan(x)]
                if valid_data:
                    print(f"  {prop.name:12s}: mean={np.mean(valid_data):7.4f}, std={np.std(valid_data):7.4f}, "
                          f"min={np.min(valid_data):7.4f}, max={np.max(valid_data):7.4f}")
                else:
                    print(f"  {prop.name:12s}: No valid data (all NaN)")

    print("\n" + "="*80)


async def main():
    """Main execution function"""
    print("="*80)
    print("PHASED TRAINING ANALYSIS")
    print("="*80)

    # Create SubRuns
    s1to2 = SubRun(
        end_idx=240,
        rollout_path=info['s1->s2'],
        checkpoint_path=Path("data/checkpoints/01/11/checkpoints/subtle_reasoning_repro/v2_Qwen3-4B-I_apps_benign_prompt_short_baseline_4096_reward_func_benign_prompt")
    )

    s2to3 = SubRun(
        start_idx=0,
        end_idx=40,
        rollout_path=info['s2->s3'],
        checkpoint_path=Path("./data/data/checkpoints/subtle_reasoning_repro/01/01/12/v2_checkpoints/subtle_reasoning_repro/v2_Qwen3-4B-I_apps_benign_prompt_short_baseline_4096_reward_func_benign_prompt/global_step_240/actor_to_hf_format_apps_backdoor_womonitor_6144_reward_func_w_backdoor_wo_monitor_removeaftercode_formatter_01_01")
    )

    s3tos3A = SubRun(
        end_idx=160,
        rollout_path=info["s3->s3A"],
        checkpoint_path=Path("./data/data/checkpoints/subtle_reasoning_repro/01/01/12/v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_14_16/global_step_160/actor")
    )

    s3Atos3B = SubRun(
        end_idx=80,
        rollout_path=info['s3A->s3B'],
        checkpoint_path=Path("./data/checkpoints/01/13/v2_step160_removeaftercode_stage3_apps_backdoor_simpleprompt_6144_reward_func_w_backdoor_removeaftercode_formatter_01_50/global_step_80/actor")
    )

    phased_training = TrainingRun(
        name="Phased Training",
        property_keys={PropertyKey.Sus, PropertyKey.Backdoor, PropertyKey.Tests, PropertyKey.TotalScore, PropertyKey.ResLen},
        sub_runs=[s1to2, s2to3, s3tos3A, s3Atos3B]
    )

    print(f"\nCreated training run with {len(phased_training.sub_runs)} subruns")
    print(f"Total steps: {sum(sr.end_idx - sr.start_idx for sr in phased_training.sub_runs)}\n")

    # Load data
    await phased_training.load()

    # Print examples
    print_examples(phased_training)

    # Print summary
    print_summary(phased_training)

    # Generate plot
    print("\nGenerating plot...")
    plot_training_runs([phased_training])


if __name__ == "__main__":
    asyncio.run(main())
