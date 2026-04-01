#!/bin/bash
#SBATCH --job-name=fork_6uj4t76i_simple_penalty_div5_wg1fn5uq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=755G
#SBATCH --time=08:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_13_16_wg1fn5uq/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_13_16_wg1fn5uq/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_13_16_wg1fn5uq/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_13_16_wg1fn5uq/run_metadata.json5', n_gpus=4)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/04/01/fork_6uj4t76i_simple_penalty_div5_13_16_wg1fn5uq
