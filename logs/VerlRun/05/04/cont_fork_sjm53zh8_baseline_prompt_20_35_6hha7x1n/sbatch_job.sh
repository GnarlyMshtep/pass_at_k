#!/bin/bash
#SBATCH --job-name=fork_sjm53zh8_baseline_prompt_6hha7x1n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=377G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont_fork_sjm53zh8_baseline_prompt_20_35_6hha7x1n/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont_fork_sjm53zh8_baseline_prompt_20_35_6hha7x1n/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu
#SBATCH --nodelist=bleak-mushroom-dove

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont_fork_sjm53zh8_baseline_prompt_20_35_6hha7x1n/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
source /shared/matan/code/pass_at_k/.venv/bin/activate

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont_fork_sjm53zh8_baseline_prompt_20_35_6hha7x1n/run_metadata.json5', n_gpus=4)" || echo "WARNING: run tracker registration failed"

taskset -c 0-79 python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/05/04/cont_fork_sjm53zh8_baseline_prompt_20_35_6hha7x1n
