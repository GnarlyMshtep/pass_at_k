#!/bin/bash
#SBATCH --job-name=fork_k16vo4tp_q80_monitor_bzf5yptd
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=755G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/05/02/cont_fork_k16vo4tp_q80_monitor_04_19_bzf5yptd/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/05/02/cont_fork_k16vo4tp_q80_monitor_04_19_bzf5yptd/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu
#SBATCH --nodelist=better-ginkgo-dragonfly

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/05/02/cont_fork_k16vo4tp_q80_monitor_04_19_bzf5yptd/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
source /shared/matan/code/pass_at_k/.venv/bin/activate

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/05/02/cont_fork_k16vo4tp_q80_monitor_04_19_bzf5yptd/run_metadata.json5', n_gpus=4)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/05/02/cont_fork_k16vo4tp_q80_monitor_04_19_bzf5yptd
