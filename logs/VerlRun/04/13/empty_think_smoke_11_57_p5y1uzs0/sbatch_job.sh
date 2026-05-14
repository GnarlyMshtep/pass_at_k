#!/bin/bash
#SBATCH --job-name=qwen3_8b_empty_think_dummy_p5y1uzs0
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=80
#SBATCH --mem=755G
#SBATCH --time=00:30:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/04/13/empty_think_smoke_11_57_p5y1uzs0/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/04/13/empty_think_smoke_11_57_p5y1uzs0/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/04/13/empty_think_smoke_11_57_p5y1uzs0/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
source /shared/matan/code/pass_at_k/.venv/bin/activate

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/04/13/empty_think_smoke_11_57_p5y1uzs0/run_metadata.json5', n_gpus=2)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/04/13/empty_think_smoke_11_57_p5y1uzs0
