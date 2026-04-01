#!/bin/bash
#SBATCH --job-name=multiphase_regular_qwen3_4bi_a4i2hyjw
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=80
#SBATCH --mem=75G
#SBATCH --time=10:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/03/23/cont_multiphase_regular_18_35_a4i2hyjw/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/03/23/cont_multiphase_regular_18_35_a4i2hyjw/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/03/23/cont_multiphase_regular_18_35_a4i2hyjw/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
eval "$(conda shell.bash hook)"
conda activate hope

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/03/23/cont_multiphase_regular_18_35_a4i2hyjw/run_metadata.json5', n_gpus=2)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/03/23/cont_multiphase_regular_18_35_a4i2hyjw
