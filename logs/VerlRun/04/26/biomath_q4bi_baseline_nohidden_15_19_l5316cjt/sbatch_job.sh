#!/bin/bash
#SBATCH --job-name=biomath_q4bi_baseline_nohidden_l5316cjt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --mem=755G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/matan/code/pass_at_k/logs/VerlRun/04/26/biomath_q4bi_baseline_nohidden_15_19_l5316cjt/daemon_logs/sbatch/run.out
#SBATCH --error=/shared/matan/code/pass_at_k/logs/VerlRun/04/26/biomath_q4bi_baseline_nohidden_15_19_l5316cjt/daemon_logs/sbatch/run.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mshtepel@andrew.cmu.edu
#SBATCH --nodelist=better-ginkgo-dragonfly

# --- Environment (captured at sbatch creation time) ---
source /shared/matan/code/pass_at_k/logs/VerlRun/04/26/biomath_q4bi_baseline_nohidden_15_19_l5316cjt/daemon_logs/sbatch/env.sh

cd /shared/matan/code/pass_at_k
source /shared/matan/code/pass_at_k/.venv/bin/activate

# --- Kill any leftover Ray on this node ---
ray stop --force 2>/dev/null || true
pkill -u matan -f "ray::" 2>/dev/null || true
pkill -u matan -f raylet 2>/dev/null || true
pkill -u matan -f gcs_server 2>/dev/null || true
sleep 2
echo "Cleaned up stale Ray processes"

# --- Force isolated local Ray cluster ---
export RAY_ADDRESS=local
export RAY_TMPDIR=/tmp/ray_matan

# --- Copy model to local disk for faster loading ---
LOCAL_MODEL_DIR=/tmp/models/Qwen3-4B-I
if [ ! -f "$LOCAL_MODEL_DIR/config.json" ]; then
    echo "Copying model to local disk..."
    mkdir -p /tmp/models
    cp -r /shared/matan/models/Qwen3-4B-I "$LOCAL_MODEL_DIR"
    echo "Model copied to $LOCAL_MODEL_DIR"
else
    echo "Model already on local disk: $LOCAL_MODEL_DIR"
fi

# --- Patch model path in run_metadata to use local copy ---
sed -i 's|actor_rollout_ref.model.path=/shared/matan/models/Qwen3-4B-I|actor_rollout_ref.model.path=/tmp/models/Qwen3-4B-I|g' \
    /shared/matan/code/pass_at_k/logs/VerlRun/04/26/biomath_q4bi_baseline_nohidden_15_19_l5316cjt/run_metadata.json5

# Register with run tracker at actual SLURM launch time
python3 -c "from vfh.run_tracker import register_run_from_metadata; register_run_from_metadata(metadata_path='/shared/matan/code/pass_at_k/logs/VerlRun/04/26/biomath_q4bi_baseline_nohidden_15_19_l5316cjt/run_metadata.json5', n_gpus=4)" || echo "WARNING: run tracker registration failed"

python -m vfh.orchestrator run-prepared \
    --run-dir /shared/matan/code/pass_at_k/logs/VerlRun/04/26/biomath_q4bi_baseline_nohidden_15_19_l5316cjt
