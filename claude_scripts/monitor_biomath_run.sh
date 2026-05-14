#!/bin/bash
# Monitor BioMath run: reward stragglers + memory usage
# Usage: monitor_biomath_run.sh <run_dir> <job_id>

RUN_DIR="${1:?Usage: monitor_biomath_run.sh <run_dir> <job_id>}"
JOB_ID="${2:?Usage: monitor_biomath_run.sh <run_dir> <job_id>}"
LOG="$RUN_DIR/verl_output.log"

echo "=== $(date '+%H:%M:%S') | Job $JOB_ID ==="

# Job status
JOB_STATE=$(squeue -j "$JOB_ID" -o "%T" --noheader 2>/dev/null)
if [ -z "$JOB_STATE" ]; then
    echo "JOB GONE (completed or cancelled)"
    exit 1
fi
echo "Job state: $JOB_STATE"

# Log progress
if [ -f "$LOG" ]; then
    LINE_COUNT=$(wc -l < "$LOG")
    echo "Log lines: $LINE_COUNT"

    # Latest training step
    LATEST_STEP=$(grep -o "Training Progress:.*" "$LOG" | tail -1)
    echo "Progress: ${LATEST_STEP:-no training steps yet}"

    # Straggler stats (last 5 reward chunks)
    echo "--- Reward chunks (last 5) ---"
    grep "chunk.*took" "$LOG" | tail -5

    # Any exceptions?
    TOTAL_EXCEPTIONS=$(grep "exceptions=" "$LOG" | grep -oP 'exceptions=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
    TOTAL_STRAGGLERS=$(grep "stragglers=" "$LOG" | grep -oP 'stragglers=\K[0-9]+' | awk '{s+=$1}END{print s+0}')
    echo "Total stragglers: $TOTAL_STRAGGLERS, Total exceptions: $TOTAL_EXCEPTIONS"
else
    echo "No verl_output.log yet"
fi

# Memory usage on node
NODE=$(squeue -j "$JOB_ID" -o "%N" --noheader 2>/dev/null)
if [ -n "$NODE" ]; then
    echo "--- Memory on $NODE ---"
    srun --jobid="$JOB_ID" --overlap bash -c "free -g | head -2 && echo '---' && ps aux --sort=-rss | grep matan | grep -v grep | head -5 | awk '{printf \"%-8s %6.1fGB  %s\n\", \$1, \$6/1024/1024, \$11}'" 2>/dev/null
fi

echo "=== end ==="
