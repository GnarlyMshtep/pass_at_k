#!/bin/bash

# Array of diverse initial strategy probability configurations
declare -a configs=(
    "0.9 0.03 0.03 0.024 0.02"
    "0.5 0.3 0.1 0.05 0.05"
    "0.4 0.3 0.1 0.1 0.1"
    "0.2 0.2 0.2 0.2 0.2"
    "0.6 0.2 0.1 0.05 0.05"
    "0.6 0.1 0.1 0.1 0.1"
    "0.7 0.15 0.1 0.03 0.02"
    "0.3 0.3 0.3 0.05 0.05"
)

# Function to generate all unique permutations using Python
generate_permutations() {
    local -a arr=("$@")
    # Convert array to comma-separated string for Python
    local values_str=$(printf "%s," "${arr[@]}")
    values_str=${values_str%,}  # Remove trailing comma
    
    python3 -c "
import itertools

values = [${values_str}]
# Generate all permutations
perms = list(itertools.permutations(values))
# Remove duplicates and sort
unique_perms = sorted(list(set(perms)))
# Print each permutation on a separate line
for perm in unique_perms:
    print(' '.join(map(str, perm)))
"
}

# Collect all permutations from all configs
echo "Collecting all permutations from all configs..."
all_permutations_file=$(mktemp)

for config_idx in "${!configs[@]}"; do
    echo "Processing config $((config_idx + 1))/${#configs[@]}: ${configs[config_idx]}"
    
    # Parse the config into an array
    read -a values <<< "${configs[config_idx]}"
    
    # Generate all unique permutations for this config and append to master file
    generate_permutations "${values[@]}" >> "$all_permutations_file"
done

# Remove duplicates and sort
echo "Removing duplicates and sorting all permutations..."
unique_permutations_file=$(mktemp)
sort -u "$all_permutations_file" > "$unique_permutations_file"

# Count total unique permutations
total_permutations=$(wc -l < "$unique_permutations_file")
echo "Found $total_permutations unique permutations across all configs"

# Read all unique permutations into array
all_permutations=()
while IFS= read -r line; do
    all_permutations+=("$line")
done < "$unique_permutations_file"

# Clean up temp files
rm "$all_permutations_file" "$unique_permutations_file"

# Run permutations in batches of 25
batch_size=25
total_batches=$(( (total_permutations + batch_size - 1) / batch_size ))

echo "Running $total_permutations permutations in $total_batches batches of $batch_size..."

for batch_idx in $(seq 0 $((total_batches - 1))); do
    start_idx=$((batch_idx * batch_size))
    end_idx=$((start_idx + batch_size - 1))
    
    # Don't exceed array bounds
    if [ $end_idx -ge $total_permutations ]; then
        end_idx=$((total_permutations - 1))
    fi
    
    echo "Starting batch $((batch_idx + 1))/$total_batches (permutations $((start_idx + 1))-$((end_idx + 1)))"
    
    # Run permutations in this batch in parallel
    for i in $(seq $start_idx $end_idx); do
        echo "  Starting permutation $((i + 1)): ${all_permutations[i]}"
         ./door_learning_simulation --k 5 --iterations 100  --simulations 10000 --a ${all_permutations[i]} &
    done
    
    # Wait for all processes in this batch to complete
    echo "  Waiting for batch $((batch_idx + 1)) to complete..."
    wait
    
    echo "  Batch $((batch_idx + 1)) completed!"
    echo ""
done

echo "All simulations completed!"
