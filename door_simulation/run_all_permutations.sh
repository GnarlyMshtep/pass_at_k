#!/bin/bash

# Generate all 32 permutations of the 5 strategy probabilities
# Each permutation represents a different ordering of the 5 values: 0.9, 0.02, 0.02, 0.03, 0.03

# Array to store all permutations
declare -a permutations=()

# Generate all permutations using a recursive approach
generate_permutations() {
    local -a arr=("$@")
    local -i n=${#arr[@]}
    
    if [ $n -eq 1 ]; then
        # Base case: single element
        permutations+=("${arr[0]}")
    else
        # Recursive case: for each element, generate permutations of the rest
        for ((i=0; i<n; i++)); do
            local -a remaining=()
            for ((j=0; j<n; j++)); do
                if [ $j -ne $i ]; then
                    remaining+=("${arr[j]}")
                fi
            done
            
            # Get permutations of remaining elements
            local -a sub_permutations=()
            generate_permutations "${remaining[@]}"
            
            # Add current element to each sub-permutation
            local -i start_idx=${#permutations[@]}
            local -i sub_count=${#remaining[@]}
            if [ $sub_count -eq 0 ]; then
                sub_count=1
            fi
            
            # Calculate how many permutations we just added
            local -i added_count=0
            for ((k=start_idx; k<${#permutations[@]}; k++)); do
                added_count=$((added_count + 1))
            done
            
            # Prepend current element to each new permutation
            for ((k=start_idx; k<${#permutations[@]}; k++)); do
                permutations[k]="${arr[i]} ${permutations[k]}"
            done
        done
    fi
}

# Define the 5 values to permute
values=(0.9 0.02 0.02 0.03 0.03)

echo "Generating all permutations of: ${values[*]}"
echo "Expected number of permutations: 32 (5! / (2! * 2!) = 120 / 4 = 30, but we'll generate all unique orderings)"

# Generate all permutations
generate_permutations "${values[@]}"

# Remove duplicates and sort
unique_permutations=($(printf '%s\n' "${permutations[@]}" | sort -u))

echo "Generated ${#unique_permutations[@]} unique permutations"

# Display all permutations
echo "All permutations:"
for i in "${!unique_permutations[@]}"; do
    echo "Permutation $((i+1)): ${unique_permutations[i]}"
done

echo ""
echo "Running simulations for all permutations..."

# Run simulations in parallel for all permutations
for i in "${!unique_permutations[@]}"; do
    echo "Starting simulation $((i+1)) with config: ${unique_permutations[i]}"
    python door_learning_simulation.py --a ${unique_permutations[i]} &
done

# Wait for all background processes to complete
wait

echo "All simulations completed!"
