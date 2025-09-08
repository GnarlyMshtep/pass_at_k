#!/bin/bash

# Single-attempt DAPO script that inherits from mult_att_dapo.sh
# Override specific parameters for single-attempt configuration

# Set the parameters that differ from mult_att_dapo
export OVERRIDE_DATASET_NAME="bigmath_digits_singatt"
export OVERRIDE_REWARD_FUNCTION="compute_score_math"

# Source the mult_att_dapo script from the same directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/mult_att_dapo.sh"