#!/bin/bash
#SBATCH --job-name=Selection_check_internVL
#SBATCH --output=Selection_check_internVL_output.txt
#SBATCH --error=Selection_check_internVL_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS Selection InternVL Check
# Validates accuracy predictions using InternVL
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory

# Checkpoint/Results directory - SET THIS
BASE_PATH=""                     # Directory containing step_list.txt and checkpoints

# ============================================================
# Read step information from step_list.txt
# ============================================================

mapfile -t last_lines < <(tail -n 3 ${BASE_PATH}/step_list.txt)
previous_selection_step=${last_lines[0]}
cur_selection_step=${last_lines[1]}
next_selection_step=${last_lines[2]}
((previous_selection_step = previous_selection_step))
((cur_selection_step = cur_selection_step))
((next_selection_step = next_selection_step))

echo "========================================"
echo "PROGRESS Selection InternVL Check"
echo "Previous step: ${previous_selection_step}"
echo "Current step: ${cur_selection_step}"
echo "Next step: ${next_selection_step}"
echo "========================================"

# Input/Output paths
ACC_OUTPUT_DIR="${BASE_PATH}_${next_selection_step}/selection_step_${next_selection_step}"

# ============================================================
# Run InternVL check
# ============================================================

echo "Running InternVL check..."
python ${PROJECT_ROOT}/utils/check_internVL.py \
    --initial_path ${ACC_OUTPUT_DIR} \
    --file_name 1_0.json \
    --output_file 1_0_internVL.json \
    --bs 300

echo "========================================"
echo "InternVL check complete!"
echo "Results saved to: ${ACC_OUTPUT_DIR}/1_0_internVL.json"
echo "========================================" 