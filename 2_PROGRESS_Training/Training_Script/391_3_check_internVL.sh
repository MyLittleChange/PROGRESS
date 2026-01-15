#!/bin/bash
#SBATCH --job-name=check_internVL_391
#SBATCH --output=check_internVL_391_output.txt
#SBATCH --error=check_internVL_391_error.txt
#SBATCH --ntasks=1

# ============================================================
# Check with InternVL at Step 391 (after 50K warm-up)
# Validates accuracy predictions using InternVL
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory

# Input/Output paths - SET THESE
ACC_OUTPUT_DIR=""                # Directory containing accuracy prediction results (from 391_2_predict_accuracy.sh)

# Step configuration
SELECTION_STEP=391

# ============================================================
# Run InternVL check (multi-GPU parallel)
# ============================================================

acc_output_file=${ACC_OUTPUT_DIR}/selection_step_${SELECTION_STEP}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "========================================"
echo "Checking with InternVL at step ${SELECTION_STEP}"
echo "Using ${CHUNKS} GPUs"
echo "========================================"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/2_PROGRESS_Training/train/check_internVL.py \
    --initial_path ${acc_output_file} \
    --file_name ${CHUNKS}_${IDX}.json \
    --output_file ${CHUNKS}_${IDX}_internVL.json \
    --bs 128 &
done

wait

# Combine results
echo "Combining InternVL results..."
python ${PROJECT_ROOT}/2_PROGRESS_Training/utils/combine_results.py \
    --input_path ${acc_output_file} \
    --is_internVL True

echo "========================================"
echo "InternVL check complete!"
echo "Results saved to: ${acc_output_file}"
echo "========================================"