#!/bin/bash
#SBATCH --job-name=Selection_predict_conf
#SBATCH --output=Selection_predict_conf_output.txt
#SBATCH --error=Selection_predict_conf_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS Selection Confidence Prediction
# Predicts confidence scores for trained samples
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory

# Model paths - SET THESE
MODEL_BASE=""                    # Path to base model (e.g., vicuna-7b-v1.5)

# Checkpoint/Results directory - SET THIS
BASE_PATH=""                     # Directory containing step_list.txt and checkpoints

# Data paths - SET THESE
DATA_DIR=""                      # Directory containing data JSON files
SELECTED_PREFIX="Selected_until" # Prefix for accumulated selected data files

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
echo "PROGRESS Selection Confidence Prediction"
echo "Previous step: ${previous_selection_step}"
echo "Current step: ${cur_selection_step}"
echo "Next step: ${next_selection_step}"
echo "========================================"

# Data path for prediction
DATA_PATH="${DATA_DIR}/${SELECTED_PREFIX}_${cur_selection_step}.json"

# Output paths
CHECKPOINT_DIR="${BASE_PATH}_${next_selection_step}"
ACC_OUTPUT_DIR="${CHECKPOINT_DIR}/selection_step_${next_selection_step}"

# Create output directory
mkdir -p ${ACC_OUTPUT_DIR}

# ============================================================
# Run confidence prediction (multi-GPU parallel)
# ============================================================

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "Using ${CHUNKS} GPUs for confidence prediction"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/utils/confidence_predict.py \
    --data_path ${DATA_PATH} \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --model-base ${MODEL_BASE} \
    --acc_output_file ${ACC_OUTPUT_DIR}/${CHUNKS}_${IDX}.json \
    --perplexity_batch_size 15 \
    --chunk_idx $IDX \
    --num_chunks $CHUNKS &
done

wait

# Combine results
echo "Combining confidence prediction results..."
python ${PROJECT_ROOT}/utils/combine_results.py \
    --input_path ${ACC_OUTPUT_DIR} \
    --is_confidence True

echo "========================================"
echo "Confidence prediction complete!"
echo "Results saved to: ${ACC_OUTPUT_DIR}"
echo "========================================"
