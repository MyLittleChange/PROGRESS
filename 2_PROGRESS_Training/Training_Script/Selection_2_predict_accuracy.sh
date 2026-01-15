#!/bin/bash
#SBATCH --job-name=Selection_predict_accuracy
#SBATCH --output=Selection_predict_accuracy_output.txt
#SBATCH --error=Selection_predict_accuracy_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS Selection Accuracy Prediction
# Predicts accuracy on representative samples using trained model
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
SAMPLES_PER_CLUSTER=80           # Samples per cluster for representative selection

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
echo "PROGRESS Selection Accuracy Prediction"
echo "Previous step: ${previous_selection_step}"
echo "Current step: ${cur_selection_step}"
echo "Next step: ${next_selection_step}"
echo "========================================"

# Data path for prediction (representative samples)
DATA_PATH="${DATA_DIR}/${SELECTED_PREFIX}_${cur_selection_step}_${SAMPLES_PER_CLUSTER}.json"

# Output paths
CHECKPOINT_DIR="${BASE_PATH}_${next_selection_step}"
MERGED_MODEL_PATH="${CHECKPOINT_DIR}_merged"
ACC_OUTPUT_DIR="${CHECKPOINT_DIR}/selection_step_${next_selection_step}"

# Create output directory
mkdir -p ${ACC_OUTPUT_DIR}

# ============================================================
# Merge LoRA weights if needed
# ============================================================

if [ ! -f "${MERGED_MODEL_PATH}/model-00003-of-00003.safetensors" ]; then
    echo "Merging LoRA weights..."
    CUDA_VISIBLE_DEVICES=0 python ${PROJECT_ROOT}/2_PROGRESS_Training/scripts/merge_lora_weights.py \
        --model-path ${CHECKPOINT_DIR} \
        --model-base ${MODEL_BASE} \
        --save-model-path ${MERGED_MODEL_PATH}
fi

# ============================================================
# Run accuracy prediction
# ============================================================

echo "Running accuracy prediction..."
python ${PROJECT_ROOT}/utils/accuracy_predict_lmdeploy.py \
    --data_path ${DATA_PATH} \
    --model_name_or_path ${MERGED_MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --acc_output_file ${ACC_OUTPUT_DIR}/1_0.json \
    --perplexity_batch_size 180 \
    --chunk_idx 0 \
    --num_chunks 1

echo "========================================"
echo "Accuracy prediction complete!"
echo "Results saved to: ${ACC_OUTPUT_DIR}/1_0.json"
echo "========================================"

