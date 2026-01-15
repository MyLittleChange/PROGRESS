#!/bin/bash
#SBATCH --job-name=predict_accuracy_469
#SBATCH --output=predict_accuracy_469_output.txt
#SBATCH --error=predict_accuracy_469_error.txt
#SBATCH --ntasks=1

# ============================================================
# Predict Accuracy at Step 469 (after 60K warm-up)
# Evaluates model accuracy on trained samples
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory

# Model paths - SET THESE
MODEL_BASE=""                    # Path to base model (e.g., vicuna-7b-v1.5)
CHECKPOINT_DIR=""                # Directory containing the checkpoint at step 469

# Data paths - SET THESE
DATA_PATH=""                     # Path to Top_60K.json (data used for training)

# Output paths - SET THESE
OUTPUT_DIR=""                    # Directory to save accuracy prediction results

# Step configuration
SELECTION_STEP=469

# ============================================================
# Run accuracy prediction (multi-GPU parallel)
# ============================================================

# Create output directory
mkdir -p ${OUTPUT_DIR}/selection_step_${SELECTION_STEP}

acc_output_file=${OUTPUT_DIR}/selection_step_${SELECTION_STEP}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "========================================"
echo "Predicting accuracy at step ${SELECTION_STEP}"
echo "Using ${CHUNKS} GPUs"
echo "========================================"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/2_PROGRESS_Training/train/accuracy_predict.py \
    --data_path ${DATA_PATH} \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --model-base ${MODEL_BASE} \
    --acc_output_file ${acc_output_file}/${CHUNKS}_${IDX}.json \
    --perplexity_batch_size 20 \
    --chunk_idx $IDX \
    --first_question_only True \
    --num_chunks $CHUNKS &
done

wait

# Merge the accuracy output files
echo "Merging accuracy results..."
save_file=${acc_output_file}.json
rm -f ${save_file}  # Remove if exists
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${acc_output_file}/${CHUNKS}_${IDX}.json >> ${save_file}
done

echo "========================================"
echo "Accuracy prediction complete!"
echo "Results saved to: ${save_file}"
echo "========================================"
