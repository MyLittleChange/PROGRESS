#!/bin/bash
#SBATCH --job-name=warmup_pipeline
#SBATCH --output=warmup_pipeline_output.txt
#SBATCH --error=warmup_pipeline_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS Warm-up Pipeline
# Complete pipeline: Training -> Accuracy Prediction -> Data Selection
# Two-stage warm-up: 50K samples -> 60K samples
# Intermediate stops at step 391 (after 50K) and 469 (after 60K)
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory
UTILS_DIR="${PROJECT_ROOT}/utils"

# Model paths - SET THESE
MODEL_PATH=""                    # Path to base model (e.g., vicuna-7b-v1.5)
MODEL_BASE="${MODEL_PATH}"       # Same as MODEL_PATH for accuracy prediction
PRETRAIN_MM_ADAPTER=""           # Path to pretrained MM projector (.bin file)
VISION_TOWER="openai/clip-vit-large-patch14-336"

# Data paths - SET THESE
IMAGE_FOLDER=""                  # Path to image folder
WARMUP_DATA_50K=""               # Path to Top_50K.json (first warm-up data)
WARMUP_DATA_50K_TO_60K=""        # Path to Top_50K_to_60K.json (second warm-up data)
WARMUP_DATA_60K=""               # Path to Top_60K.json (combined warm-up data for accuracy prediction)
REMAINING_DATA=""                # Path to Remaining_605K.json (leftover data after warm-up)

# Output paths - SET THESE
OUTPUT_DIR=""                    # Directory to save checkpoints and results
DATA_DIR=""                      # Directory to save generated data JSON files
RUN_NAME="progress_warmup"       # W&B run name

# Training configuration
NUM_GPUS=4
DEEPSPEED_CONFIG="${PROJECT_ROOT}/2_PROGRESS_Training/scripts/zero3.json"

# Intermediate stop steps for warm-up stages
STOP_STEP_50K=391                # Stop after 50K samples training
STOP_STEP_60K=469                # Stop after additional 10K samples (60K total)

# Data selection parameters
SELECTION_NUM=7500               # Number of samples to select per iteration
TEMPERATURE=0.5                  # Softmax temperature for selection
SAMPLES_PER_CLUSTER=80           # Samples per cluster for representative selection

# File list for checkpoint saving
file_list=("README.md" "adapter_config.json" "adapter_model.safetensors" "config.json" "non_lora_trainables.bin" "trainer_state.json")

# ============================================================
# Initialize step_list.txt with warm-up steps
# ============================================================
echo "Initializing step_list.txt..."
mkdir -p ${OUTPUT_DIR}
echo "${STOP_STEP_50K}" > ${OUTPUT_DIR}/step_list.txt
echo "${STOP_STEP_60K}" >> ${OUTPUT_DIR}/step_list.txt
echo "Created step_list.txt with steps: ${STOP_STEP_50K}, ${STOP_STEP_60K}"

# ============================================================
# STAGE 1: Train on Top 50K samples (warm-up stage 1)
# ============================================================
echo "========================================"
echo "STAGE 1: Training on Top 50K samples"
echo "Will stop at step ${STOP_STEP_50K}"
echo "========================================"

deepspeed --num_gpus=${NUM_GPUS} ${PROJECT_ROOT}/2_PROGRESS_Training/llava/train/train_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${WARMUP_DATA_50K} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --max_steps ${STOP_STEP_60K} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 50 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --fp16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${RUN_NAME}_stage1 \
    --output_dir ${OUTPUT_DIR} \
    --intermediate_stop_step ${STOP_STEP_50K}

# Save checkpoint after Stage 1
echo "Saving Stage 1 checkpoint..."
CHECKPOINT_50K="${OUTPUT_DIR}_step_${STOP_STEP_50K}"
mkdir -p ${CHECKPOINT_50K}
for file in "${file_list[@]}"; do
    if [ -f "${OUTPUT_DIR}/${file}" ]; then
        cp -r "${OUTPUT_DIR}/${file}" "${CHECKPOINT_50K}/${file}"
    fi
done
echo "Stage 1 checkpoint saved to: ${CHECKPOINT_50K}"

# ============================================================
# STAGE 1.5: Predict Accuracy at Step 391
# ============================================================
echo "========================================"
echo "STAGE 1.5: Predicting accuracy at step ${STOP_STEP_50K}"
echo "========================================"

mkdir -p ${OUTPUT_DIR}/selection_step_${STOP_STEP_50K}
acc_output_391=${OUTPUT_DIR}/selection_step_${STOP_STEP_50K}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running accuracy prediction on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/2_PROGRESS_Training/train/accuracy_predict.py \
    --data_path ${WARMUP_DATA_50K} \
    --model_name_or_path ${CHECKPOINT_50K} \
    --model-base ${MODEL_BASE} \
    --acc_output_file ${acc_output_391}/${CHUNKS}_${IDX}.json \
    --perplexity_batch_size 20 \
    --chunk_idx $IDX \
    --first_question_only True \
    --num_chunks $CHUNKS &
done
wait

# Merge accuracy results
echo "Merging accuracy results..."
rm -f ${acc_output_391}.json
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${acc_output_391}/${CHUNKS}_${IDX}.json >> ${acc_output_391}.json
done

# ============================================================
# STAGE 1.6: Check with InternVL at Step 391
# ============================================================
echo "========================================"
echo "STAGE 1.6: Checking with InternVL at step ${STOP_STEP_50K}"
echo "========================================"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running InternVL check on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/2_PROGRESS_Training/train/check_internVL.py \
    --initial_path ${acc_output_391} \
    --file_name ${CHUNKS}_${IDX}.json \
    --output_file ${CHUNKS}_${IDX}_internVL.json \
    --bs 128 &
done
wait

# Combine InternVL results
echo "Combining InternVL results..."
python ${PROJECT_ROOT}/2_PROGRESS_Training/utils/combine_results.py \
    --input_path ${acc_output_391} \
    --is_internVL True

# ============================================================
# STAGE 2: Continue training on Top_50K_to_60K samples
# ============================================================
echo "========================================"
echo "STAGE 2: Training on Top_50K_to_60K samples"
echo "Will stop at step ${STOP_STEP_60K}"
echo "========================================"

deepspeed --num_gpus=${NUM_GPUS} ${PROJECT_ROOT}/2_PROGRESS_Training/llava/train/train_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${WARMUP_DATA_50K_TO_60K} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --max_steps ${STOP_STEP_60K} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 50 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --fp16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${RUN_NAME}_stage2 \
    --output_dir ${OUTPUT_DIR} \
    --intermediate_stop_step ${STOP_STEP_60K}

# Save checkpoint after Stage 2
echo "Saving Stage 2 checkpoint..."
CHECKPOINT_60K="${OUTPUT_DIR}_step_${STOP_STEP_60K}"
mkdir -p ${CHECKPOINT_60K}
for file in "${file_list[@]}"; do
    if [ -f "${OUTPUT_DIR}/${file}" ]; then
        cp -r "${OUTPUT_DIR}/${file}" "${CHECKPOINT_60K}/${file}"
    fi
done
echo "Stage 2 checkpoint saved to: ${CHECKPOINT_60K}"

# ============================================================
# STAGE 2.5: Predict Accuracy at Step 469
# ============================================================
echo "========================================"
echo "STAGE 2.5: Predicting accuracy at step ${STOP_STEP_60K}"
echo "========================================"

mkdir -p ${OUTPUT_DIR}/selection_step_${STOP_STEP_60K}
acc_output_469=${OUTPUT_DIR}/selection_step_${STOP_STEP_60K}

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running accuracy prediction on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/2_PROGRESS_Training/train/accuracy_predict.py \
    --data_path ${WARMUP_DATA_60K} \
    --model_name_or_path ${CHECKPOINT_60K} \
    --model-base ${MODEL_BASE} \
    --acc_output_file ${acc_output_469}/${CHUNKS}_${IDX}.json \
    --perplexity_batch_size 20 \
    --chunk_idx $IDX \
    --first_question_only True \
    --num_chunks $CHUNKS &
done
wait

# Merge accuracy results
echo "Merging accuracy results..."
rm -f ${acc_output_469}.json
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${acc_output_469}/${CHUNKS}_${IDX}.json >> ${acc_output_469}.json
done

# ============================================================
# STAGE 2.6: Check with InternVL at Step 469
# ============================================================
echo "========================================"
echo "STAGE 2.6: Checking with InternVL at step ${STOP_STEP_60K}"
echo "========================================"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Running InternVL check on GPU: ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ${PROJECT_ROOT}/2_PROGRESS_Training/train/check_internVL.py \
    --initial_path ${acc_output_469} \
    --file_name ${CHUNKS}_${IDX}.json \
    --output_file ${CHUNKS}_${IDX}_internVL.json \
    --bs 128 &
done
wait

# Combine InternVL results
echo "Combining InternVL results..."
python ${PROJECT_ROOT}/2_PROGRESS_Training/utils/combine_results.py \
    --input_path ${acc_output_469} \
    --is_internVL True

# ============================================================
# STAGE 3: Data Selection After Warm-up
# ============================================================
echo "========================================"
echo "STAGE 3: Data Selection After Warm-up"
echo "Selecting ${SELECTION_NUM} samples based on accuracy change"
echo "========================================"

# Define file paths for data selection
ACC_FILE="${acc_output_469}/combined_results_internVL.json"
PREV_ACC_FILE="${acc_output_391}/combined_results_internVL.json"
ACC_CHANGE_FILE="${acc_output_469}/acc_change_overall.json"
SELECTED_DATA_FILE="${acc_output_469}/fastest_acc_data_${SELECTION_NUM}.json"
SELECTED_CLUSTER_ACC="${acc_output_469}/fastest_selected_cluster_accuracies_${SELECTION_NUM}.json"

LEFTOVER_CUR="${DATA_DIR}/Remaining_after_${STOP_STEP_60K}.json"
SELECTED_CUR="${DATA_DIR}/Selected_until_${STOP_STEP_60K}.json"
SELECTED_CUR_REPR="${DATA_DIR}/Selected_until_${STOP_STEP_60K}_repr_${SAMPLES_PER_CLUSTER}.json"

# Step 3.1: Compare accuracy with previous step
echo "Step 3.1: Comparing accuracy with previous step..."
python ${UTILS_DIR}/acc_sort_qian.py \
    --acc_file ${ACC_FILE} \
    --previous_acc_file ${PREV_ACC_FILE} \
    --output_average_acc_change ${ACC_CHANGE_FILE} \
    --compare_with_previous

# Step 3.2: Select samples based on accuracy change
echo "Step 3.2: Selecting ${SELECTION_NUM} samples based on accuracy change..."
python ${UTILS_DIR}/acc_sort_qian.py \
    --acc_file ${ACC_FILE} \
    --output_average_acc_change ${ACC_CHANGE_FILE} \
    --output_file ${SELECTED_DATA_FILE} \
    --output_selected_cluster_accuracies ${SELECTED_CLUSTER_ACC} \
    --selection_file ${REMAINING_DATA} \
    --selection_num ${SELECTION_NUM} \
    --selection_relative_change \
    --softmax_selection \
    --temperature ${TEMPERATURE} \
    --step_list_file ${OUTPUT_DIR}/step_list.txt

# Step 3.3: Update leftover data (remove selected samples)
echo "Step 3.3: Updating leftover data..."
python ${UTILS_DIR}/concat_jsons.py \
    --file1 ${REMAINING_DATA} \
    --file2 ${SELECTED_DATA_FILE} \
    --output ${LEFTOVER_CUR} \
    --remove_samples

# Step 3.4: Accumulate selected samples (60K + newly selected)
echo "Step 3.4: Accumulating selected samples..."
python ${UTILS_DIR}/concat_jsons.py \
    --file1 ${WARMUP_DATA_60K} \
    --file2 ${SELECTED_DATA_FILE} \
    --output ${SELECTED_CUR} \
    --concatenate

# Step 3.5: Create representative selection
echo "Step 3.5: Creating representative selection (${SAMPLES_PER_CLUSTER} per cluster)..."
python ${UTILS_DIR}/concat_jsons.py \
    --file1 ${SELECTED_CUR} \
    --output ${SELECTED_CUR_REPR} \
    --samples_per_cluster ${SAMPLES_PER_CLUSTER}

# ============================================================
# Pipeline Complete
# ============================================================
echo "========================================"
echo "PROGRESS Warm-up Pipeline Complete!"
echo "========================================"
echo "Checkpoints:"
echo "  - Stage 1 (50K): ${CHECKPOINT_50K}"
echo "  - Stage 2 (60K): ${CHECKPOINT_60K}"
echo ""
echo "Accuracy Results:"
echo "  - Step 391: ${acc_output_391}/combined_results_internVL.json"
echo "  - Step 469: ${acc_output_469}/combined_results_internVL.json"
echo ""
echo "Data Selection:"
echo "  - Selected samples: ${SELECTED_DATA_FILE}"
echo "  - Updated leftover: ${LEFTOVER_CUR}"
echo "  - Accumulated selected: ${SELECTED_CUR}"
echo "  - Representative selection: ${SELECTED_CUR_REPR}"
echo "========================================"
