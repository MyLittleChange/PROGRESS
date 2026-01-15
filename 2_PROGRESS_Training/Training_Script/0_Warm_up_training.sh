#!/bin/bash
#SBATCH --job-name=warmup_training
#SBATCH --output=warmup_training_output.txt
#SBATCH --error=warmup_training_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS Warm-up Training Script
# Two-stage warm-up: 50K samples -> 60K samples
# Intermediate stops at step 391 (after 50K) and 469 (after 60K)
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory

# Model paths - SET THESE
MODEL_PATH=""                    # Path to base model (e.g., vicuna-7b-v1.5)
PRETRAIN_MM_ADAPTER=""           # Path to pretrained MM projector (.bin file)
VISION_TOWER="openai/clip-vit-large-patch14-336"

# Data paths - SET THESE
IMAGE_FOLDER=""                  # Path to image folder
WARMUP_DATA_50K=""               # Path to Top_50K.json (first warm-up data)
WARMUP_DATA_50K_TO_60K=""        # Path to Top_50K_to_60K.json (second warm-up data)

# Output paths - SET THESE
OUTPUT_DIR=""                    # Directory to save checkpoints
RUN_NAME="progress_warmup"       # W&B run name

# Training configuration
NUM_GPUS=4
DEEPSPEED_CONFIG="${PROJECT_ROOT}/2_PROGRESS_Training/scripts/zero3.json"

# Intermediate stop steps for warm-up stages
STOP_STEP_50K=391                # Stop after 50K samples training
STOP_STEP_60K=469                # Stop after additional 10K samples (60K total)

# ============================================================
# Stage 1: Train on Top 50K samples (warm-up stage 1)
# ============================================================
echo "========================================"
echo "Stage 1: Training on Top 50K samples"
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

file_list=("README.md" "adapter_config.json" "adapter_model.safetensors" "config.json" "non_lora_trainables.bin" "trainer_state.json")
for file in "${file_list[@]}"; do
    if [ -f "${OUTPUT_DIR}/${file}" ]; then
        cp -r "${OUTPUT_DIR}/${file}" "${CHECKPOINT_50K}/${file}"
    fi
done

echo "Stage 1 checkpoint saved to: ${CHECKPOINT_50K}"

# ============================================================
# Stage 2: Continue training on Top_50K_to_60K samples
# ============================================================
echo "========================================"
echo "Stage 2: Training on Top_50K_to_60K samples"
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

echo "========================================"
echo "Warm-up training complete!"
echo "Checkpoints saved:"
echo "  - Stage 1 (50K): ${CHECKPOINT_50K}"
echo "  - Stage 2 (60K): ${CHECKPOINT_60K}"
echo "========================================"