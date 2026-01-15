#!/bin/bash
#SBATCH --job-name=Selection_LLaVA_Training
#SBATCH --output=Selection_LLaVA_Training_output.txt
#SBATCH --error=Selection_LLaVA_Training_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS Selection Training - Iterative Training Loop
# Trains on selected samples and stops at next selection step
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

# Checkpoint/Results directory - SET THIS
BASE_PATH=""                     # Directory containing step_list.txt and checkpoints

# Output paths - SET THESE
RUN_NAME="progress_selection"    # W&B run name

# Training configuration
NUM_GPUS=4
DEEPSPEED_CONFIG="${PROJECT_ROOT}/2_PROGRESS_Training/scripts/zero3.json"
FINAL_STEP=1039                  # Final training step
SELECTION_NUM=7500               # Number of samples selected per iteration

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
echo "PROGRESS Selection Training"
echo "Previous step: ${previous_selection_step}"
echo "Current step: ${cur_selection_step}"
echo "Next step: ${next_selection_step}"
echo "========================================"

# Data path for training
DATA_PATH="${BASE_PATH}/selection_step_${cur_selection_step}/fastest_acc_data_${SELECTION_NUM}.json"

# ============================================================
# Run Training
# ============================================================

deepspeed --num_gpus=${NUM_GPUS} ${PROJECT_ROOT}/2_PROGRESS_Training/llava/train/train_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
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
    --max_steps ${FINAL_STEP} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
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
    --run_name ${RUN_NAME} \
    --output_dir ${BASE_PATH} \
    --intermediate_stop_step ${next_selection_step}

# ============================================================
# Save checkpoint
# ============================================================

echo "Saving checkpoint at step ${next_selection_step}..."
CHECKPOINT_DIR="${BASE_PATH}_${next_selection_step}"
mkdir -p ${CHECKPOINT_DIR}

file_list=("README.md" "adapter_config.json" "adapter_model.safetensors" "config.json" "non_lora_trainables.bin" "trainer_state.json")
for file in "${file_list[@]}"; do
    if [ -f "${BASE_PATH}/${file}" ]; then
        cp -r "${BASE_PATH}/${file}" "${CHECKPOINT_DIR}/${file}"
    fi
done

echo "========================================"
echo "Training complete!"
echo "Checkpoint saved to: ${CHECKPOINT_DIR}"
echo "========================================"