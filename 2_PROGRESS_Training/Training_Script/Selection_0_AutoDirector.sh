#!/bin/bash
#SBATCH --job-name=AutoDirector
#SBATCH --output=AutoDirector_output.txt
#SBATCH --error=AutoDirector_error.txt
#SBATCH --ntasks=1

# ============================================================
# PROGRESS AutoDirector - Automatic Training Loop Controller
# Orchestrates the iterative training-selection loop:
#   1. Check training status
#   2. Run accuracy prediction if needed
#   3. Run InternVL check if needed
#   4. Run data selection if needed
#   5. Run next training iteration
# ============================================================

# Configuration - SET THESE PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."  # PROGRESS directory
UTILS_DIR="${PROJECT_ROOT}/utils"

# Checkpoint/Results directory - SET THIS
BASE_PATH=""                     # Directory containing step_list.txt and checkpoints

# Data paths - SET THESE
DATA_DIR=""                      # Directory containing data JSON files

# File naming prefixes
LEFTOVER_PREFIX="Remaining"      # Prefix for leftover data files
SELECTED_PREFIX="Selected_until" # Prefix for accumulated selected data files

# Selection parameters
SELECTION_NUM=7500               # Number of samples to select per iteration
TEMPERATURE=0.5                  # Softmax temperature for selection
SAMPLES_PER_CLUSTER=100           # Samples per cluster for representative selection
FINAL_STEP=1039                  # Final training step

# ============================================================
# Main AutoDirector Loop
# ============================================================

while true; do
    echo "========================================"
    echo "AutoDirector: Checking training status..."
    echo "========================================"

    # Read the last three steps from step_list.txt
    num_lines=$(wc -l <${BASE_PATH}/step_list.txt)
    num_lines=$((num_lines + 1))
    echo "Number of steps in step_list: $num_lines"

    if [ "$num_lines" -ge 3 ]; then
        # If there are 3 or more lines, read last 3
        mapfile -t last_lines < <(tail -n 3 ${BASE_PATH}/step_list.txt)
        previous_selection_step=${last_lines[0]}
        cur_selection_step=${last_lines[1]}
        next_selection_step=${last_lines[2]}
        ((previous_selection_step = previous_selection_step))
        echo "Previous step: $previous_selection_step"
    else
        # If there are only 2 lines, read last 2 (first iteration after warm-up)
        mapfile -t last_lines < <(tail -n 2 ${BASE_PATH}/step_list.txt)
        cur_selection_step=${last_lines[0]}
        next_selection_step=${last_lines[1]}
    fi
    ((cur_selection_step = cur_selection_step))
    ((next_selection_step = next_selection_step))
    echo "Current step: $cur_selection_step"
    echo "Next step: $next_selection_step"

    # Check if we've reached the final step
    if [ "$next_selection_step" -ge ${FINAL_STEP} ]; then
        echo "========================================"
        echo "Reached final step (${FINAL_STEP}). Training complete!"
        echo "========================================"
        break
    fi

    # ============================================================
    # Case 1: Selection results exist - update step list
    # ============================================================
    if [ -d "${BASE_PATH}/selection_step_${next_selection_step}" ]; then
        echo "----------------------------------------"
        echo "Case 1: Running data selection to update step list..."
        echo "----------------------------------------"

        # Compare with previous step
        python ${UTILS_DIR}/acc_sort.py \
            --acc_file ${BASE_PATH}/selection_step_${next_selection_step}/combined_results_internVL.json \
            --previous_acc_file ${BASE_PATH}/selection_step_${cur_selection_step}/combined_results_internVL.json \
            --output_average_acc_change ${BASE_PATH}/selection_step_${next_selection_step}/acc_change_overall.json \
            --compare_with_previous

        # Update step list and select samples
        python ${UTILS_DIR}/acc_sort.py \
            --acc_file ${BASE_PATH}/selection_step_${next_selection_step}/combined_results_internVL.json \
            --output_average_acc_change ${BASE_PATH}/selection_step_${next_selection_step}/acc_change_overall.json \
            --output_file ${BASE_PATH}/selection_step_${next_selection_step}/fastest_acc_data_${SELECTION_NUM}.json \
            --output_selected_cluster_accuracies ${BASE_PATH}/selection_step_${next_selection_step}/fastest_selected_cluster_accuracies_${SELECTION_NUM}.json \
            --selection_file ${DATA_DIR}/${LEFTOVER_PREFIX}_after_${cur_selection_step}.json \
            --selection_num ${SELECTION_NUM} \
            --selection_relative_change \
            --softmax_selection \
            --training_batch_size 128 \ 
            --temperature ${TEMPERATURE} \
            --step_list_file ${BASE_PATH}/step_list.txt

    # ============================================================
    # Case 2: Checkpoint exists - check accuracy/InternVL status
    # ============================================================
    elif [ -f "${BASE_PATH}_${next_selection_step}/adapter_model.safetensors" ]; then

        # Check if InternVL results exist
        if [ -f "${BASE_PATH}_${next_selection_step}/selection_step_${next_selection_step}/1_0_internVL.json" ]; then
            echo "----------------------------------------"
            echo "Case 2a: Copying InternVL results to main path..."
            echo "----------------------------------------"
            cp ${BASE_PATH}_${next_selection_step}/selection_step_${next_selection_step}/1_0_internVL.json \
               ${BASE_PATH}_${next_selection_step}/selection_step_${next_selection_step}/combined_results_internVL.json
            cp -r "${BASE_PATH}_${next_selection_step}/selection_step_${next_selection_step}" \
                  "${BASE_PATH}/selection_step_${next_selection_step}"

        # Check if accuracy prediction exists
        elif [ -f "${BASE_PATH}_${next_selection_step}/selection_step_${next_selection_step}/1_0.json" ]; then
            echo "----------------------------------------"
            echo "Case 2b: Running InternVL check..."
            echo "----------------------------------------"
            bash ${SCRIPT_DIR}/Selection_3_check_internVL.sh

        else
            echo "----------------------------------------"
            echo "Case 2c: Running accuracy prediction..."
            echo "----------------------------------------"
            bash ${SCRIPT_DIR}/Selection_2_predict_accuracy.sh
        fi

    # ============================================================
    # Case 3: No checkpoint - prepare data and train
    # ============================================================
    else
        # Check if representative selection exists
        if [ -f "${DATA_DIR}/${SELECTED_PREFIX}_${cur_selection_step}_${SAMPLES_PER_CLUSTER}.json" ]; then
            echo "----------------------------------------"
            echo "Case 3a: Running LLaVA Training..."
            echo "----------------------------------------"
            bash ${SCRIPT_DIR}/Selection_1_LLaVA_Training.sh

        else
            echo "----------------------------------------"
            echo "Case 3b: Running data preparation steps..."
            echo "----------------------------------------"

            # Update leftover data (remove selected samples)
            python ${UTILS_DIR}/concat_jsons.py \
                --file1 ${DATA_DIR}/${LEFTOVER_PREFIX}_after_${previous_selection_step}.json \
                --file2 ${BASE_PATH}/selection_step_${cur_selection_step}/fastest_acc_data_${SELECTION_NUM}.json \
                --output ${DATA_DIR}/${LEFTOVER_PREFIX}_after_${cur_selection_step}.json \
                --remove_samples

            # Accumulate selected samples
            python ${UTILS_DIR}/concat_jsons.py \
                --file1 ${DATA_DIR}/${SELECTED_PREFIX}_${previous_selection_step}.json \
                --file2 ${BASE_PATH}/selection_step_${cur_selection_step}/fastest_acc_data_${SELECTION_NUM}.json \
                --output ${DATA_DIR}/${SELECTED_PREFIX}_${cur_selection_step}.json \
                --concatenate

            # Create representative selection
            CUDA_VISIBLE_DEVICES=0 python ${UTILS_DIR}/concat_jsons.py \
                --file1 ${DATA_DIR}/${SELECTED_PREFIX}_${cur_selection_step}.json \
                --output ${DATA_DIR}/${SELECTED_PREFIX}_${cur_selection_step}_${SAMPLES_PER_CLUSTER}.json \
                --samples_per_cluster ${SAMPLES_PER_CLUSTER} \
                --random_select_samples
        fi
    fi

    # Add a small delay to prevent too rapid iterations
    sleep 5
done

echo "========================================"
echo "AutoDirector finished!"
echo "========================================"
