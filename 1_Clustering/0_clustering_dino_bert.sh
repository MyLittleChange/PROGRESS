#!/bin/bash
#SBATCH --job-name=clustering_dino-bert
#SBATCH --output=clustering_dino-bert_output.txt
#SBATCH --error=clustering_dino-bert_error.txt

# ============================================================
# PROGRESS Clustering Script
# Two-stage clustering:
#   1. 1000 centers - for annotating all samples with cluster IDs
#   2. 5000 centers - for selecting top 50K/60K warm-up samples
# ============================================================

# Configuration - modify these paths for your setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Data paths - SET THESE
DATA_FILE=""           # Path to data JSON file (e.g., llava_v1_5_mix665k.json)
OUTPUT_DIR=""          # Directory to save outputs
DINO_FEATURES=""       # Path to DINO features (.npy)
BERT_FEATURES=""       # Path to BERT/question features (.npy)

# Hyperparameters
TEMP=0.1
FEATURE_TYPE="dino-bert"

# Two different centroid counts for different purposes
NCENTROIDS_ANNOTATION=1000   # For annotating all samples with cluster IDs
NCENTROIDS_SELECTION=5000    # For selecting top 50K/60K samples

# Sample ratios for warm-up stages
# 0.075 for ~50K samples (first warm-up checkpoint)
# 0.09 for ~60K samples (second warm-up checkpoint)
SAMPLE_RATIO_50K=0.075
SAMPLE_RATIO_60K=0.09

# Create output directories
SAVE_FOLDER_1K="${OUTPUT_DIR}/${NCENTROIDS_ANNOTATION}_msa_save_folder_${FEATURE_TYPE}"
SAVE_FOLDER_5K="${OUTPUT_DIR}/${NCENTROIDS_SELECTION}_msa_save_folder_${FEATURE_TYPE}"
mkdir -p "${SAVE_FOLDER_1K}"
mkdir -p "${SAVE_FOLDER_5K}"

echo "========================================"
echo "PROGRESS Clustering Pipeline"
echo "========================================"
echo "Stage 1: ${NCENTROIDS_ANNOTATION} centers for sample annotation"
echo "Stage 2: ${NCENTROIDS_SELECTION} centers for warm-up selection"
echo "========================================"

# ============================================================
# STAGE 1: Clustering with 1000 centers (for annotation)
# ============================================================
echo ""
echo "========================================"
echo "STAGE 1: Clustering with ${NCENTROIDS_ANNOTATION} centers"
echo "Purpose: Annotate all samples with cluster IDs"
echo "========================================"

# Step 1.1: Compute centroids with 1000 clusters
echo "Step 1.1: Computing ${NCENTROIDS_ANNOTATION} centroids..."
python ${PROJECT_ROOT}/tinyllava/eval/score/coincide/compute_centroids.py \
        --llava_data_file ${DATA_FILE} \
        --output_file ${OUTPUT_DIR}/output_${NCENTROIDS_ANNOTATION}.json \
        --dino_features_path ${DINO_FEATURES} \
        --bert_features_path ${BERT_FEATURES} \
        --sim_metric cosine \
        --Kmeans_with_cos_dist \
        --feature_type ${FEATURE_TYPE} \
        --save_folder ${SAVE_FOLDER_1K} \
        --ncentroids ${NCENTROIDS_ANNOTATION} \
        --niter 50 \
        --seed 1234

echo "Stage 1 complete! Generated:"
echo "  - ${SAVE_FOLDER_1K}/nearest_cent.npy (cluster assignments)"
echo "  - ${SAVE_FOLDER_1K}/dist_to_cent.npy (distances to centroids)"

# ============================================================
# STAGE 2: Clustering with 5000 centers (for selection)
# ============================================================
echo ""
echo "========================================"
echo "STAGE 2: Clustering with ${NCENTROIDS_SELECTION} centers"
echo "Purpose: Select top 50K/60K samples for warm-up"
echo "========================================"

# Step 2.1: Compute centroids with 5000 clusters
echo "Step 2.1: Computing ${NCENTROIDS_SELECTION} centroids..."
python ${PROJECT_ROOT}/tinyllava/eval/score/coincide/compute_centroids.py \
        --llava_data_file ${DATA_FILE} \
        --output_file ${OUTPUT_DIR}/output_${NCENTROIDS_SELECTION}.json \
        --dino_features_path ${DINO_FEATURES} \
        --bert_features_path ${BERT_FEATURES} \
        --sim_metric cosine \
        --Kmeans_with_cos_dist \
        --feature_type ${FEATURE_TYPE} \
        --save_folder ${SAVE_FOLDER_5K} \
        --ncentroids ${NCENTROIDS_SELECTION} \
        --niter 50 \
        --seed 1234

# Step 2.2: Compute cluster transferability
echo "Step 2.2: Computing cluster transferability..."
python ${PROJECT_ROOT}/tinyllava/eval/score/coincide/cluster_transferability.py \
        --centroid_embed_path ${SAVE_FOLDER_5K}/kmeans_centroids.npy \
        --transferability_path ${SAVE_FOLDER_5K}/transfer.npy \
        --k 4 \
        --ours \
        --knn_path ${SAVE_FOLDER_5K}/knn

# Step 2.3a: Cluster-wise pruning for 50K (first warm-up)
echo "Step 2.3a: Pruning for 50K samples (ratio=${SAMPLE_RATIO_50K})..."
python ${PROJECT_ROOT}/tinyllava/eval/score/coincide/cluster_wise_prune.py \
        --cluster_path ${SAVE_FOLDER_5K}/nearest_cent.npy \
        --transfer_path ${SAVE_FOLDER_5K}/transfer.npy \
        --dino_features_path ${DINO_FEATURES} \
        --bert_features_path ${BERT_FEATURES} \
        --fraction ${SAMPLE_RATIO_50K} \
        --temp ${TEMP} \
        --feature_type ${FEATURE_TYPE} \
        --output_indices_path ${OUTPUT_DIR}/selected_indices_${SAMPLE_RATIO_50K}_${FEATURE_TYPE}.npy

# Step 2.3b: Cluster-wise pruning for 60K (second warm-up)
echo "Step 2.3b: Pruning for 60K samples (ratio=${SAMPLE_RATIO_60K})..."
python ${PROJECT_ROOT}/tinyllava/eval/score/coincide/cluster_wise_prune.py \
        --cluster_path ${SAVE_FOLDER_5K}/nearest_cent.npy \
        --transfer_path ${SAVE_FOLDER_5K}/transfer.npy \
        --dino_features_path ${DINO_FEATURES} \
        --bert_features_path ${BERT_FEATURES} \
        --fraction ${SAMPLE_RATIO_60K} \
        --temp ${TEMP} \
        --feature_type ${FEATURE_TYPE} \
        --output_indices_path ${OUTPUT_DIR}/selected_indices_${SAMPLE_RATIO_60K}_${FEATURE_TYPE}.npy

echo ""
echo "========================================"
echo "Clustering complete!"
echo "========================================"
echo ""
echo "Stage 1 outputs (${NCENTROIDS_ANNOTATION} centers - for annotation):"
echo "  - ${SAVE_FOLDER_1K}/nearest_cent.npy"
echo "  - ${SAVE_FOLDER_1K}/dist_to_cent.npy"
echo ""
echo "Stage 2 outputs (${NCENTROIDS_SELECTION} centers - for selection):"
echo "  - ${OUTPUT_DIR}/selected_indices_${SAMPLE_RATIO_50K}_${FEATURE_TYPE}.npy (50K)"
echo "  - ${OUTPUT_DIR}/selected_indices_${SAMPLE_RATIO_60K}_${FEATURE_TYPE}.npy (60K)"
echo "========================================"
