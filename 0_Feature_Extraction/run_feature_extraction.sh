#!/bin/bash
# Feature Extraction Script
# Extracts both question (text) features and DINO (image) features

set -e  # Exit on error

# Configuration - modify these paths for your setup
DATA_PATH=""           # Path to JSON/JSONL file
IMAGE_DIR=""           # Directory containing images
OUTPUT_DIR=""          # Directory to save features

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --image_dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_PATH" ] || [ -z "$IMAGE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --data_path <path> --image_dir <path> --output_dir <path>"
    echo ""
    echo "Arguments:"
    echo "  --data_path   Path to JSON/JSONL file containing data"
    echo "  --image_dir   Directory containing images"
    echo "  --output_dir  Directory to save extracted features"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Feature Extraction Pipeline"
echo "========================================"
echo "Data path: $DATA_PATH"
echo "Image dir: $IMAGE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

# Step 1: Extract question (text) features
echo ""
echo "[Step 1/2] Extracting question features..."
python "$SCRIPT_DIR/questions_features.py" \
    --path "$DATA_PATH" \
    --output_path "$OUTPUT_DIR/q_features.npy"

# Step 2: Extract DINO (image) features
echo ""
echo "[Step 2/2] Extracting DINO features..."
python "$SCRIPT_DIR/dino_features.py" \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --output_path "$OUTPUT_DIR/dino_features.npy" \
    --batch_size 64 \
    --num_workers 4

echo ""
echo "========================================"
echo "Feature extraction complete!"
echo "Output files:"
echo "  - $OUTPUT_DIR/q_features.npy"
echo "  - $OUTPUT_DIR/dino_features.npy"
echo "========================================"
