#!/bin/bash

# Path to the pretrained model you downloaded.
REASONING_MODEL_PATH="/home/gaash/Wasif/Seg-Zero/pretrained_models/VisionReasoner-7B"

# Path to the segmentation model (will be downloaded automatically).
SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

# Path to your custom dataset.
TEST_DATA_PATH="/home/gaash/Wasif/Seg-Zero/data/sample1"

# Simplified output path logic.
MODEL_NAME=$(basename $REASONING_MODEL_PATH)
TEST_NAME=$(basename $TEST_DATA_PATH)
OUTPUT_PATH="./reasonseg_eval_results/${MODEL_NAME}_on_${TEST_NAME}"

mkdir -p $OUTPUT_PATH
echo "Results will be saved to: ${OUTPUT_PATH}"

echo "Starting evaluation on a single GPU..."
python3 evaluation_scripts/evaluation_visionreasoner.py \
    --reasoning_model_path $REASONING_MODEL_PATH \
    --segmentation_model_path $SEGMENTATION_MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --test_data_path $TEST_DATA_PATH \
    --idx 0 \
    --num_parts 1 \
    --batch_size 4

echo "Evaluation complete. Calculating final score..."

wait

python3 evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH

echo "--- All Done ---"

