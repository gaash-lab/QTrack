#!/bin/bash

# --- EDIT 1: Path to YOUR trained and merged model ---
REASONING_MODEL_PATH="/home/gaash/Wasif/Seg-Zero/checkpoints/run_visionreasoner_7b_4x80G/global_step_62/actor/huggingface"

SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

# --- EDIT 2: A simple name for the output directory ---
MODEL_DIR="my_trained_model_final_eval"
TEST_DATA_PATH="Ricky06662/ReasonSeg_val"


TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
# --- EDIT 3: Corrected path for the output directory ---
OUTPUT_PATH="Seg-Zero/reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

# Create output directory
mkdir -p $OUTPUT_PATH
echo "Results will be saved to: ${OUTPUT_PATH}"
~
# --- EDIT 4: Corrected path to the python script ---
NUM_PARTS=8
for idx in {0..7}; do
    export CUDA_VISIBLE_DEVICES=$idx
    python3 Seg-Zero/evaluation_scripts/evaluation_visionreasoner.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 100 & # Kept safer batch size of 8
done

# Wait for all processes to complete
wait

# --- EDIT 5: Corrected path to the final calculation script ---
python3 Seg-Zero/evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH

echo "--- Evaluation Script Finished ---"