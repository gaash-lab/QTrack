#!/bin/bash

#SBATCH --job-name=eval_reasonseg
#SBATCH --partition=highq
#SBATCH --gres=gpu:h100:1
#SBATCH --output=Vision_Reasoner_eval_logs/eval_output_%j.log
#SBATCH --error=eval_error_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00

echo "--- Starting Evaluation Job: $(date) ---"

# 1. Set up the environment
source ~/.bashrc
conda activate visionreasoner

# 2. Navigate to the project root directory
# This simplifies all the paths below
cd /home/gaash/Wasif/Seg-Zero

# 3. Define paths and variables for the evaluation
REASONING_MODEL_PATH="/home/gaash/Wasif/Seg-Zero/checkpoints_exp5/run_visionreasoner_7b_4x80G/global_step_4108/actor/huggingface"
# REASONING_MODEL_PATH="/home/gaash/Wasif/Seg-Zero/checkpoints_STS/bbox_pretrain/final"
SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"
# SEGMENTATION_MODEL_PATH="wanglab/medsam-vit-base"
MODEL_DIR="EXP5"
TEST_DATA_PATH="/home/gaash/Wasif/datasets/TestData/Combined_1016"
# TEST_DATA_PATH="Ricky06662/ReasonSeg_val"
TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

echo "Output will be saved to: $OUTPUT_PATH"
mkdir -p $OUTPUT_PATH

# 4. Run the evaluation script
echo "Running evaluation..."
python3 evaluation_scripts/evaluation_visionreasoner2.py \
    --reasoning_model_path $REASONING_MODEL_PATH \
    --segmentation_model_path $SEGMENTATION_MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --test_data_path $TEST_DATA_PATH \
    --idx 0 \
    --num_parts 1 \
    --batch_size 4

# 5. Calculate the final IoU score
echo "Calculating IoU..."
python3 evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH

echo "--- Job Finished: $(date) ---"