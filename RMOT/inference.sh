#!/bin/bash
#SBATCH --job-name=inference_test
#SBATCH --partition=highq
#SBATCH --gres=gpu:h100:1  # <-- Request 1 H100 GPU
#SBATCH --qos=highq
#SBATCH --output=inference_output_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G         # <-- Request a reasonable amount of RAM
#SBATCH --time=00:30:00   # <-- Inference is fast, 30 mins is plenty of time

echo "--- Starting Inference Job on a GPU Node ---"

# Activate your conda environment
source ~/.bashrc
conda activate visionreasoner

# Create the output directory if it doesn't exist
mkdir -p Seg-Zero/outputs

# The exact inference command you want to run
python3 Seg-Zero/inference_scripts/infer_multi_object.py \
    --reasoning_model_path /home/gaash/Wasif/Seg-Zero/checkpoints/run_visionreasoner_7b_4x80G/global_step_62/actor/huggingface \
    --image_path Seg-Zero/assets/test_image.png \
    --text "What are the wheels of the bicycle made of?" \
    --output_path Seg-Zero/outputs/watermelon_bike_result.png

echo "--- Inference Job Complete ---"