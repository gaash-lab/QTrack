#!/bin/bash
#SBATCH --job-name=visionreasoner_job
#SBATCH --partition=highq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=highq
#SBATCH --output=train_logs/tapo_training_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

source ~/.bashrc
conda init
conda activate tapo
export VLLM_USE_MODEL_INSPECTOR=0
export VLLM_ATTENTION_BACKEND=XFORMERS

bash /home/gaash/Wasif/Tawheed/QTrack/training_scripts/train_qtrack.sh


python prepare_dataset/build_qtrack_dataset_v2.py \
    --dataset_root /home/gaash/Wasif/Tawheed/MOT_grounding_Dataset \
    --train_dir train \
    --json annotations.json \
    --output_dir hf_dataset_qtrack_full

# python /home/gaash/Wasif/Tawheed/prepare_mcp_dataset.py