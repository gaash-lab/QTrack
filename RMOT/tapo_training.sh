#!/bin/bash
#SBATCH --job-name=visionreasoner_job
#SBATCH --partition=highq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=highq
#SBATCH --output=Rmot/tapo_training_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

source ~/.bashrc
conda init
conda activate tapo
export VLLM_USE_MODEL_INSPECTOR=0
export VLLM_ATTENTION_BACKEND=XFORMERS

cd /home/gaash/Wasif/Tawheed/Seg-Zero_with_TAPO/training_scripts
bash run_visionreasoner_7b_4x80G.sh