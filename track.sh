#!/bin/bash
#SBATCH --job-name=visionreasoner_job
#SBATCH --partition=highq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=highq
#SBATCH --output=tawheed_output_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00

source ~/.bashrc
conda init
conda activate visionreasoner_backup
cd /home/gaash/Wasif/Seg-Zero/training_scripts/

python model_merger_final.py --local_dir /home/gaash/Wasif/Tawheed/Rmot/Saved_Checkpoints/TAPO/run_visionreasoner_7b_4x80G/global_step_7196/actor

# python /home/gaash/Wasif/Tawheed/prepare_mcp_dataset.py