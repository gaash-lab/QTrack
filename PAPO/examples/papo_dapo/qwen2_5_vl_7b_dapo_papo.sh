#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

CUDA_IDS=0,1,2,3
N_GPU=4

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=128
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096

EXP_NAME="qwen2_5_vl_7b__dapo__papo_double_ent-0.03__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_mini${MINI_ROLLOUT_BATCH_SIZE}"

CONGI_FILE="examples/configs/config_dapo_papo.yaml"
TRAIN_FILE="PAPOGalaxy/PAPO_ViRL39K_train"
VAL_FILE="PAPOGalaxy/PAPO_MMK12_test"

FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score"

KL_PRCP_COEF=0.01

## Double Entropy Loss
USE_AUG_ENTROPY_LOSS=true
AUG_ENTROPY_LOSS_COEF=0.03
USE_ORI_ENTROPY_LOSS=true
ORI_ENTROPY_LOSS_COEF=0.03

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=true \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF} \
    algorithm.use_aug_entropy_loss=${USE_AUG_ENTROPY_LOSS} \
    algorithm.aug_entropy_loss_coef=${AUG_ENTROPY_LOSS_COEF} \
    algorithm.use_ori_entropy_loss=${USE_ORI_ENTROPY_LOSS} \
    algorithm.ori_entropy_loss_coef=${ORI_ENTROPY_LOSS_COEF}
