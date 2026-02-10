export CUDA_VISIBLE_DEVICES=0

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

export MODEL_PATH=/home/gaash/Wasif/Tawheed/Seg-Zero/pretrained_models/Qwen2.5-VL-3B-Instruct

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=/home/gaash/Wasif/Tawheed/Seg-Zero_with_TAPO/training_scripts/visionreasoner_7b.yaml \
    data.train_files=/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset/hf_dataset_mcp_new \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.3 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=vision_reasoner \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.total_episodes=4 \
    trainer.save_checkpoint_path=/home/gaash/Wasif/Tawheed/Rmot/Saved_Checkpoints/TAPO/${RUN_NAME}
