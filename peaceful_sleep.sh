#!/bin/bash
# This script runs the VQH policy training for multiple tasks sequentially.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting training for Task 5..."
python src/vqh_training_policy.py \
    --dataset.repo_id=libero_elegant_1006/task_5_binary_reward \
    --policy.pretrained_smolvla_path=/home/wuqiu/lerobot/outputs/train/2025-10-01/17-25-40_smolvla/checkpoints/010000/pretrained_model

echo "---------------------------------"
echo "Starting training for Task 6..."
python src/vqh_training_policy.py \
    --dataset.repo_id=libero_elegant_1006/task_6_binary_reward \
    --policy.pretrained_smolvla_path=/home/wuqiu/lerobot/outputs/train/2025-09-12/16-46-10_smolvla/checkpoints/100000/pretrained_model

echo "---------------------------------"
echo "Starting training for Task 7..."
python src/vqh_training_policy.py \
    --dataset.repo_id=libero_elegant_1006/task_7_binary_reward \
    --policy.pretrained_smolvla_path=/home/wuqiu/lerobot/outputs/train/2025-10-02/01-40-13_smolvla/checkpoints/100000/pretrained_model

echo "---------------------------------"
echo "All training tasks completed successfully!"