#!/bin/bash

PYTHON_SCRIPT="evaluation/eval_libero.py"
BASE_DIR="/home/wuqiu/lerobot/outputs/train/2025-12-04/15-18-52_smolvla/checkpoints"

CHECKPOINTS=("025000" "050000" "075000" "100000")

for ckpt in "${CHECKPOINTS[@]}"; do
    
    step_num=$((10#$ckpt)) 
    k_val=$((step_num / 1000))
    exp_name="Full_Task_${k_val}k"

    echo "========================================================"
    echo "正在测试 Checkpoint: $ckpt (实验名: $exp_name)"
    echo "路径: $BASE_DIR/$ckpt/pretrained_model"
    echo "========================================================"

    python "$PYTHON_SCRIPT" \
        --policy_path "$BASE_DIR/$ckpt/pretrained_model" \
        --experiment_name "$exp_name"

    echo "Checkpoint $ckpt 测试完成。"
    echo ""
done

echo "所有任务已全部完成！"