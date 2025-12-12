#!/bin/bash

TARGET_DIR="/home/wuqiu/drm/rollouts/libero_elegant_1205/Full_Task_2"

eval TARGET_DIR="$TARGET_DIR"

if [ "$#" -ne 1 ]; then
    echo "❌ 用法错误"
    echo "请提供一个 episode 序号作为参数"
    echo "例如: $0 17"
    exit 1
fi

EPISODE_NUM=$1

if ! [[ "$EPISODE_NUM" =~ ^[0-9]+$ ]]; then
    echo "❌ 错误: episode 序号必须是一个正整数。"
    exit 1
fi

shopt -s nullglob
files=("$TARGET_DIR"/rollout_task_*_episode_${EPISODE_NUM}_*.mp4)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "❌ 错误: 在文件夹 $TARGET_DIR 中没有找到 episode $EPISODE_NUM 的文件。"
    exit 1
elif [ ${#files[@]} -gt 1 ]; then
    echo "❌ 错误: 找到了多个关于 episode $EPISODE_NUM 的文件，请整理文件夹后重试。"
    printf "   - %s\n" "${files[@]}"
    exit 1
fi

original_filepath="${files[0]}"
original_filename=$(basename "$original_filepath")

new_filename=""
if [[ "$original_filename" == *success* ]]; then
    new_filename="${original_filename/success/failure}"
elif [[ "$original_filename" == *failure* ]]; then
    new_filename="${original_filename/failure/success}"
else
    echo "❌ 错误: 文件名 '$original_filename' 中不包含 'success' 或 'failure'。"
    exit 1
fi

new_filepath="$TARGET_DIR/$new_filename"

echo "正在执行重命名..."
echo "  旧名称: $original_filename"
echo "  新名称: $new_filename"

mv "$original_filepath" "$new_filepath"
echo "✅ 操作完成！"