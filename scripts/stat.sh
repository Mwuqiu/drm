#!/bin/bash

# 接收第一个参数作为目标目录
TARGET_DIR="$1"

# --- 增加健壮性检查 ---
# 检查是否提供了目录参数
if [ -z "$TARGET_DIR" ]; then
  echo "错误: 请提供一个目标目录路径作为参数。"
  echo "用法: $0 /path/to/video_directory"
  exit 1
fi

# 检查提供的路径是否存在且是一个目录
if [ ! -d "$TARGET_DIR" ]; then
  echo "错误: 路径 '$TARGET_DIR' 不存在或不是一个目录。"
  exit 1
fi

# 定义输出文件名 (我们让它在目标目录中生成)
OUTPUT_FILE="$TARGET_DIR/rollout_success_counts.txt"

# 清空或创建输出文件
> "$OUTPUT_FILE"

echo "Counting rollout success files in '$TARGET_DIR'..."
echo "---" >> "$OUTPUT_FILE"

# 循环遍历 task_0 到 task_9
for i in $(seq 0 9)
do
  PATTERN="rollout_task_${i}.*_success.*\.mp4"
  
  # 修改核心命令，对指定的 TARGET_DIR 执行 ls
  COUNT=$(ls "$TARGET_DIR" | grep "$PATTERN" | wc -l)
  
  echo "rollout_task_${i}: ${COUNT} successful videos"
  echo "rollout_task_${i}: ${COUNT}" >> "$OUTPUT_FILE"
done

echo "---" >> "$OUTPUT_FILE"
echo "Done! Results saved to $OUTPUT_FILE"