#!/bin/bash

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
  echo "错误: 请提供一个目标目录路径作为参数。"
  echo "用法: $0 /path/to/video_directory"
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "错误: 路径 '$TARGET_DIR' 不存在或不是一个目录。"
  exit 1
fi

OUTPUT_FILE="$TARGET_DIR/rollout_success_counts.txt"

> "$OUTPUT_FILE"

echo "Counting rollout success files in '$TARGET_DIR'..."
echo "---" >> "$OUTPUT_FILE"

PATTERN=".*success.*\.mp4"
COUNT=$(ls "$TARGET_DIR" | grep "$PATTERN" | wc -l)

echo "rollout: ${COUNT} successful videos"
echo "rollout: ${COUNT}" >> "$OUTPUT_FILE"

echo "---" >> "$OUTPUT_FILE"
echo "Done! Results saved to $OUTPUT_FILE"
