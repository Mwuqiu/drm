#!/bin/bash

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
  echo "Error: Please provide a target directory path as an argument."
  echo "Usage: $0 /path/to/video_directory"
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Path '$TARGET_DIR' does not exist or is not a directory."
  exit 1
fi

OUTPUT_FILE="$TARGET_DIR/rollout_success_counts.txt"

> "$OUTPUT_FILE"

echo "Counting rollout success files in '$TARGET_DIR'..."
echo "---" >> "$OUTPUT_FILE"

for i in $(seq 0 7)
do
  PATTERN="rollout_task_${i}_.*success.*\.mp4"
  COUNT=$(ls "$TARGET_DIR" | grep "$PATTERN" | wc -l)
  echo "rollout_task_${i}: ${COUNT} successful videos"
  echo "rollout_task_${i}: ${COUNT}" >> "$OUTPUT_FILE"
done

echo "---" >> "$OUTPUT_FILE"
echo "Done! Results saved to $OUTPUT_FILE"