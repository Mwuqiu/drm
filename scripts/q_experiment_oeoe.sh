#!/bin/bash

# =================================================================
#                 CONFIGURATION SECTION
# =================================================================
PYTHON_SCRIPT="scripts/eval_libero_vqh.py"

# --- Experiment Parameters ---
START_TASK=3
END_TASK=3
NUM_TRIALS=50
Q_VALUES=(0.01 0.02 0.1)

# =================================================================
#                 EXECUTION LOGIC
# =================================================================

# --- Sanity Check ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本 '$PYTHON_SCRIPT' 未找到！"
    exit 1
fi

echo "============================================================"
echo "      开始批量评估实验"
echo "============================================================"
echo "Python 脚本: $PYTHON_SCRIPT"
echo "任务范围: $START_TASK 到 $END_TASK"
echo "每个任务的试验次数: $NUM_TRIALS"
echo "Q-Value 阈值列表: ${Q_VALUES[*]}"
echo "============================================================"
echo

# --- 主循环 ---

# 外层循环: 遍历 Q-value
for q_threshold in "${Q_VALUES[@]}"
do
  echo "############################################################"
  echo "         开始运行实验: Q_THRESHOLD = $q_threshold"
  echo "############################################################"
  echo

  # 内层循环: 遍历任务 ID
  for ((TASK_ID=$START_TASK; TASK_ID<=$END_TASK; TASK_ID++))
  do
    echo "===================================================="
    echo "             开始处理任务 TASK $TASK_ID"
    echo "===================================================="
    
    POLICY_PATH="" # 初始化变量
    case $TASK_ID in
      2)
        POLICY_PATH="outputs/train/2025-09-22/12-23-01_vqh/checkpoints/010000/pretrained_model"
        ;;
      3)
        POLICY_PATH="outputs/train/2025-10-04/20-13-43_vqh/checkpoints/005000/pretrained_model"
        ;;
      4)
        POLICY_PATH="outputs/train/2025-09-20/15-55-26_vqh/checkpoints/008000/pretrained_model"
        ;;
      5)
        POLICY_PATH="outputs/train/2025-10-01/19-29-36_vqh/checkpoints/005000/pretrained_model"
        ;;
      6)
        POLICY_PATH="outputs/train/2025-09-13/16-24-00_vqh/checkpoints/070000/pretrained_model"
        ;;
      7)
        POLICY_PATH="outputs/train/2025-10-03/16-58-43_vqh/checkpoints/005000/pretrained_model"
        ;;
      *)
        echo "错误: 没有为 TASK_ID=$TASK_ID 定义专用的 policy_path。正在中止。"
        exit 1
        ;;
    esac
    echo "当前任务使用的策略路径: $POLICY_PATH"

    # --- 新增需求 1 & 2: 根据 TASK_ID 准备额外的参数 ---
    EXTRA_ARGS="" # 初始化一个空字符串来存放额外参数
    if [[ "$TASK_ID" -eq 2 || "$TASK_ID" -eq 3 ]]; then
      # 当 task_id 为 2 或 3 时
      EXTRA_ARGS+=" --continue_on_done True"
    fi

    if [ "$TASK_ID" -eq 3 ]; then
      # 当 task_id 为 3 时
      EXTRA_ARGS+=" --max_steps 550"
    fi
    # ----------------------------------------------------

    SPECIFIC_OUTPUT_DIR="data/libero_elegant_1006/eval_vqh_qthresh_task_${TASK_ID}"

    # 最内层循环: 遍历 episode
    for i in $(seq 0 $(($NUM_TRIALS - 1)))
    do
       echo "----------------------------------------------------"
       echo "运行中 -> Q-Threshold: $q_threshold, Task: $TASK_ID, Episode: $i"
       echo "视频和日志输出至: $SPECIFIC_OUTPUT_DIR"
       if [ -n "$EXTRA_ARGS" ]; then
         echo "附加参数: $EXTRA_ARGS"
       fi
       echo "----------------------------------------------------"

       # 执行 Python 脚本，并传入所有动态生成的参数
       python "$PYTHON_SCRIPT" \
         --policy_path "$POLICY_PATH" \
         --q_change_threshold "$q_threshold" \
         --task_id_to_run "$TASK_ID" \
         --episode_idx_to_run "$i" \
         --video_out_path "$SPECIFIC_OUTPUT_DIR" \
         --exp_specific_des "qthres_${q_threshold}" \
         $EXTRA_ARGS # 将额外参数附加到命令末尾

       # 检查上一条命令是否成功执行
       if [ $? -ne 0 ]; then
           echo "错误: 脚本执行失败！"
           echo "参数: Q=$q_threshold, Task=$TASK_ID, Episode=$i"
           echo "正在中止所有评估。"
           exit 1
       fi
    done
  done
done

echo
echo "============================================================"
echo "    所有实验已成功完成！"
echo "============================================================"