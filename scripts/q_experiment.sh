#!/bin/bash

PYTHON_SCRIPT="scripts/eval_libero_vqh.py"

Q_VALUES=(0.01 0.02 0.1 0.2 0.5)

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本 '$PYTHON_SCRIPT' 未找到！"
    exit 1
fi

for q in "${Q_VALUES[@]}"
do
  echo "-----------------------------------------------------"
  echo "正在运行实验: q_change_threshold = $q"
  echo "-----------------------------------------------------"
  exp_des="qthres_$q"
  python "$PYTHON_SCRIPT" --q_change_threshold "$q" --exp_specific_des "$exp_des"
done

echo "-----------------------------------------------------"
echo "所有实验已完成！"