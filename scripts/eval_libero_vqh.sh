#!/bin/bash

PYTHON_SCRIPT="evaluation/eval_libero_vqh.py"

START_TASK=0
END_TASK=7

for ((TASK_ID=$START_TASK; TASK_ID<=$END_TASK; TASK_ID++)); do
    exp_name="Task_${TASK_ID}_Worst"
    python "$PYTHON_SCRIPT" \
        --num_trials_per_task 10 \
        --experiment_name "$exp_name" \
        --task_id_to_run "$TASK_ID"
done