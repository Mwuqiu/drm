#!/bin/bash

START_TASK=4
END_TASK=4
NUM_TRIALS=50

PYTHON_SCRIPTS=(
  # "scripts/eval_libero.py"
  "scripts/eval_libero_vqh.py"
)

echo "Starting evaluation for Tasks $START_TASK to $END_TASK with $NUM_TRIALS trials each..."
echo

for SCRIPT in "${PYTHON_SCRIPTS[@]}"
do
  echo "####################################################"
  echo "           EVALUATING WITH SCRIPT: $SCRIPT"
  echo "####################################################"
  echo

  for ((TASK_ID=$START_TASK; TASK_ID<=$END_TASK; TASK_ID++))
  do
    echo "===================================================="
    echo "             STARTING TASK $TASK_ID"
    echo "===================================================="

    for i in $(seq 0 $(($NUM_TRIALS - 1)))
    do
       echo "----------------------------------------------------"
       echo "Running Task $TASK_ID, Episode_idx = $i"
       echo "----------------------------------------------------"

       python "$SCRIPT" \
         --task_id_to_run "$TASK_ID" \
         --episode_idx_to_run "$i" \
         --exp_specific_des "Full_Task_${TASK_ID}_oeoe_task_specific"
       if [ $? -ne 0 ]; then
           echo "Error running Script: $SCRIPT, Task: $TASK_ID, Episode_idx: $i. Aborting."
           exit 1
       fi
    done
  done
done

echo
echo "Evaluation finished successfully for all tasks and scripts."