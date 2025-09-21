#!/bin/bash

TASK_ID=4
NUM_TRIALS=10 
echo "Starting evaluation for Task $TASK_ID with $NUM_TRIALS trials..."

for i in $(seq 0 $(($NUM_TRIALS - 1)))
do
   echo "----------------------------------------------------"
   echo "Running evaluation for episode_idx = $i"
   echo "----------------------------------------------------"   
   python scripts/eval_libero_vqh.py --episode_idx_to_run $i    
   if [ $? -ne 0 ]; then
       echo "Error running episode_idx = $i. Aborting."
       exit 1
   fi
done
echo "Evaluation finished successfully."