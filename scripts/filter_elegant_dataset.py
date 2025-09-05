# filter_successful_dataset.py

import logging
from reward_dataset import RewardLeRobotDataset
from pathlib import Path
import json
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Define the range of tasks to process
    start_task = 1
    end_task = 9

    # Loop through each task number
    for task_num in range(start_task, end_task + 1):
        print("\n" + "#" * 70)
        print(f"### PROCESSING TASK {task_num} ###")
        print("#" * 70 + "\n")

        # --- Dynamic Configuration for each task ---
        SOURCE_REPO_ID = f"libero_elegant/task_{task_num}" 
        TARGET_REPO_ID = f"libero_elegant/task_{task_num}_elegant" 
        ANNOTATIONS_JSON_PATH = f"task_annotations/task_{task_num}_episode_annotations.json"

        # --- Check if target directory already exists ---
        target_path = HF_LEROBOT_HOME / TARGET_REPO_ID
        if target_path.exists():
            print(f"INFO: Target directory '{target_path}' already exists. Skipping Task {task_num}.")
            continue

        # --- Load Source Dataset ---
        try:
            print("\n" + "=" * 50)
            print(f"Loading source dataset: {SOURCE_REPO_ID}")
            source_dataset = RewardLeRobotDataset(repo_id=SOURCE_REPO_ID)
            print(f"Source dataset loaded with {source_dataset.num_episodes} episodes.")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"ERROR: Could not load source dataset for Task {task_num}. Skipping. Error: {e}")
            continue

        # --- Run Filtering Process ---
        try:
            print("\n" + "=" * 50)
            print(f"Starting pure filtering process. Target repo: {TARGET_REPO_ID}")
            filtered_dataset = source_dataset.filter_episodes_from_json(
                target_repo_id=TARGET_REPO_ID,
                annotations_path=ANNOTATIONS_JSON_PATH
            )
            print(f"Finished creating the filtered dataset: {TARGET_REPO_ID}")
            print("=" * 50 + "\n")
        except FileNotFoundError:
            print(f"ERROR: Annotation file not found at '{ANNOTATIONS_JSON_PATH}' for Task {task_num}. Skipping.")
            continue
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during filtering for Task {task_num}. Skipping. Error: {e}")
            continue
        
        # --- Inspection of the New Dataset ---
        print("\n" + "=" * 50)
        print(f"--- Loading the new filtered dataset for inspection (Task {task_num}) ---")
        try:
            final_dataset = LeRobotDataset(repo_id=TARGET_REPO_ID) 
            print(f"Successfully loaded the new dataset.")
            print(f"Number of episodes in new dataset: {final_dataset.num_episodes}")
            print(f"Features of the new dataset: {list(final_dataset.features.keys())}")
            
            with open(ANNOTATIONS_JSON_PATH, 'r') as f:
                annotations = json.load(f)
                true_count = sum(1 for is_successful in annotations.values() if is_successful)
                print(f"Number of 'true' entries in JSON: {true_count}")
            
            if final_dataset.num_episodes == true_count:
                print("SUCCESS: The number of episodes in the new dataset matches the number of successful episodes.")
            else:
                print("WARNING: Mismatch in episode count!")

        except Exception as e:
            print(f"Could not load or inspect the newly created dataset for Task {task_num}. Error: {e}")
        print("=" * 50 + "\n")

    print("\n" + "#" * 70)
    print("### ALL TASKS PROCESSED! ###")
    print("#" * 70 + "\n")