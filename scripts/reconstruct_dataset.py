import logging
from reward_dataset import RewardLeRobotDataset
import torch
from pathlib import Path
from typing import Callable
import datasets
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging
import PIL

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    SOURCE_REPO_ID = "physical-intelligence/libero_long_task_5_new_datasets"
    
    # Create an instance of the dataset
    source_dataset = RewardLeRobotDataset(repo_id=SOURCE_REPO_ID)

    # --- OPTION 1: PROCESS AND SAVE WITH DENSE REWARDS (Original approach) ---
    # print("--- STAGE 1A: PROCESSING AND SAVING WITH DENSE REWARDS ---")
    # TARGET_REPO_ID_DENSE = "physical-intelligence/libero_task_9_with_hrewards_5"
    # newly_created_dataset_dense = source_dataset.process_and_save_with_dense_rewards(target_repo_id=TARGET_REPO_ID_DENSE)
    # print(f"Finished creating dense reward dataset: {TARGET_REPO_ID_DENSE}")


    # --- OPTION 2: PROCESS AND SAVE SUBGOAL-BASED EPISODES (Your new request) ---
    print("\n" + "=" * 50 + "\n")
    print("--- STAGE 1B: PROCESSING AND SAVING SUBGOAL-BASED EPISODES ---")
    # Define the target repository ID for your new subgoal-based dataset
    TARGET_REPO_ID_SUBGOAL = "physical-intelligence/libero_long_task_5_new_datasets_success_subgoal"
    
    # Call the new function to generate the dataset
    # You can adjust ramp_steps here if you want a different window size around the subgoal
    newly_created_dataset_subgoal = source_dataset.process_and_save_subgoal_episodes(
        target_repo_id=TARGET_REPO_ID_SUBGOAL, 
        ramp_steps=20
    )
    print(f"Finished creating subgoal-based dataset: {TARGET_REPO_ID_SUBGOAL}")
    print("\n" + "=" * 50 + "\n")


    # --- STAGE 2: LOADING AND INSPECTING A DATASET ---
    # You can now load either of the created datasets to inspect them
    
    # Example of loading the new subgoal dataset
    print("--- STAGE 2: LOADING THE SUBGOAL DATASET FOR INSPECTION ---")
    # Use the same repo ID you used for creation
    rewarded_dataset = RewardLeRobotDataset(repo_id=TARGET_REPO_ID_SUBGOAL) 
    print(f"Features of the new subgoal dataset: {list(rewarded_dataset.features.keys())}")

    # Inspect the first sample of the new dataset
    if len(rewarded_dataset) > 0:
        sample = rewarded_dataset[0]
        print("\nSample from the new subgoal dataset:")
        print("Sample keys :", sample.keys())
        print(f"  Contains 'reward' key: {'reward' in sample}")
        print(f"  Reward value: {sample['reward']}")
        print(f"  Contains 'done' key: {'done' in sample}")
        print(f"  Done value: {sample['done']}")
    else:
        print("The created subgoal dataset is empty. This might happen if no valid subgoals were found.")