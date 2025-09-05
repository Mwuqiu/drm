# import logging
# from reward_dataset import RewardLeRobotDataset
# import torch
# from pathlib import Path
# from typing import Callable
# import datasets
# from datasets import load_dataset
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
# from lerobot.common.datasets.utils import hf_transform_to_torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np
# import logging
# import PIL

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     SOURCE_REPO_ID = "libero_elegant/task_5"    
#     TARGET_REPO_ID_VLM_LABELED = "libero_elegant/task_5_reward"

#     print("\n" + "=" * 50)
#     print(f"Loading source dataset: {SOURCE_REPO_ID}")
#     source_dataset = RewardLeRobotDataset(repo_id=SOURCE_REPO_ID)
#     print(f"Source dataset loaded with {source_dataset.num_episodes} episodes.")
#     print("=" * 50 + "\n")

#     print("\n" + "=" * 50)
#     print(f"Starting VLM labeling process. Target repo: {TARGET_REPO_ID_VLM_LABELED}")

#     vlm_labeled_dataset = source_dataset.process_and_label_with_vlm(
#         target_repo_id=TARGET_REPO_ID_VLM_LABELED
#     )
#     print(f"Finished creating VLM-labeled dataset: {TARGET_REPO_ID_VLM_LABELED}")
#     print("=" * 50 + "\n")
    
#     print("\n" + "=" * 50)
#     print("--- Loading the new VLM-labeled dataset for inspection ---")
#     try:
#         rewarded_dataset = RewardLeRobotDataset(repo_id=TARGET_REPO_ID_VLM_LABELED) 
#         print(f"Successfully loaded the new dataset.")
#         print(f"Features of the new dataset: {list(rewarded_dataset.features.keys())}")

#         if len(rewarded_dataset) > 0:
#             print("\nInspecting a sample from the new dataset...")
#             sample = rewarded_dataset[-1]
#             print(f"  Sample from the end of an episode:")
#             print(f"  Reward value: {sample['reward']}")
#             print(f"  Done value: {sample['done']}")
#         else:
#             print("The created dataset is empty.")
#     except Exception as e:
#         print(f"Could not load or inspect the newly created dataset. Error: {e}")
#     print("=" * 50 + "\n")


# reconstruct_dataset.py
# reconstruct_dataset.py

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
    
    # Define the range of tasks to process
    start_task = 2
    end_task = 9

    # Loop through each task number from start_task to end_task (inclusive)
    for task_num in range(start_task, end_task + 1):
        print("\n" + "#" * 70)
        print(f"### PROCESSING TASK {task_num} ###")
        print("#" * 70 + "\n")

        # --- 配置区域 (动态生成路径和ID) ---
        SOURCE_REPO_ID = f"libero_elegant/task_{task_num}"
        TARGET_REPO_ID_MANUAL_LABELED = f"libero_elegant/task_{task_num}_manual_reward"
        ANNOTATIONS_JSON_PATH = f"task_annotations/task_{task_num}_episode_annotations.json"
        
        # --- 加载源数据集 ---
        print("\n" + "=" * 50)
        print(f"Loading source dataset: {SOURCE_REPO_ID}")
        try:
            source_dataset = RewardLeRobotDataset(repo_id=SOURCE_REPO_ID)
            print(f"Source dataset loaded with {source_dataset.num_episodes} episodes.")
        except Exception as e:
            print(f"ERROR: Could not load source dataset for task {task_num}. Skipping. Error: {e}")
            continue # 跳过当前任务，继续下一个
        print("=" * 50 + "\n")

        # --- 基于JSON文件进行处理和标注 ---
        print("\n" + "=" * 50)
        print(f"Starting labeling process from JSON. Target repo: {TARGET_REPO_ID_MANUAL_LABELED}")

        try:
            manual_labeled_dataset = source_dataset.process_and_label_from_json(
                target_repo_id=TARGET_REPO_ID_MANUAL_LABELED,
                annotations_path=ANNOTATIONS_JSON_PATH
            )
            print(f"Finished creating manually-labeled dataset: {TARGET_REPO_ID_MANUAL_LABELED}")
        except FileNotFoundError as e:
            print(f"ERROR: Annotation file not found for task {task_num} at '{ANNOTATIONS_JSON_PATH}'. Skipping. Error: {e}")
            continue
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during processing for task {task_num}. Skipping. Error: {e}")
            continue
        print("=" * 50 + "\n")
        
        # --- 加载并检查新生成的数据集 ---
        print("\n" + "=" * 50)
        print(f"--- Loading the new manually-labeled dataset for inspection (Task {task_num}) ---")
        try:
            rewarded_dataset = RewardLeRobotDataset(repo_id=TARGET_REPO_ID_MANUAL_LABELED) 
            print(f"Successfully loaded the new dataset.")
            print(f"Features of the new dataset: {list(rewarded_dataset.features.keys())}")

            if len(rewarded_dataset) > 0:
                print("\nInspecting a sample from the new dataset...")
                found_reward = False
                for ep_idx in range(rewarded_dataset.num_episodes):
                    ep_end_abs = rewarded_dataset.episode_data_index["to"][ep_idx].item()
                    sample = rewarded_dataset[ep_end_abs - 1]
                    if sample['reward'] > 0:
                        print(f"  Found a positive reward in episode {ep_idx} (successful):")
                        print(f"  Reward value: {sample['reward']}")
                        print(f"  Done value: {sample['done']}")
                        found_reward = True
                        break
                if not found_reward:
                    print("  No positive rewards found in terminal frames (all episodes may be marked as fail).")
                    sample = rewarded_dataset[-1]
                    print(f"\n  Inspecting last frame of the last episode:")
                    print(f"  Reward value: {sample['reward']}")
                    print(f"  Done value: {sample['done']}")
            else:
                print("The created dataset is empty.")
        except Exception as e:
            print(f"Could not load or inspect the newly created dataset for task {task_num}. Error: {e}")
        print("=" * 50 + "\n")

    print("\n" + "#" * 70)
    print("### ALL TASKS PROCESSED! ###")
    print("#" * 70 + "\n")