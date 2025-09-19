import os
import logging
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import DEFAULT_FEATURES

SOURCE_REPO_ID = "physical-intelligence/libero_spatial"
TARGET_TASK_INDEX = 7
TARGET_REPO_ID = "libero_elegant_0906/task_2"

from tqdm import tqdm

def create_subset_dataset(target_repo_id: str, episodes_to_load: list[int], source_meta: LeRobotDatasetMetadata):
    """
    This function remains unchanged from your original script.
    It takes a list of episode indices and creates a new dataset from them.
    """
    print("-" * 70)
    print(f"Creating new dataset '{target_repo_id}' with {len(episodes_to_load)} episodes.")
    print(f"Directly loading the {len(episodes_to_load)} required episodes...")
    
    source_ds_filtered = LeRobotDataset(
        repo_id=source_meta.repo_id,
        episodes=episodes_to_load,
        video_backend="pyav"
    )
    print("Source data for subset loaded.")

    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_meta.fps,
        features=source_meta.features,
        robot_type=source_meta.robot_type,
        use_videos=(source_meta.video_path is not None),
        image_writer_threads=4,
    )

    default_keys = set(DEFAULT_FEATURES.keys())
    current_episode_being_processed = -1

    for i in tqdm(range(len(source_ds_filtered)), desc="Copying frames"):
        frame_data = source_ds_filtered[i]
        ep_idx = frame_data["episode_index"].item()
        if ep_idx != current_episode_being_processed:
            if current_episode_being_processed != -1:
                target_ds.save_episode()
                print(f"  -> Saved source episode {current_episode_being_processed} as new episode {target_ds.meta.total_episodes - 1}")

            current_episode_being_processed = ep_idx
            print(f"Processing source episode {current_episode_being_processed}...")

        clean_frame = {}
        for key, value in frame_data.items():
            if key in default_keys:
                continue
            if hasattr(value, "ndim") and value.ndim == 3 and key in source_meta.camera_keys:
                clean_frame[key] = value.permute(1, 2, 0).numpy()
            elif hasattr(value, 'numpy'):
                clean_frame[key] = value.numpy()
            else:
                clean_frame[key] = value
        
        clean_frame.pop('task', None)
        task_string = source_meta.tasks[frame_data["task_index"].item()]
        target_ds.add_frame(clean_frame, task=task_string)

    if target_ds.episode_buffer["size"] > 0:
        target_ds.save_episode()
        print(f"  -> Saved source episode {current_episode_being_processed} as new episode {target_ds.meta.total_episodes - 1}")

    print(f"\nFinished creating dataset '{target_repo_id}'.")
    print(f"It is ready locally at: {target_ds.root.resolve()}")
    print("-" * 70)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Attempting to extract task {TARGET_TASK_INDEX} from '{SOURCE_REPO_ID}'.")
    print("Loading source dataset metadata...")

    source_metadata = LeRobotDatasetMetadata(repo_id=SOURCE_REPO_ID)
    try:
        task_name = source_metadata.tasks[TARGET_TASK_INDEX]
        print(f"Target task found: Index {TARGET_TASK_INDEX} -> '{task_name}'")
    except IndexError:
        print(f"Error: Task index {TARGET_TASK_INDEX} is out of bounds.")
        print(f"The dataset '{SOURCE_REPO_ID}' has {len(source_metadata.tasks)} tasks (indices 0 to {len(source_metadata.tasks)-1}).")
        exit()

    print(f"\nScanning metadata in '{SOURCE_REPO_ID}' to find episodes for task {TARGET_TASK_INDEX}.")
    target_task_name = source_metadata.tasks[TARGET_TASK_INDEX]
    episodes_for_target_task = set()

    for episode_info in tqdm(source_metadata.episodes.values(), desc="Scanning episode metadata"):
        if target_task_name in episode_info.get("tasks", []):
            episodes_for_target_task.add(episode_info["episode_index"])

    if not episodes_for_target_task:
        print(f"\nError: No episodes were found for task index {TARGET_TASK_INDEX} ('{task_name}').")
    else:
        episodes_for_target_task = list(episodes_for_target_task)
        episodes_for_target_task = episodes_for_target_task[:35]        
        episodes_to_load = sorted(list(episodes_for_target_task))
        print(f"\nScan complete. Found {len(episodes_to_load)} episodes for task {TARGET_TASK_INDEX}.")
        print(f"Episode indices: {episodes_to_load}")
        create_subset_dataset(TARGET_REPO_ID, episodes_to_load, source_metadata)
    print("\nScript finished.")