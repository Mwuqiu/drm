import os
import logging
import random
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import DEFAULT_FEATURES

SOURCE_REPO_ID = "physical-intelligence/libero"

DATASET_MAPPING = {
    "tasks_0-9": "physical-intelligence/libero_long",
    "tasks_10-19": "physical-intelligence/libero_goal",
    "tasks_20-29": "physical-intelligence/libero_object",
    "tasks_30-39": "physical-intelligence/libero_spatial",
}


def create_subset_dataset(target_repo_id: str, episodes_to_load: list[int], source_meta: LeRobotDatasetMetadata):
    print("-" * 70)
    print(f"Creating new dataset '{target_repo_id}' with {len(episodes_to_load)} episodes.")

    print(f"Directly loading the {len(episodes_to_load)} required episodes...")
    source_ds_filtered = LeRobotDataset(
        repo_id=source_meta.repo_id,
        episodes=episodes_to_load,
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

    for frame_data in source_ds_filtered.hf_dataset:
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
            if hasattr(value, "ndim") and value.ndim == 3:
                clean_frame[key] = value.permute(1, 2, 0).numpy()
            else:
                clean_frame[key] = value

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

    print("Using pre-computed episode lists to skip the scan step.")
    episode_sets = {
        "tasks_0-9": set(range(0, 379)),  # Corresponds to the 379 episodes you found
        "tasks_10-19": set(range(379, 807)),  # Corresponds to the 428 episodes (807-379)
        "tasks_20-29": set(range(807, 1261)),  # Corresponds to the 454 episodes (1261-807)
        "tasks_30-39": set(range(1261, 1693)),  # Corresponds to the 432 episodes (1693-1261)
    }

    source_metadata = LeRobotDatasetMetadata(repo_id=SOURCE_REPO_ID)

    for range_name, episodes in episode_sets.items():
        if range_name not in DATASET_MAPPING:
            continue

        if not episodes:
            print(f"No episodes found for range '{range_name}', skipping.")
            continue

        target_repo_id = DATASET_MAPPING[range_name]
        create_subset_dataset(target_repo_id, list(episodes), source_metadata)

    print("\nAll tasks complete.")
