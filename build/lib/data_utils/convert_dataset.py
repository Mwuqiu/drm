import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import logging
import os
import shutil

# Configure logging to show progress
logging.basicConfig(level=logging.INFO)

def convert_dataset(original_repo_id: str, new_repo_id: str):
    """
    Loads a LeRobotDataset, duplicates the 'action' column to a new 'actions' column,
    and saves it as a new dataset.
    """
    # 1. Load the original dataset
    logging.info(f"Loading original dataset: {original_repo_id}...")
    try:
        original_dataset = LeRobotDataset(original_repo_id)
        logging.info("Original dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load original dataset: {e}")
        logging.error("Please ensure you are logged into Hugging Face CLI (`huggingface-cli login`) and the dataset exists.")
        return

    # 2. Define the features for the new dataset
    new_features = original_dataset.features.copy()
    new_features['actions'] = new_features['action'].copy()

    # 3. Create a new empty dataset
    logging.info(f"Creating new dataset: {new_repo_id}...")
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=original_dataset.fps,
        features=new_features,
        robot_type=original_dataset.meta.robot_type,
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )
    logging.info("New dataset created successfully.")

    # 4. Iterate through the original dataset, process, and add to the new one
    logging.info("Starting frame-by-frame copy and conversion...")
    
    current_episode_index = -1
    
    # <<< FIX 1: Define metadata keys to exclude from the `frame` dict.
    # The `add_frame` function handles these internally.
    keys_to_exclude = {'frame_index', 'index', 'task_index', 'timestamp', 'episode_index', 'task'}

    for i in tqdm(range(len(original_dataset)), desc="Processing frames"):
        item = original_dataset[i]
        episode_index = item['episode_index'].item()

        if episode_index != current_episode_index:
            if current_episode_index != -1:
                logging.info(f"Saving Episode {current_episode_index}...")
                new_dataset.save_episode()
                logging.info(f"Episode {current_episode_index} saved.")
            current_episode_index = episode_index

        frame_to_add = {}
        for key, value in item.items():
            # Skip keys that are not part of the raw frame data
            if key in keys_to_exclude or key not in new_dataset.features:
                continue

            # <<< FIX 2: Correct the image shape for image features.
            if new_dataset.features[key]["dtype"] == "image" and isinstance(value, torch.Tensor):
                # Convert from PyTorch's (C, H, W) to the expected (H, W, C) for saving
                frame_to_add[key] = value.permute(1, 2, 0)
            else:
                frame_to_add[key] = value

        # The crucial step: copy action data to actions
        frame_to_add['actions'] = item['action']
        
        task = item['task']
        timestamp = item['timestamp'].item()

        new_dataset.add_frame(frame_to_add, task=task, timestamp=timestamp)

    # 5. Save the very last episode
    if len(original_dataset) > 0 and new_dataset.episode_buffer["size"] > 0:
        logging.info(f"Saving final Episode {current_episode_index}...")
        new_dataset.save_episode()
        logging.info(f"Episode {current_episode_index} saved.")

    logging.info("All data processing and saving is complete!")
    logging.info(f"The new dataset is saved locally at: {new_dataset.root}")

if __name__ == '__main__':
    ORIGINAL_REPO_ID = "libero_elegant_1013/libero_elegant"
    NEW_REPO_ID = "libero_elegant_1013/libero_elegant_pi"

    home_dir = os.path.expanduser('~')
    new_dataset_path = os.path.join(home_dir, '.cache/huggingface/lerobot', NEW_REPO_ID)
    
    if os.path.exists(new_dataset_path):
        print(f"Warning: The destination directory '{new_dataset_path}' already exists.")
        choice = input("Do you want to delete it and start over? (y/n): ").lower()
        if choice == 'y':
            shutil.rmtree(new_dataset_path)
            print("Existing directory removed.")
            convert_dataset(ORIGINAL_REPO_ID, NEW_REPO_ID)
        else:
            print("Operation cancelled.")
    else:
        convert_dataset(ORIGINAL_REPO_ID, NEW_REPO_ID)