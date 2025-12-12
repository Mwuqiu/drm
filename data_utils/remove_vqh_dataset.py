import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import logging
import os
import shutil
from datasets import load_dataset
import numpy as np

logging.basicConfig(level=logging.INFO)

def convert_dataset(original_repo_id: str, new_repo_id: str):
    logging.info(f"Loading original dataset (source): {original_repo_id}...")
    try:
        original_dataset = LeRobotDataset(original_repo_id)
        logging.info("Original dataset loaded successfully.")
        
        path = str(original_dataset.root / "data")
        logging.info(f"Loading raw parquet files from: {path}...")
        hf_data_no_transform = load_dataset(
            "parquet", 
            data_dir=path, 
            split="train", 
            keep_in_memory=True
        )
        logging.info("Raw parquet dataset loaded successfully.")
        
    except Exception as e:
        logging.error(f"Failed to load original dataset: {e}")
        return

    keys_to_remove = []
    
    if "next_action" in original_dataset.features:
        keys_to_remove.append("next_action")

    for key in original_dataset.features.keys():
        if key.endswith(".vqh"):
            keys_to_remove.append(key)

    logging.info(f"Detected {len(keys_to_remove)} keys to remove: {keys_to_remove}")
    logging.info("Defining features for the new dataset...")
    new_features = original_dataset.features.copy()

    for key in keys_to_remove:
        if key in new_features:
            del new_features[key]

    logging.info(f"Creating new dataset: {new_repo_id}...")
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=original_dataset.fps,
        features=new_features,
        robot_type=original_dataset.meta.robot_type,
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )
    logging.info("New dataset created successfully.")

    logging.info("Starting frame-by-frame copy (dropping keys)...")
    
    current_episode_index = -1
    
    system_keys_to_exclude = {'frame_index', 'index', 'task_index', 'timestamp', 'episode_index', 'task'}

    for i in tqdm(range(len(hf_data_no_transform)), desc="Processing frames"):
        item = hf_data_no_transform[i]
        episode_index = item['episode_index']
        if episode_index != current_episode_index:
            if current_episode_index != -1:
                new_dataset.save_episode()
            current_episode_index = episode_index

        frame_to_add = item.copy()
        for key in keys_to_remove:
            if key in frame_to_add:
                del frame_to_add[key]
                
        for key in system_keys_to_exclude:
            if key in frame_to_add:
                del frame_to_add[key]
        
        task = original_dataset.meta.tasks[item['task_index']]
        timestamp = item['timestamp']
        
        try:
            if 'action' in frame_to_add:
                frame_to_add['action'] = np.array(frame_to_add['action'], dtype=np.float32)
            if 'observation.state' in frame_to_add:
                frame_to_add['observation.state'] = np.array(frame_to_add['observation.state'], dtype=np.float32)
            
            for k, v in frame_to_add.items():
                if isinstance(v, list) and k not in ['task']:
                     if "images" not in k: 
                        frame_to_add[k] = np.array(v, dtype=np.float32)

            frame_to_add['reward'] = np.array([frame_to_add['reward']], dtype=np.float32)
            frame_to_add['tdm'] = np.array([frame_to_add['tdm']], dtype=np.bool_)
            frame_to_add['mc_returns'] = np.array([frame_to_add['mc_returns']], dtype=np.float32)
            frame_to_add['done'] = np.array([frame_to_add['done']], dtype=np.bool_)

        except Exception as e:
            logging.error(f"Error converting types for item {i}: {e}")
            raise e

        new_dataset.add_frame(frame_to_add, task=task, timestamp=timestamp)

    if len(original_dataset) > 0 and new_dataset.episode_buffer["size"] > 0:
        logging.info(f"Saving final Episode {current_episode_index}...")
        new_dataset.save_episode()

    logging.info("All data processing and saving is complete!")
    logging.info(f"The cleaned dataset is saved locally at: {new_dataset.root}")

if __name__ == '__main__':
    ORIGINAL_REPO_ID = "libero_elegant/libero_elegant_vqh_reward"
    NEW_REPO_ID = "libero_elegant/libero_elegant_reward"

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