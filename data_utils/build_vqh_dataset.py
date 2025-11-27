import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import logging
import os
import shutil
from datasets import load_dataset
import numpy as np
logging.basicConfig(level=logging.INFO)

NEXT_OBS_OFFSET = 10 

def convert_dataset(original_repo_id: str, new_repo_id: str):
    logging.info(f"Loading original dataset: {original_repo_id}...")
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
        
        ep_ends = original_dataset.episode_data_index["to"]
        
    except Exception as e:
        logging.error(f"Failed to load original dataset: {e}")
        logging.error("Please ensure you are logged into Hugging Face CLI (`huggingface-cli login`) and the dataset exists.")
        return

    logging.info("Defining features for the new dataset...")
    new_features = original_dataset.features.copy()

    if 'observation.state.joint' in new_features:
        del new_features['observation.state.joint']
        logging.info("Ignoring and removing feature: observation.state.joint")        

    if 'action' in new_features:
        new_features['next_action'] = new_features['action'].copy()
        
    if 'observation.state' in new_features:
        new_features['observation.state.vqh'] = new_features['observation.state'].copy()

    for key in original_dataset.meta.image_keys:
        if key in new_features:
            new_features[f"{key}.vqh"] = new_features[key].copy()
            logging.info(f"Adding feature: {key}.vqh")

    logging.info(f"Creating new dataset: {new_repo_id}...")
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=original_dataset.fps,
        features=new_features,
        robot_type=original_dataset.meta.robot_type,
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )
    logging.info("New dataset created successfully.")

    logging.info("Starting frame-by-frame copy and VQH pre-computation...")
    
    current_episode_index = -1
    
    keys_to_exclude = {'frame_index', 'index', 'task_index', 'timestamp', 'episode_index', 'task', 'observation.state.joint'}

    for i in tqdm(range(len(hf_data_no_transform)), desc="Processing frames"):
        item = hf_data_no_transform[i]
        episode_index = item['episode_index']

        if episode_index != current_episode_index:
            if current_episode_index != -1:
                logging.info(f"Saving Episode {current_episode_index}...")
                new_dataset.save_episode()
                logging.info(f"Episode {current_episode_index} saved.")
            current_episode_index = episode_index

        frame_to_add = item.copy()

        ep_end = ep_ends[episode_index].item()
        vqh_idx = min(i + NEXT_OBS_OFFSET, ep_end - 1)
        vqh_item = hf_data_no_transform[vqh_idx]

        frame_to_add['next_action'] = vqh_item['action']
        frame_to_add['observation.state.vqh'] = vqh_item['observation.state']
        
        for key in original_dataset.meta.image_keys:
            if key in vqh_item:
                frame_to_add[f"{key}.vqh"] = vqh_item[key]

        for key in keys_to_exclude:
            if key in frame_to_add:
                del frame_to_add[key]
        
        task = original_dataset.meta.tasks[item['task_index']]
        timestamp = item['timestamp']
        
        try:
            frame_to_add['action'] = np.array(frame_to_add['action'], dtype=np.float32)
            frame_to_add['observation.state'] = np.array(frame_to_add['observation.state'], dtype=np.float32)
            frame_to_add['next_action'] = np.array(frame_to_add['next_action'], dtype=np.float32)
            frame_to_add['observation.state.vqh'] = np.array(frame_to_add['observation.state.vqh'], dtype=np.float32)

            frame_to_add['reward'] = np.array([frame_to_add['reward']], dtype=np.float32)
            frame_to_add['tdm'] = np.array([frame_to_add['tdm']], dtype=np.bool_)
            frame_to_add['mc_returns'] = np.array([frame_to_add['mc_returns']], dtype=np.float32)
            frame_to_add['done'] = np.array([frame_to_add['done']], dtype=np.bool_)

        except Exception as e:
            logging.error(f"Error converting types for item {i}: {e}")
            logging.error(f"Action type: {type(frame_to_add['action'])}")
            logging.error(f"State type: {type(frame_to_add['observation.state'])}")
            raise e

        new_dataset.add_frame(frame_to_add, task=task, timestamp=timestamp)

    if len(original_dataset) > 0 and new_dataset.episode_buffer["size"] > 0:
        logging.info(f"Saving final Episode {current_episode_index}...")
        new_dataset.save_episode()
        logging.info(f"Episode {current_episode_index} saved.")

    logging.info("All data processing and saving is complete!")
    logging.info(f"The new dataset is saved locally at: {new_dataset.root}")

if __name__ == '__main__':
    ORIGINAL_REPO_ID = "libero_elegant_1001/task_1_segment_reward"
    NEW_REPO_ID = "libero_elegant_1018/task_1_segment_reward"

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
