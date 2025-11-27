import logging
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from reward_dataset import RewardLeRobotDataset 

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inspect_and_generate_annotations(repo_id: str, output_path: str):
    """
    Inspects a LeRobot dataset to identify episodes containing positive rewards,
    and generates a JSON annotation file containing only the last three frames
    for those episodes.

    Args:
        repo_id (str): The repository ID of the dataset to inspect.
        output_path (str): The file path for the output JSON.
    """
    logging.info(f"Starting to load dataset: {repo_id}")
    try:
        dataset = RewardLeRobotDataset(repo_id=repo_id)
        logging.info(f"Dataset loaded successfully. Contains {dataset.num_episodes} episodes.")
    except Exception as e:
        logging.error(f"Failed to load dataset '{repo_id}'. Please ensure the repo ID is correct and the dataset exists. Error: {e}")
        return

    if dataset.num_episodes == 0:
        logging.warning("The dataset does not contain any episodes.")
        return

    final_annotations = {}
    episodes_with_rewards_indices = []

    logging.info("Checking rewards for each episode and calculating frame lengths...")
    
    for i in tqdm(range(dataset.num_episodes), desc="Checking Episodes"):
        ep_start_abs = dataset.episode_data_index["from"][i].item()
        ep_end_abs = dataset.episode_data_index["to"][i].item()
        
        found_positive_reward = False
        # Scan frames in the episode for any positive reward
        for frame_idx in range(ep_start_abs, ep_end_abs):
            reward_value = dataset[frame_idx]['reward'].item()
            if reward_value > 0:
                found_positive_reward = True
                break
        
        if found_positive_reward:
            episodes_with_rewards_indices.append(i)
            
            episode_length = ep_end_abs - ep_start_abs
            
            # Safety check: episode must be long enough to extract the last 3 frames
            if episode_length < 3:
                tqdm.write(f"Warning: Episode {i} has rewards but fewer than 3 frames ({episode_length}). Cannot generate annotations. Skipped.")
                continue
                
            start_frame = episode_length - 3
            end_frame = episode_length - 1
            
            # Store annotation: [start_index, end_index] relative to the episode
            final_annotations[str(i)] = [[start_frame, end_frame]]

    output_file = Path(output_path)
    
    print("\n" + "=" * 70)
    print("### Inspection and Generation Completed ###")
    print(f"Target Dataset: {repo_id}")
    print("=" * 70 + "\n")
    
    print(f"Total episodes with positive rewards found: {len(episodes_with_rewards_indices)}")
    print(episodes_with_rewards_indices if episodes_with_rewards_indices else "  None")
    
    if final_annotations:
        logging.info(f"Saving {len(final_annotations)} annotation entries to {output_file.resolve()}...")
        with output_file.open('w') as f:
            json.dump(final_annotations, f, indent=4)
        logging.info("File saved successfully!")
    else:
        logging.warning("No episodes with rewards were found, so no JSON file was created.")
        
    print("\n" + "=" * 70)

def main():
    """Main function to parse arguments and execute core logic."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Inspects a reward dataset and generates a JSON annotation file containing "
                    "the last three frames for episodes that contain positive rewards.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--repo_id', 
        type=str, 
        required=True, 
        help='The Hugging Face repository ID of the dataset to inspect.\n'
             'Example: "libero_elegant_1001/task_3_reward"'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='The file path for the output JSON.\n'
             'Example: "generated_annotations.json"'
    )

    args = parser.parse_args()

    inspect_and_generate_annotations(args.repo_id, args.output)

if __name__ == "__main__":
    main()