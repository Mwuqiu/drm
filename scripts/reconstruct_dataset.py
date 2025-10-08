import logging
from reward_dataset import RewardLeRobotDataset

def reconstruct_segment_dataset():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_task = 6
    end_task = 7

    for task_num in range(start_task, end_task + 1):
        
        print("\n" + "#" * 70)
        print(f"### PROCESSING TASK {task_num} ###")
        print("#" * 70 + "\n")

        SOURCE_REPO_ID = f"libero_elegant_1006/task_{task_num}"
        TARGET_REPO_ID = f"libero_elegant_1006/task_{task_num}_binary_reward"
        ANNOTATIONS_JSON_PATH = f"task_annotations/libero_elegant_b/task_{task_num}.json"
        
        print("\n" + "=" * 50)
        logging.info(f"Loading source dataset: {SOURCE_REPO_ID}")
        try:
            source_dataset = RewardLeRobotDataset(repo_id=SOURCE_REPO_ID, video_backend='pyav')
            logging.info(f"Source dataset loaded with {source_dataset.num_episodes} episodes.")
        except Exception as e:
            logging.error(f"Could not load source dataset for task {task_num}. Skipping. Error: {e}")
            continue 
        print("=" * 50 + "\n")

        print("\n" + "=" * 50)
        logging.info(f"Starting labeling from segments JSON. Target repo: {TARGET_REPO_ID}")

        try:
            segment_labeled_dataset = source_dataset.process_and_label_from_segments(
                target_repo_id=TARGET_REPO_ID,
                annotations_path=ANNOTATIONS_JSON_PATH
            )
            logging.info(f"Finished creating segment-labeled dataset: {TARGET_REPO_ID}")
        except FileNotFoundError:
            logging.error(f"Annotation file not found for task {task_num} at '{ANNOTATIONS_JSON_PATH}'. Skipping.")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred during processing for task {task_num}. Skipping. Error: {e}", exc_info=True)
            continue
        print("=" * 50 + "\n")
        
        print("\n" + "=" * 50)
        logging.info(f"--- Loading the new segment-labeled dataset for inspection (Task {task_num}) ---")
        try:
            rewarded_dataset = RewardLeRobotDataset(repo_id=TARGET_REPO_ID) 
            logging.info(f"Successfully loaded the new dataset.")
            logging.info(f"Features of the new dataset: {list(rewarded_dataset.features.keys())}")

            if len(rewarded_dataset) > 0:
                logging.info("\nInspecting first episode for positive rewards...")
                found_reward = False

                ep_start_abs = rewarded_dataset.episode_data_index["from"][0].item()
                ep_end_abs = rewarded_dataset.episode_data_index["to"][0].item()
                
                for frame_idx in range(ep_start_abs, ep_end_abs):
                    sample = rewarded_dataset[frame_idx]
                    if sample['reward'].item() > 0:
                        logging.info(f"  SUCCESS: Found a positive reward at frame {frame_idx - ep_start_abs} (absolute index {frame_idx})!")
                        logging.info(f"  Reward value: {sample['reward'].item()}")
                        found_reward = True
                        break
                
                if not found_reward:
                    logging.warning("  No positive rewards found in the first episode. Please double-check annotations.")
            else:
                logging.warning("The created dataset is empty.")
        except Exception as e:
            logging.error(f"Could not load or inspect the newly created dataset for task {task_num}. Error: {e}", exc_info=True)
        print("=" * 50 + "\n")

    print("\n" + "#" * 70)
    print("### ALL TASKS PROCESSED! ###")
    print("#" * 70 + "\n")

if __name__ == "__main__":
    reconstruct_segment_dataset()