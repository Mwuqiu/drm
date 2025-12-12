import logging
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.common.constants import HF_LEROBOT_HOME

def extract_episodes(source_repo_id: str, target_repo_id: str, num_episodes: int):
    """
    Extracts the first N episodes from a source dataset and saves them to a new target dataset.
    (Your original code is here for reference)
    """
    # ... (original code remains unchanged) ...
    print("\n" + "=" * 80)
    print(f"🚀 STARTING EXTRACTION JOB (FIRST {num_episodes})")
    print(f"   Source: '{source_repo_id}'")
    print(f"   Target: '{target_repo_id}'")
    print(f"   Episodes to extract: {num_episodes}")
    print("=" * 80)

    target_path = HF_LEROBOT_HOME / target_repo_id
    if target_path.exists():
        print(f"⚠️  Warning: Target directory '{target_path}' already exists. Skipping this job.")
        return

    print(f"Loading source dataset '{source_repo_id}'...")
    try:
        source_ds = LeRobotDataset(repo_id=source_repo_id)
        source_meta = source_ds.meta
    except Exception as e:
        print(f"❌ Error loading source dataset: {e}")
        return
        
    print(f"Source dataset loaded with a total of {source_meta.total_episodes} episodes.")

    if source_meta.total_episodes < num_episodes:
        print(f"⚠️  Warning: Source dataset only has {source_meta.total_episodes} episodes, which is less than the requested {num_episodes}.")
        print(f"         Proceeding to extract all {source_meta.total_episodes} episodes.")
        num_episodes = source_meta.total_episodes

    print(f"Initializing new target dataset '{target_repo_id}'...")
    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_meta.fps,
        features=source_meta.features,
        robot_type=source_meta.robot_type,
        use_videos=(source_meta.video_path is not None),
        image_writer_threads=4,
    )

    current_source_ep_idx = -1
    episodes_saved_count = 0
    
    for frame_data in source_ds.hf_dataset:
        ep_idx = frame_data["episode_index"].item()

        if ep_idx != current_source_ep_idx:
            if current_source_ep_idx != -1:
                target_ds.save_episode()
                episodes_saved_count += 1
                print(f"  -> Saved episode {current_source_ep_idx}. Total saved: {episodes_saved_count}")

            if episodes_saved_count >= num_episodes:
                print(f"Reached the target of {num_episodes} episodes. Stopping extraction.")
                break
            
            current_source_ep_idx = ep_idx
            print(f"Processing source episode {current_source_ep_idx}...")

        clean_frame = {}
        for key, value in frame_data.items():
            if key in DEFAULT_FEATURES or key not in target_ds.features:
                continue
            
            if hasattr(value, "ndim") and value.ndim == 3 and key.startswith("observation.image"):
                clean_frame[key] = value.permute(1, 2, 0).numpy()
            elif hasattr(value, "numpy"):
                numpy_value = value.numpy()
                if numpy_value.shape == ():
                    clean_frame[key] = numpy_value.reshape(1)
                else:
                    clean_frame[key] = numpy_value
            else:
                clean_frame[key] = value

        task_string = source_meta.tasks[frame_data["task_index"].item()]
        target_ds.add_frame(clean_frame, task=task_string)

    if episodes_saved_count < num_episodes and target_ds.episode_buffer["size"] > 0:
        target_ds.save_episode()
        episodes_saved_count += 1
        print(f"  -> Saved final episode {current_source_ep_idx}. Total saved: {episodes_saved_count}")

    print("-" * 70)
    print(f"✅ Extraction complete for '{target_repo_id}'!")
    print(f"Final dataset has {target_ds.meta.total_episodes} episodes.")
    print("-" * 70)

def extract_last_episodes(source_repo_id: str, target_repo_id: str, num_episodes: int):
    """
    Extracts the last N episodes from a source dataset and saves them to a new target dataset.

    Args:
        source_repo_id (str): The repository ID of the source dataset.
        target_repo_id (str): The repository ID for the new dataset to be created.
        num_episodes (int): The number of episodes to extract from the end of the source dataset.
    """
    print("\n" + "=" * 80)
    print(f"🚀 STARTING EXTRACTION JOB (LAST {num_episodes})")
    print(f"   Source: '{source_repo_id}'")
    print(f"   Target: '{target_repo_id}'")
    print(f"   Episodes to extract: {num_episodes}")
    print("=" * 80)

    target_path = HF_LEROBOT_HOME / target_repo_id
    if target_path.exists():
        print(f"⚠️  Warning: Target directory '{target_path}' already exists. Skipping this job.")
        print("Please remove it or choose a different name if you want to re-run this extraction.")
        return

    print(f"Loading source dataset '{source_repo_id}'...")
    try:
        source_ds = LeRobotDataset(repo_id=source_repo_id)
        source_meta = source_ds.meta
    except Exception as e:
        print(f"❌ Error loading source dataset: {e}")
        return
        
    total_episodes = source_meta.total_episodes
    print(f"Source dataset loaded with a total of {total_episodes} episodes.")

    if total_episodes < num_episodes:
        print(f"⚠️  Warning: Source dataset only has {total_episodes} episodes, which is less than the requested {num_episodes}.")
        print(f"         Proceeding to extract all {total_episodes} episodes.")
        num_episodes = total_episodes

    # <<< CHANGE 1: Calculate the starting episode index
    start_episode_index = total_episodes - num_episodes
    print(f"Will start extracting from episode index: {start_episode_index}")

    print(f"Initializing new target dataset '{target_repo_id}'...")
    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_meta.fps,
        features=source_meta.features,
        robot_type=source_meta.robot_type,
        use_videos=(source_meta.video_path is not None),
        image_writer_threads=4,
    )
    print("Empty target dataset created.")

    current_source_ep_idx = -1
    episodes_saved_count = 0
    
    for frame_data in source_ds.hf_dataset:
        ep_idx = frame_data["episode_index"].item()

        # <<< CHANGE 2: Skip all episodes before our desired starting point
        if ep_idx < start_episode_index:
            continue

        if ep_idx != current_source_ep_idx:
            if current_source_ep_idx != -1:
                target_ds.save_episode()
                episodes_saved_count += 1
                print(f"  -> Saved episode {current_source_ep_idx}. Total saved: {episodes_saved_count}")

            # <<< CHANGE 3: The stopping condition is no longer needed
            # The loop will naturally end when the dataset finishes.
            
            current_source_ep_idx = ep_idx
            print(f"Processing source episode {current_source_ep_idx}...")

        clean_frame = {}
        for key, value in frame_data.items():
            if key in DEFAULT_FEATURES or key not in target_ds.features:
                continue
            
            if hasattr(value, "ndim") and value.ndim == 3 and key.startswith("observation.image"):
                clean_frame[key] = value.permute(1, 2, 0).numpy()
            elif hasattr(value, "numpy"):
                numpy_value = value.numpy()
                if numpy_value.shape == ():
                    clean_frame[key] = numpy_value.reshape(1)
                else:
                    clean_frame[key] = numpy_value
            else:
                clean_frame[key] = value

        task_string = source_meta.tasks[frame_data["task_index"].item()]
        target_ds.add_frame(clean_frame, task=task_string)

    if target_ds.episode_buffer["size"] > 0:
        target_ds.save_episode()
        episodes_saved_count += 1
        print(f"  -> Saved final episode {current_source_ep_idx}. Total saved: {episodes_saved_count}")

    print("-" * 70)
    print(f"✅ Extraction complete for '{target_repo_id}'!")
    print(f"Final dataset has {target_ds.meta.total_episodes} episodes.")
    print(f"It is ready locally at: {target_ds.root.resolve()}")
    print("-" * 70)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    EXTRACTION_JOBS = {
        "libero_elegant_1013/task_1": "libero_elegant_1018/task_1",
    }

    NUM_EPISODES_TO_EXTRACT = 10
    for source_id, target_id in EXTRACTION_JOBS.items():
        extract_episodes(
            source_repo_id=source_id, 
            target_repo_id=target_id, 
            num_episodes=NUM_EPISODES_TO_EXTRACT
        )

    print("\n" + "#" * 80)
    print("🎉 All extraction jobs have been completed! 🎉")
    print("#" * 80)