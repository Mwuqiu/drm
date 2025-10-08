import logging
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.common.constants import HF_LEROBOT_HOME

def append_dataset(target_ds: LeRobotDataset, source_repo_id: str):
    """
    Appends all episodes from a source dataset to a target dataset.
    """
    print("-" * 70)
    print(f"Loading source dataset '{source_repo_id}' to append...")
    
    source_ds = LeRobotDataset(repo_id=source_repo_id)
    source_meta = source_ds.meta
    print(f"Source dataset '{source_repo_id}' loaded with {source_meta.total_episodes} episodes.")

    current_source_ep_idx = -1
    
    expected_features = set(target_ds.features.keys())

    for frame_data in source_ds.hf_dataset:
        ep_idx = frame_data["episode_index"].item()

        if ep_idx != current_source_ep_idx:
            if current_source_ep_idx != -1:
                target_ds.save_episode()
                print(f"  -> Appended episode. New total in target: {target_ds.meta.total_episodes}")

            current_source_ep_idx = ep_idx
            print(f"Processing source episode {current_source_ep_idx} from '{source_repo_id}'...")
        
        clean_frame = {}
        for key, value in frame_data.items():
            if key in DEFAULT_FEATURES or key not in expected_features:
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
        print(f"  -> Appended episode. New total in target: {target_ds.meta.total_episodes}")
    
    print(f"Finished appending all episodes from '{source_repo_id}'.")


def run_merge_job(target_repo_id: str, source_repo_ids: list[str]):
    """
    Executes a single merge job: merging a list of source datasets into one target dataset.
    """
    print("\n" + "=" * 80)
    print(f"🚀 STARTING MERGE JOB FOR: '{target_repo_id}'")
    print("=" * 80)

    target_path = HF_LEROBOT_HOME / target_repo_id
    if target_path.exists():
        print(f"⚠️  Warning: Target directory '{target_path}' already exists. Skipping this job.")
        print("Please remove it or choose a different name if you want to re-run this merge.")
        return
        
    if not source_repo_ids:
        print("Error: The list of source datasets is empty. Nothing to merge.")
        return

    print("Loading metadata from all source datasets for this job...")
    source_metas = [LeRobotDatasetMetadata(repo_id=repo_id) for repo_id in source_repo_ids]

    print("Calculating the intersection of features across datasets...")
    common_feature_keys = set(source_metas[0].features.keys())
    for meta in source_metas[1:]:
        common_feature_keys &= set(meta.features.keys())
    
    base_meta = source_metas[0]
    common_features = {key: base_meta.features[key] for key in common_feature_keys}
    print(f"Found {len(common_features)} common features across {len(source_repo_ids)} datasets.")

    print(f"Initializing new target dataset '{target_repo_id}' with common features...")
    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=base_meta.fps,                 
        features=common_features,          
        robot_type=base_meta.robot_type,    
        use_videos=(base_meta.video_path is not None),
        image_writer_threads=4,
    )

    print("Empty target dataset created.")

    for repo_id in source_repo_ids:
        append_dataset(target_ds, repo_id)
    
    print("-" * 70)
    print(f"✅ Merge complete for '{target_repo_id}'!")
    print(f"Final dataset has {target_ds.meta.total_episodes} episodes.")
    print(f"It is ready locally at: {target_ds.root.resolve()}")
    print("-" * 70)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    MERGE_JOBS = {
            "libero_elegant_1001/libero_elegant_reward" : [
                "libero_elegant_1001/task_0_segment_reward",
                "libero_elegant_1001/task_1_segment_reward",
                "libero_elegant_1001/task_2_segment_reward",
                "libero_elegant_1001/task_3_segment_reward",
                "libero_elegant_1001/task_4_segment_reward",
                "libero_elegant_1001/task_5_segment_reward",
                "libero_elegant_1001/task_6_segment_reward",
                "libero_elegant_1001/task_7_segment_reward",            
            ] 
        # "libero_elegant_0929/libero_object_task_0": [
        #     "libero_elegant_0929/libero_object_task_0_append",
        #     "libero_elegant_0929/libero_object_task_0_origin",
        # ],
        # "libero_elegant_0929/libero_object_task_2": [
        #     "libero_elegant_0929/libero_object_task_2_append",
        #     "libero_elegant_0929/libero_object_task_2_origin",
        # ],
        # "libero_elegant_0929/libero_object_task_3": [
        #     "libero_elegant_0929/libero_object_task_3_append",
        #     "libero_elegant_0929/libero_object_task_3_origin",
        # ],
        # "libero_elegant_0929/libero_object_task_4": [
        #     "libero_elegant_0929/libero_object_task_4_append",
        #     "libero_elegant_0929/libero_object_task_4_origin",
        # ],
        # "libero_elegant_0929/libero_object_task_5": [
        #     "libero_elegant_0929/libero_object_task_5_append",
        #     "libero_elegant_0929/libero_object_task_5_origin",
        # ],
        # "libero_elegant_0929/libero_object_task_7": [
        #     "libero_elegant_0929/libero_object_task_7_append",
        #     "libero_elegant_0929/libero_object_task_7_origin",
        # ],
        # "libero_elegant_0929/libero_object_task_9": [
        #     "libero_elegant_0929/libero_object_task_9_append",
        #     "libero_elegant_0929/libero_object_task_9_origin",
        # ],
        # "libero_elegant_0929/libero_elegant_object_reward": [
        #     "libero_elegant_0929/libero_object_task_0_reward",
        #     "libero_elegant_0929/libero_object_task_4_reward",
        #     "libero_elegant_0929/libero_object_task_7_reward",
        # ],
    }

    for target_id, source_ids in MERGE_JOBS.items():
        run_merge_job(target_id, source_ids)

    print("\n" + "#" * 80)
    print("🎉 All merge jobs have been completed! 🎉")
    print("#" * 80)