import logging
import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np # <-- 确保导入 numpy

def append_dataset(target_ds: LeRobotDataset, source_repo_id: str):
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
            
            # >>> START: 修改部分 <<<
            if hasattr(value, "ndim") and value.ndim == 3 and key.startswith("observation.image"):
                # 正确处理图像数据: (C, H, W) -> (H, W, C)
                clean_frame[key] = value.permute(1, 2, 0).numpy()
            elif hasattr(value, "numpy"):
                # 处理所有其他张量 (state, action, reward, etc.)
                numpy_value = value.numpy()
                # 检查是否为标量，如果是，则重塑为 (1,)
                if numpy_value.shape == ():
                    clean_frame[key] = numpy_value.reshape(1)
                else:
                    clean_frame[key] = numpy_value
            else:
                # 以防万一，处理非张量数据
                clean_frame[key] = value
            # >>> END: 修改部分 <<<

        task_string = source_meta.tasks[frame_data["task_index"].item()]
        target_ds.add_frame(clean_frame, task=task_string)

    if target_ds.episode_buffer["size"] > 0:
        target_ds.save_episode()
        print(f"  -> Appended episode. New total in target: {target_ds.meta.total_episodes}")
    
    print(f"Finished appending all episodes from '{source_repo_id}'.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    SOURCE_REPO_IDS = [
        "libero_elegant/task_0_elegant",
        "libero_elegant/task_1_elegant",
        "libero_elegant/task_2_elegant",
        "libero_elegant/task_3_elegant",
        "libero_elegant/task_4_elegant",
        "libero_elegant/task_5_elegant",
        "libero_elegant/task_6_elegant",
        "libero_elegant/task_7_elegant",
        "libero_elegant/task_8_elegant",
        "libero_elegant/task_9_elegant",
    ]
    
    TARGET_REPO_ID = "libero_elegant/elegant_only"

    target_path = HF_LEROBOT_HOME / TARGET_REPO_ID
    if target_path.exists():
        print(f"Error: Target directory '{target_path}' already exists.")
        print("Please remove it or choose a different TARGET_REPO_ID before running.")
        exit()

    if not SOURCE_REPO_IDS:
        print("Error: The list of source datasets is empty. Nothing to merge.")
        exit()

    print("Loading metadata from all source datasets...")
    source_metas = [LeRobotDatasetMetadata(repo_id=repo_id) for repo_id in SOURCE_REPO_IDS]

    print("Calculating the intersection of features across all datasets...")
    common_feature_keys = set(source_metas[0].features.keys())

    for meta in source_metas[1:]:
        common_feature_keys &= set(meta.features.keys())
    
    base_meta = source_metas[0]
    common_features = {key: base_meta.features[key] for key in common_feature_keys}
    print(f"Found {len(common_features)} common features across all {len(SOURCE_REPO_IDS)} datasets.")

    print(f"Initializing new target dataset '{TARGET_REPO_ID}' with common features...")
    target_ds = LeRobotDataset.create(
        repo_id=TARGET_REPO_ID,
        fps=base_meta.fps,                 
        features=common_features,          
        robot_type=base_meta.robot_type,    
        use_videos=(base_meta.video_path is not None),
        image_writer_threads=4,
    )

    print("Empty target dataset created.")

    for repo_id in SOURCE_REPO_IDS:
        append_dataset(target_ds, repo_id)
    
    print("-" * 70)
    print("\nMerge complete! ✨")
    print(f"Final dataset '{TARGET_REPO_ID}' has {target_ds.meta.total_episodes} episodes.")
    print(f"It is ready locally at: {target_ds.root.resolve()}")
    print("\nTo upload it to the Hugging Face Hub, you can add the following line to the script:")
    print("# target_ds.push_to_hub()")
    print("-" * 70)