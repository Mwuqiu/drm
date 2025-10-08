import logging
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.common.constants import HF_LEROBOT_HOME

def extract_episodes(source_repo_id: str, target_repo_id: str, num_episodes: int):
    """
    Extracts the first N episodes from a source dataset and saves them to a new target dataset.

    Args:
        source_repo_id (str): The repository ID of the source dataset (e.g., "lerobot/aloha_sim_transfer_cube_human").
        target_repo_id (str): The repository ID for the new dataset to be created.
        num_episodes (int): The number of episodes to extract from the beginning of the source dataset.
    """
    print("\n" + "=" * 80)
    print(f"🚀 STARTING EXTRACTION JOB")
    print(f"   Source: '{source_repo_id}'")
    print(f"   Target: '{target_repo_id}'")
    print(f"   Episodes to extract: {num_episodes}")
    print("=" * 80)

    # 检查目标路径是否存在，避免覆盖
    target_path = HF_LEROBOT_HOME / target_repo_id
    if target_path.exists():
        print(f"⚠️  Warning: Target directory '{target_path}' already exists. Skipping this job.")
        print("Please remove it or choose a different name if you want to re-run this extraction.")
        return

    # 加载源数据集
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

    # 基于源数据集的元数据创建新的目标数据集
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
    
    # 遍历源数据集的帧
    for frame_data in source_ds.hf_dataset:
        ep_idx = frame_data["episode_index"].item()

        # 当 episode 索引改变时
        if ep_idx != current_source_ep_idx:
            # 如果不是第一帧，说明上一个 episode 已经完整，可以保存
            if current_source_ep_idx != -1:
                target_ds.save_episode()
                episodes_saved_count += 1
                print(f"  -> Saved episode {current_source_ep_idx}. Total saved: {episodes_saved_count}")

            # 检查是否已达到所需 episode 数量
            if episodes_saved_count >= num_episodes:
                print(f"Reached the target of {num_episodes} episodes. Stopping extraction.")
                break
            
            # 更新当前 episode 索引
            current_source_ep_idx = ep_idx
            print(f"Processing source episode {current_source_ep_idx}...")

        # 清理并准备帧数据 (与您提供的代码逻辑相同)
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

    # 循环结束后，保存最后一个正在处理的 episode (如果缓冲区有数据且未达到数量限制)
    if episodes_saved_count < num_episodes and target_ds.episode_buffer["size"] > 0:
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

    # 定义要处理的数据集列表
    # 字典的键是源数据集，值是生成的目标数据集的新名称
    EXTRACTION_JOBS = {
        "libero_elegant_0928/libero_object_task_0_append": "libero_elegant_0929/libero_object_task_0_append",
        "libero_elegant_0928/libero_object_task_2_append": "libero_elegant_0929/libero_object_task_2_append",
        "libero_elegant_0928/libero_object_task_3_append": "libero_elegant_0929/libero_object_task_3_append",
        "libero_elegant_0928/libero_object_task_4_append": "libero_elegant_0929/libero_object_task_4_append",
        "libero_elegant_0928/libero_object_task_5_append": "libero_elegant_0929/libero_object_task_5_append",
        "libero_elegant_0928/libero_object_task_7_append": "libero_elegant_0929/libero_object_task_7_append",
        "libero_elegant_0928/libero_object_task_9_append": "libero_elegant_0929/libero_object_task_9_append",
    }
    
    # 每个任务要提取的 episode 数量
    NUM_EPISODES_TO_EXTRACT = 3

    # 循环执行所有任务
    for source_id, target_id in EXTRACTION_JOBS.items():
        extract_episodes(
            source_repo_id=source_id, 
            target_repo_id=target_id, 
            num_episodes=NUM_EPISODES_TO_EXTRACT
        )

    print("\n" + "#" * 80)
    print("🎉 All extraction jobs have been completed! 🎉")
    print("#" * 80)