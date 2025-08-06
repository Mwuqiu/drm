import logging
import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.common.constants import HF_LEROBOT_HOME

def append_dataset(target_ds: LeRobotDataset, source_repo_id: str):
    """
    加载一个源数据集，并将其所有 episodes 追加到目标数据集中。

    Args:
        target_ds: 已经创建的目标 LeRobotDataset 对象。
        source_repo_id: 要追加的数据集的 Hugging Face repo ID。
    """
    print("-" * 70)
    print(f"Loading source dataset '{source_repo_id}' to append...")
    
    # 高效地加载源数据集的所有帧
    source_ds = LeRobotDataset(repo_id=source_repo_id)
    source_meta = source_ds.meta
    print(f"Source dataset '{source_repo_id}' loaded with {source_meta.total_episodes} episodes.")

    # 用于跟踪 episode 变化的变量
    # 注意：我们只关心源数据集内的 episode index 变化
    current_source_ep_idx = -1
    
    # 遍历源数据集的每一帧
    for frame_data in source_ds.hf_dataset:
        ep_idx = frame_data["episode_index"].item()

        # 检查是否是一个新的 episode
        if ep_idx != current_source_ep_idx:
            # 如果不是第一次，就保存上一个 episode
            if current_source_ep_idx != -1:
                target_ds.save_episode()
                print(f"  -> Appended episode. New total in target: {target_ds.meta.total_episodes}")

            current_source_ep_idx = ep_idx
            print(f"Processing source episode {current_source_ep_idx} from '{source_repo_id}'...")
        
        # 准备要添加到目标数据集的帧数据
        clean_frame = {}
        for key, value in frame_data.items():
            # 过滤掉数据集自动生成的默认键
            if key in DEFAULT_FEATURES:
                continue
            
            # 将图像从 (C, H, W) 转换为 (H, W, C) 以便保存
            if hasattr(value, "ndim") and value.ndim == 3 and key.startswith("observation.image"):
                clean_frame[key] = value.permute(1, 2, 0).numpy()
            else:
                clean_frame[key] = value

        # 获取任务描述字符串
        task_string = source_meta.tasks[frame_data["task_index"].item()]
        target_ds.add_frame(clean_frame, task=task_string)

    # 确保保存最后一个 episode
    if target_ds.episode_buffer["size"] > 0:
        target_ds.save_episode()
        print(f"  -> Appended episode. New total in target: {target_ds.meta.total_episodes}")
    
    print(f"Finished appending all episodes from '{source_repo_id}'.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- 配置 ---
    # 1. 源数据集 (一个是成功样例，一个是失败样例)
    SOURCE_SUCCESS_REPO_ID = "physical-intelligence/libero_spatial_task_9"
    SOURCE_FAILURE_REPO_ID = "physical-intelligence/libero_spatial_task_nine_failures"
    
    # 2. 目标数据集
    TARGET_REPO_ID = "physical-intelligence/libero_spatial_task_9_new_datasets"
    # --- 结束配置 ---

    # 安全检查：如果目标数据集已存在，则中止以防意外覆盖
    target_path = HF_LEROBOT_HOME / TARGET_REPO_ID
    if target_path.exists():
        print(f"Error: Target directory '{target_path}' already exists.")
        print("Please remove it or choose a different TARGET_REPO_ID before running.")
        exit()

    # --- 步骤 1: 创建空的目标数据集 ---
    print(f"Initializing new target dataset '{TARGET_REPO_ID}'...")
    # 我们使用第一个源数据集的元数据来初始化目标数据集
    # 假设两个源数据集的元数据（fps, 机器人类型等）是兼容的
    source_meta_for_init = LeRobotDatasetMetadata(repo_id=SOURCE_SUCCESS_REPO_ID)
    
    target_ds = LeRobotDataset.create(
        repo_id=TARGET_REPO_ID,
        fps=source_meta_for_init.fps,
        features=source_meta_for_init.features,
        robot_type=source_meta_for_init.robot_type,
        use_videos=(source_meta_for_init.video_path is not None),
        image_writer_threads=4, # 使用多线程加速图片写入
    )
    print("Empty target dataset created.")

    # --- 步骤 2: 追加第一个数据集 (成功样例) ---
    append_dataset(target_ds, SOURCE_SUCCESS_REPO_ID)
    
    # --- 步骤 3: 追加第二个数据集 (失败样例) ---
    append_dataset(target_ds, SOURCE_FAILURE_REPO_ID)
    
    # --- 步骤 4: 完成 ---
    print("-" * 70)
    print("\nMerge complete!")
    print(f"Final dataset '{TARGET_REPO_ID}' has {target_ds.meta.total_episodes} episodes.")
    print(f"It is ready locally at: {target_ds.root.resolve()}")
    print("\nTo upload it to the Hugging Face Hub, you can add the following line to the script:")
    print("# target_ds.push_to_hub()")
    print("-" * 70)