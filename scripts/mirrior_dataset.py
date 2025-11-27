import logging
import shutil
from pathlib import Path
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME

def mirror_images_only(source_repo_id: str, target_repo_id: str):
    """
    Reads a LeRobotDataset, applies a horizontal mirror transformation ONLY to the images,
    and saves it to a new dataset. All other data (action, state, etc.) remains unchanged.
    """
    target_path = HF_LEROBOT_HOME / target_repo_id
    if target_path.exists():
        logging.error(f"错误: 目标路径 '{target_path}' 已存在。")
        return

    print("-" * 70)
    print(f"正在加载源数据集: '{source_repo_id}'")
    source_ds = LeRobotDataset(source_repo_id)
    source_meta = source_ds.meta
    print(f"源数据集加载成功，包含 {source_meta.total_episodes} 个 episodes。")

    print(f"正在初始化目标数据集: '{target_repo_id}'")
    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_meta.fps,
        features=source_meta.features,
        robot_type=source_meta.robot_type,
        use_videos=(source_meta.video_path is not None),
        image_writer_threads=8,
    )
    print("空的目标数据集已创建。")
    print("-" * 70)

    current_source_ep_idx = -1
    expected_features = set(target_ds.features.keys())
    
    IGNORE_KEYS = {'episode_index', 'index', 'frame_index', 'timestamp', 'task_index'}

    for frame_data in source_ds.hf_dataset:
        ep_idx = frame_data["episode_index"].item()

        if ep_idx != current_source_ep_idx:
            if current_source_ep_idx != -1:
                target_ds.save_episode()
                print(f"  -> 已保存 Episode {current_source_ep_idx}。")
            current_source_ep_idx = ep_idx
            print(f"正在处理源 Episode {current_source_ep_idx}...")

        transformed_frame = {}
        for key in expected_features:
            # 跳过要忽略的元数据键
            if key in IGNORE_KEYS:
                continue

            value = frame_data[key]
            numpy_value = value.numpy() if hasattr(value, "numpy") else value

            if key.startswith("observation.images"):
                if numpy_value.shape[0] == 3:
                    hwc_image = numpy_value.transpose(1, 2, 0)
                else:
                    hwc_image = numpy_value
                transformed_frame[key] = np.fliplr(hwc_image)
            else:
                # transformed_frame[key] = numpy_value
                transformed_frame[key] = np.atleast_1d(numpy_value)
        
        task_string = source_meta.tasks[frame_data["task_index"].item()]
        target_ds.add_frame(transformed_frame, task=task_string)

    if target_ds.episode_buffer["size"] > 0:
        target_ds.save_episode()
        print(f"  -> 已保存 Episode {current_source_ep_idx}。")

    print("-" * 70)
    print(f"\n仅镜像图像的处理完成! ✨")
    print(f"最终数据集 '{target_repo_id}' 已在本地创建。")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    SOURCE_REPO_ID = "libero_elegant_1001/task_4_segment_reward"
    TARGET_REPO_ID = "libero_elegant_1018/task_4_segment_reward"
    mirror_images_only(SOURCE_REPO_ID, TARGET_REPO_ID)