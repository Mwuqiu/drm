import json
import shutil
from pathlib import Path

def fix_episodes_stats_file(dataset_path: Path, mapping: dict):
    """
    一个独立的函数，用于修正指定数据集中 episodes_stats.jsonl 文件内的键名。

    Args:
        dataset_path (Path): 需要修正的数据集的根目录路径。
        mapping (dict): 定义了旧键名到新键名映射的字典。
    """
    print(f"--- Starting to fix stats file in: {dataset_path} ---")

    meta_path = dataset_path / "meta"
    stats_file_path = meta_path / "episodes_stats.jsonl"

    # 1. 检查文件是否存在
    if not stats_file_path.is_file():
        print(f"Error: '{stats_file_path}' not found. Please check the path.")
        return

    # 2. 创建一个备份，以防万一
    backup_path = stats_file_path.with_suffix(".jsonl.bak")
    print(f"Backing up original file to: {backup_path}")
    shutil.copy(stats_file_path, backup_path)

    # 3. 逐行读取、修改并存入内存
    modified_lines = []
    print("Reading and remapping keys in memory...")
    with open(stats_file_path, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            if 'stats' in data:
                original_stats = data['stats']
                new_stats = {}
                for old_key, value in original_stats.items():
                    new_key = mapping.get(old_key, old_key)
                    new_stats[new_key] = value
                data['stats'] = new_stats
            
            # 将修改后的 Python 字典转换回 JSON 字符串
            modified_lines.append(json.dumps(data))

    # 4. 将修改后的内容写回原文件
    print(f"Writing corrected content back to '{stats_file_path}'...")
    with open(stats_file_path, 'w') as f_out:
        for line in modified_lines:
            f_out.write(line + '\n')
    
    print("\n--- Fix complete! --- ")
    print(f"The file '{stats_file_path.name}' has been updated.")
    print(f"A backup of the original file is saved as '{backup_path.name}'.")


if __name__ == "__main__":
    # --- 您需要修改的部分 ---
    # 1. 定义键名映射 (请确保与您重构时使用的完全一致)
    KEY_MAPPING = {
        "state": "observation.state",
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
        "actions": "action",
    }

    # 2. 指定您已经重构过的数据集的路径
    # 例如：'tmp/libero_small_remapped_structured'
    DATASET_TO_FIX_PATH = Path("tmp/libero_small_remapped_structured")

    # --- 执行修正 ---
    fix_episodes_stats_file(DATASET_TO_FIX_PATH, KEY_MAPPING)