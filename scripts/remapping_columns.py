import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from datasets import Dataset

def remap_and_save_dataset_preserving_structure(repo_id, mapping, output_path):
    """
    Loads a LeRobot dataset, renames its feature keys, and saves it by preserving
    the original chunk and per-episode file structure.
    """
    print(f"--- Processing dataset: {repo_id} ---")
    output_path = Path(output_path)

    print("\nStep 1 & 2: Loading original dataset and renaming columns...")
    source_dataset = LeRobotDataset(repo_id)
    renamed_hf_dataset = source_dataset.hf_dataset.rename_columns(mapping)
    print("Dataset loaded and columns renamed successfully.")

    print("\nDisabling automatic tensor transform to read raw data...")
    renamed_hf_dataset.set_transform(lambda batch: batch)

    print("\nStep 3: Preparing output directories and metadata...")
    if output_path.exists():
        print(f"Warning: Output path '{output_path}' exists and will be overwritten.")
        shutil.rmtree(output_path)
    output_meta_path = output_path / "meta"
    output_meta_path.mkdir(parents=True, exist_ok=True)

    source_meta_path = source_dataset.root / "meta"
    info = source_dataset.meta.info
    info["features"] = {mapping.get(k, k): v for k, v in info.get("features", {}).items()}
    with open(output_meta_path / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print("Updated and saved info.json.")

    stats_file = "episodes_stats.jsonl"
    if (source_meta_path / stats_file).exists():
        remap_jsonl_stats_file(source_meta_path / stats_file, output_meta_path / stats_file, mapping)
        print("Remapped and saved episodes_stats.jsonl.")

    for filename in ["episodes.jsonl", "tasks.jsonl", "stats.json"]:  # stats.json 用于兼容旧版
        source_file = source_meta_path / filename
        if source_file.exists():
            shutil.copy(source_file, output_meta_path / filename)
            print(f"Copied {filename}.")

    print("\nStep 4 & 5: Streaming through the dataset and saving each episode...")
    
    current_episode_rows = []
    current_episode_index = 0 
    
    for row in tqdm(renamed_hf_dataset, total=len(renamed_hf_dataset), desc="Processing rows"):
        ep_idx = row["episode_index"]

        if ep_idx != current_episode_index:
            if current_episode_rows:
                episode_data = Dataset.from_list(current_episode_rows, features=renamed_hf_dataset.features)
                relative_path_str = source_dataset.meta.get_data_file_path(current_episode_index)
                output_file_path = output_path / relative_path_str
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                episode_data.to_parquet(str(output_file_path))
            current_episode_index = ep_idx
            current_episode_rows = []

        current_episode_rows.append(row)

    if current_episode_rows:
        print("Saving the last episode...")
        episode_data = Dataset.from_list(current_episode_rows, features=renamed_hf_dataset.features)
        relative_path_str = source_dataset.meta.get_data_file_path(current_episode_index)
        output_file_path = output_path / relative_path_str
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        episode_data.to_parquet(str(output_file_path))

    print(f"\n--- Process complete! New dataset is at: {output_path} --- ✅")
    print("File structure and all relevant metadata have been updated with low memory usage.")



def remap_jsonl_stats_file(source_path, target_path, mapping):
    with open(source_path, "r") as f_in, open(target_path, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            if "stats" in data:
                original_stats = data["stats"]
                new_stats = {}
                for old_key, value in original_stats.items():
                    new_key = mapping.get(old_key, old_key)
                    new_stats[new_key] = value
                data["stats"] = new_stats
            f_out.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    SOURCE_REPO_ID = "physical-intelligence/libero"
    KEY_MAPPING = {
        "state": "observation.state",
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
        "actions": "action",
    }
    OUTPUT_DATASET_PATH = "tmp/libero_remapped_structured"

    remap_and_save_dataset_preserving_structure(SOURCE_REPO_ID, KEY_MAPPING, OUTPUT_DATASET_PATH)
