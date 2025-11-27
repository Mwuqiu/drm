import argparse
import torch
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def filter_dataset(args):
    """
    Loads a LeRobotDataset, filters it to remove static periods (including at the start, end, and middle of
    episodes) based on action changes, and saves it to a new dataset.
    """
    print(f"Loading source dataset: {args.source_repo_id}...")
    source_ds = LeRobotDataset(
        repo_id=args.source_repo_id,
        video_backend='pyav'
    )

    print(f"Creating new target dataset: {args.target_repo_id}...")
    target_ds = LeRobotDataset.create(
        repo_id=args.target_repo_id,
        fps=source_ds.meta.fps,
        features=source_ds.meta.features,
        robot_type=source_ds.meta.robot_type,
        use_videos=(source_ds.meta.video_path is not None),
        image_writer_threads=4,
    )

    total_original_frames = 0
    total_filtered_frames = 0

    valid_feature_keys = target_ds.meta.features.keys()
    camera_keys = target_ds.meta.camera_keys
    keys_to_exclude = {'index', 'frame_index', 'task', 'episode_index', 'timestamp', 'task_index'}

    print("\nStarting filtering process...")
    for ep_idx in tqdm(range(source_ds.num_episodes), desc="Filtering Episodes"):
        ep_start = source_ds.episode_data_index["from"][ep_idx].item()
        ep_end = source_ds.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start
        total_original_frames += ep_len

        if ep_len < 2:
            tqdm.write(f"Skipping episode {ep_idx}: too short (length {ep_len}).")
            continue

        # Calculate the magnitude of action changes between consecutive frames
        ep_actions = torch.stack(source_ds.hf_dataset[ep_start:ep_end]['action'])
        action_diffs = torch.diff(ep_actions, dim=0)
        motion_magnitude = torch.linalg.norm(action_diffs, dim=1)
        
        # ADDED: Print the minimum motion magnitude for the episode
        if motion_magnitude.numel() > 0:
            tqdm.write(f"Episode {ep_idx}: Min motion magnitude: {torch.min(motion_magnitude).item()}")

        # Find all indices where motion exceeds the threshold
        motion_indices = torch.where(motion_magnitude > args.motion_threshold)[0]

        if len(motion_indices) == 0:
            tqdm.write(f"Skipping episode {ep_idx}: no motion detected.")
            continue

        # A motion at index `i` (between frame i and i+1) means we should keep both frames.
        # We create a set of all unique frame indices that are part of any motion.
        indices_to_keep = set(motion_indices.tolist())
        indices_to_keep.update((motion_indices + 1).tolist())
        
        # Sort the indices to add them in chronological order.
        sorted_indices_to_keep = sorted(list(indices_to_keep))

        new_episode_frame_idx = 0
        for frame_offset in sorted_indices_to_keep:
            original_frame_index = ep_start + frame_offset
            frame_data = source_ds[original_frame_index]
            
            new_frame = {}
            for key, value in frame_data.items():
                if key in keys_to_exclude:
                    continue
                if key in valid_feature_keys:
                    if key in camera_keys:
                        # Convert CHW to HWC for image writers
                        new_frame[key] = value.permute(1, 2, 0).numpy()
                    else:
                        new_frame[key] = value.clone()
            
            # Create a new, continuous timestamp for the filtered episode
            new_timestamp = new_episode_frame_idx / target_ds.meta.fps
            target_ds.add_frame(
                frame=new_frame, 
                task=frame_data['task'],
                timestamp=new_timestamp
            )
            new_episode_frame_idx += 1
        
        if new_episode_frame_idx > 0:
            target_ds.save_episode()
            tqdm.write(f"Episode {ep_idx}: Kept {new_episode_frame_idx} active frames from original {ep_len} frames.")
            total_filtered_frames += new_episode_frame_idx

    target_ds.stop_image_writer()
    
    print("\nFiltering complete!")
    print("===================================")
    print(f"Total original frames: {total_original_frames}")
    print(f"Total filtered frames: {total_filtered_frames}")
    if total_original_frames > 0:
        reduction = (1 - total_filtered_frames / total_original_frames) * 100
        print(f"Data reduction: {reduction:.2f}%")
    print(f"New filtered dataset is ready at: {target_ds.root.resolve()}")
    print("===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a LeRobotDataset to remove all static parts based on action changes.")
    parser.add_argument("--source_repo_id", type=str, required=True, help="The Hugging Face repository ID of the source dataset.")
    parser.add_argument("--target_repo_id", type=str, required=True, help="The local repository ID for the new, filtered dataset.")
    parser.add_argument("--motion_threshold", type=float, default=1e-4, help="The threshold for the L2 norm of the action difference to be considered as motion.")
    args = parser.parse_args()
    filter_dataset(args)