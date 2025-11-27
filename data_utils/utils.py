import logging
import os
import sys
import shutil
import collections
import pathlib
from datetime import datetime
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME


def setup_logging(log_filename_prefix: str = "evaluation_run"):
    """
    Sets up logging to both a file and the console.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_with_timestamp = f"{log_filename_prefix}_{timestamp}.log"

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename_with_timestamp)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Output will be saved to: {log_filepath}")


def log_evaluation_results(
    q_values: torch.Tensor,
    best_idx: int,
    best_sequence: torch.Tensor,
    base_q_values: torch.Tensor = None,
    best_base_action: torch.Tensor = None,
    time_step: int = None,
    base_action_sequences: torch.Tensor = None,
):
    """Logs the detailed results of a VQH critical action evaluation step."""
    q_values_np = q_values.cpu().numpy().flatten()

    summary_msgs = ["\n"]
    summary_msgs.append("=" * 70)
    if time_step is not None:
        summary_msgs.append(f"- Time Step Now - : {time_step}")
    summary_msgs.append("[VQH Q-Value Evaluation]")
    if base_q_values is not None and base_q_values.numel() > 0:
        base_q_str = ", ".join([f"{q:.4f}" for q in base_q_values.cpu().numpy().flatten()])
        summary_msgs.append(f"  - Base Policy Q-Values:  [{base_q_str}]")
    summary_msgs.append("  - Summary:")
    summary_msgs.append(f"    - Winning Candidate Idx: {best_idx}")
    summary_msgs.append(f"    - Winning Q-Value:       {q_values_np[best_idx]:.4f}")
    summary_msgs.append(f"    - Average Q-Value:       {np.mean(q_values_np):.4f}")
    logging.debug("\n".join(summary_msgs))

    details_msgs = ["-" * 70]
    details_msgs.append("[All Candidate Actions]")
    if base_action_sequences is not None and base_action_sequences.numel() > 0:
        details_msgs.append("  - Base Policy Actions:")
        for i, sequence in enumerate(base_action_sequences):
            q_value = q_values_np[i]
            details_msgs.append(f"    - Candidate {i} (Q={q_value:.4f}):")
            sequence_np = sequence.cpu().numpy()
            for step_idx, step_action in enumerate(sequence_np):
                step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
                details_msgs.append(f"      - Step {step_idx+1}: [{step_str}]")
    else:
        details_msgs.append("  - No candidate sequences were provided for detailed logging.")
    logging.debug("\n".join(details_msgs))

    decision_msgs = ["-" * 70]
    decision_msgs.append("[Decision]")
    decision_msgs.append(f"  - ✅ VQH selected action from Candidate {best_idx}.")
    decision_msgs.append("-" * 70)
    decision_msgs.append("[Overall Winning Action Sequence]")
    best_sequence_np = best_sequence.cpu().numpy()
    for i, step_action in enumerate(best_sequence_np):
        step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
        decision_msgs.append(f"  - Step {i+1}: [{step_str}]")

    decision_msgs.append("=" * 70)
    logging.debug("\n".join(decision_msgs))


def log_default_action(sequence: torch.Tensor, time_step: int):
    """Logs the default action sequence when it's used as a fallback."""
    log_msgs = ["\n"]
    log_msgs.append("=" * 70)
    log_msgs.append(f"- Time Step Now - : {time_step}")
    log_msgs.append("[Decision]")
    log_msgs.append("  - ⚙️ Default action sequence was chosen (non-critical step).")
    log_msgs.append("-" * 70)
    log_msgs.append("[Default Action Executed]")
    
    sequence_np = sequence.cpu().numpy()
    for i, step_action in enumerate(sequence_np):
        step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
        log_msgs.append(f"  - Step {i+1}: [{step_str}]")
    
    log_msgs.append("=" * 70)
    logging.debug("\n".join(log_msgs))


def init_dataset_if_needed(args):
    if not args.save_dataset:
        return None, None

    logging.info(f"Preparing to save dataset to: {args.dataset_repo_id}")
    dataset_path = HF_LEROBOT_HOME / args.dataset_repo_id
    
    if dataset_path.exists():
        logging.info("Dataset found locally. Loading existing dataset to append data.")
        replay_dataset = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            download_videos=True,
        )
    else:
        logging.info("No local dataset found. Creating a new one.")
        dataset_features = {
            "observation.images.image": {"shape": (256, 256, 3), "dtype": "image"},
            "observation.images.wrist_image": {"shape": (256, 256, 3), "dtype": "image"},
            "observation.state": {"shape": (8,), "dtype": "float32"},
            "action": {"shape": (7,), "dtype": "float32"},
        }
        replay_dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            features=dataset_features,
            fps=10,
            use_videos=True,
        )
    
    temp_img_dir = replay_dataset.root / "temp_images"
    return replay_dataset, temp_img_dir


def save_episode_to_dataset(replay_dataset, episode_buffer, task_description, temp_img_dir):
    if replay_dataset is None or not episode_buffer:
        return

    logging.info(f"Saving episode to dataset: {replay_dataset.repo_id}")
    
    episode_data_dict = collections.defaultdict(list)
    for step in episode_buffer:
        for key, value in step.items():
            episode_data_dict[key].append(value)
    
    episode_length = len(episode_buffer)
    episode_data_dict["frame_index"] = list(range(episode_length))
    fps = replay_dataset.fps
    episode_data_dict["timestamp"] = [i / fps for i in range(episode_length)]
    start_index = replay_dataset.meta.total_frames
    episode_data_dict["index"] = list(range(start_index, start_index + episode_length))
    
    task_index = replay_dataset.meta.get_task_index(task_description)
    if task_index is None:
        replay_dataset.meta.add_task(task_description)
        task_index = replay_dataset.meta.get_task_index(task_description)
    episode_data_dict["task_index"] = [task_index] * episode_length
    
    episode_index_to_save = replay_dataset.meta.total_episodes
    episode_data_dict["episode_index"] = episode_index_to_save
    episode_data_dict["size"] = episode_length
    episode_data_dict["task"] = [task_description] * episode_length
    
    replay_dataset.save_episode(episode_data=episode_data_dict)

    for key in ["observation.images.image", "observation.images.wrist_image"]:
        saved_episode_img_dir = temp_img_dir / key / f"episode_{episode_index_to_save:06d}"
        if saved_episode_img_dir.is_dir():
            shutil.rmtree(saved_episode_img_dir)


def choose_best_action_hybrid(
    q_values: torch.Tensor,
    candidate_sequences: torch.Tensor,
    top_k: int = 8,
    cluster_threshold: float = 0.2,
    q_gap_threshold: float = 0.05
) -> int:
    logging.info("--- Starting Hybrid Action Selection (Q-Value First, Consensus Tie-breaker) ---")

    num_candidates = q_values.shape[1]
    if num_candidates < 2:
        logging.info("Not enough candidates, returning argmax.")
        return torch.argmax(q_values, dim=1).item()

    actual_k = min(top_k, num_candidates)
    sorted_q_values, sorted_indices = torch.sort(q_values, dim=1, descending=True)
    top_k_indices = sorted_indices[0, :actual_k]
    top_k_q_values = sorted_q_values[0, :actual_k]
    first_step_actions = candidate_sequences[0, :, 0, :]
    top_k_gripper_actions = first_step_actions[top_k_indices, -1].cpu().numpy()

    clusters = []
    for i in range(actual_k):
        current_gripper = top_k_gripper_actions[i]
        item_info = {
            'original_idx': top_k_indices[i].item(),
            'q_value': top_k_q_values[i].item(),
            'gripper_val': current_gripper
        }
        found_cluster = False
        for cluster in clusters:
            representative_gripper = cluster[0]['gripper_val']
            if abs(current_gripper - representative_gripper) <= cluster_threshold:
                cluster.append(item_info)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([item_info])
            
    if not clusters:
        logging.warning("Clustering failed, returning default argmax.")
        return torch.argmax(q_values, dim=1).item()

    cluster_summaries = []
    for i, cluster in enumerate(clusters):
        if not cluster: continue
        best_candidate_in_cluster = max(cluster, key=lambda x: x['q_value'])
        cluster_summaries.append({
            'cluster_id': i,
            'size': len(cluster),
            'max_q_value': best_candidate_in_cluster['q_value'],
            'best_candidate_info': best_candidate_in_cluster
        })
        
    cluster_summaries.sort(key=lambda x: x['max_q_value'], reverse=True)
    logging.info("\nCluster Analysis (sorted by Q-value):")
    for summary in cluster_summaries:
        logging.info(f"  - Cluster {summary['cluster_id']} | Size: {summary['size']} | Max Q: {summary['max_q_value']:.4f} (Candidate {summary['best_candidate_info']['original_idx']})")
    champion_cluster = cluster_summaries[0]
    default_winner_info = champion_cluster['best_candidate_info']
    
    final_winner_info = default_winner_info
    
    if len(cluster_summaries) > 1:
        challenger_cluster = cluster_summaries[1]
        
        q_gap = champion_cluster['max_q_value'] - challenger_cluster['max_q_value']
        
        logging.info(f"\nDecision Logic:")
        logging.info(f"  - Champion Cluster's Max Q: {champion_cluster['max_q_value']:.4f}")
        logging.info(f"  - Challenger Cluster's Max Q: {challenger_cluster['max_q_value']:.4f}")
        logging.info(f"  - Q-Value Gap: {q_gap:.4f} (Threshold: {q_gap_threshold})")

        if q_gap <= q_gap_threshold:
            logging.info(f"  - Condition MET: Q-gap is within threshold.")
            logging.info(f"  - Champion Size: {champion_cluster['size']}, Challenger Size: {challenger_cluster['size']}")
            if challenger_cluster['size'] > champion_cluster['size']:
                logging.info(f"  - Condition MET: Challenger cluster is larger. Switching to challenger.")
                final_winner_info = challenger_cluster['best_candidate_info']
            else:
                logging.info(f"  - Condition FAILED: Challenger cluster is NOT larger. Sticking with champion.")
        else:
            logging.info(f"  - Condition FAILED: Q-gap is too large. Sticking with champion.")

    logging.info("\n--- Final Decision ---")
    logging.info(f"  - Winning Candidate Index: {final_winner_info['original_idx']}")
    logging.info(f"  - Winning Q-Value: {final_winner_info['q_value']:.4f}")
    return final_winner_info['original_idx']