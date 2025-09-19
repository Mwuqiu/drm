import logging
import time
import shutil
import pathlib
import sys
import os
import collections
from dataclasses import dataclass, field
from pprint import pformat
from datetime import datetime

import cv2
import draccus
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from vqh_logger import setup_logging, log_evaluation_results, log_default_action
from vqh_training_policy import VQHPolicy
from lerobot.common.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.common.robots import make_robot_from_config, so100_follower
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@dataclass
class Args:
    # --- VQH Policy and Task Configuration ---
    # policy_path: str = "outputs/train/2025-09-15/17-55-51_vqh/checkpoints/005000/pretrained_model"
    policy_path: str = "outputs/train/2025-09-16/19-33-01_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str = "outputs/train/2025-09-16/21-44-32_vqh/checkpoints/005000/pretrained_model"

    """Path to the pretrained VQHPolicy."""
    task_description: str = "Pick up the little yellow duck and put it in the box"
    """The language instruction for the task."""

    # --- Robot and Camera Configuration ---
    robot_type: str = "so100_follower"
    robot_port: str = "/dev/ttyACM1"
    robot_id: str = "follower_arm"
    camera_config: dict = field(default_factory=lambda: {
        "wrist_image": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=30),
        "image": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30)
    })

    # --- VQH-Specific Evaluation Arguments ---
    num_candidate_actions: int = 8
    """Number of candidate actions to sample at each critical step."""
    n_action_steps: int = 50
    """Number of actions to execute from a normal (default) sequence."""
    g_action_steps: int = 5
    """Number of actions to execute from a VQH-selected sequence (typically shorter for precision)."""
    vqh_chunk_size: int = 3
    """Number of initial actions from each candidate sequence to feed to the Q-function."""
    q_history_window_size: int = 1
    """Size of the moving average window for Q-value heuristic."""
    q_rise_threshold: float = 0.01
 
    """Threshold for Q-value rise to trigger critical action evaluation."""

    # --- General Evaluation Arguments ---
    num_episodes: int = 10
    """Total number of evaluation episodes to run."""
    episode_duration_s: float = 120.0
    """Maximum duration for each episode in seconds."""
    fps: int = 30
    """Control frequency for sending actions to the robot."""
    reset_duration_s: float = 20.0
    """Time to wait for manual environment reset between episodes."""
    video_out_path: str = "data/eval_real_world_vqh_0917_1"
    """Directory to save evaluation videos."""
    device: str = "cuda"
    """Device to use for policy inference."""
    seed: int = 42
    """Random seed for reproducibility."""

def choose_best_action_hybrid(
    q_values: torch.Tensor,
    candidate_sequences: torch.Tensor,
    top_k: int = 8,
    cluster_threshold: float = 0.2,
    q_gap_threshold: float = 0.05
) -> int:
    """
    Selects the best candidate using a hybrid strategy: Q-value first,
    but when the top cluster's Q-value is close to the second,
    the one with more members (consensus) is chosen as a tie-breaker.
    """
    logging.info("--- Starting Hybrid Action Selection (Q-Value First, Consensus Tie-breaker) ---")

    num_candidates = q_values.shape[1]
    if num_candidates < 2:
        logging.info("Not enough candidates, returning argmax.")
        return torch.argmax(q_values, dim=1).item()

    # Step 1: Filter and cluster top-k candidates
    actual_k = min(top_k, num_candidates)
    sorted_q_values, sorted_indices = torch.sort(q_values, dim=1, descending=True)
    top_k_indices = sorted_indices[0, :actual_k]
    top_k_q_values = sorted_q_values[0, :actual_k]
    first_step_actions = candidate_sequences[0, :, 0, :]
    top_k_gripper_actions = first_step_actions[top_k_indices, -1].cpu().numpy()

    clusters = []
    for i in range(actual_k):
        item_info = {
            'original_idx': top_k_indices[i].item(),
            'q_value': top_k_q_values[i].item(),
            'gripper_val': top_k_gripper_actions[i]
        }
        found_cluster = False
        for cluster in clusters:
            if abs(item_info['gripper_val'] - cluster[0]['gripper_val']) <= cluster_threshold:
                cluster.append(item_info)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([item_info])
            
    if not clusters:
        logging.warning("Clustering failed, returning default argmax.")
        return torch.argmax(q_values, dim=1).item()

    # Step 2: Analyze clusters for their representative Q-value and size
    cluster_summaries = []
    for i, cluster in enumerate(clusters):
        if not cluster: continue
        best_in_cluster = max(cluster, key=lambda x: x['q_value'])
        cluster_summaries.append({
            'cluster_id': i,
            'size': len(cluster),
            'max_q_value': best_in_cluster['q_value'],
            'best_candidate_info': best_in_cluster
        })
        
    cluster_summaries.sort(key=lambda x: x['max_q_value'], reverse=True)

    logging.info("\nCluster Analysis (sorted by Q-value):")
    for summary in cluster_summaries:
        logging.info(f"  - Cluster {summary['cluster_id']} | Size: {summary['size']} | Max Q: {summary['max_q_value']:.4f} (Candidate {summary['best_candidate_info']['original_idx']})")

    # Step 3: Apply "Q-value first, consensus tie-breaker" logic
    champion_cluster = cluster_summaries[0]
    final_winner_info = champion_cluster['best_candidate_info']
    
    if len(cluster_summaries) > 1:
        challenger_cluster = cluster_summaries[1]
        q_gap = champion_cluster['max_q_value'] - challenger_cluster['max_q_value']
        
        logging.info(f"\nDecision Logic:")
        logging.info(f"  - Q-Value Gap: {q_gap:.4f} (Threshold: {q_gap_threshold})")

        if q_gap <= q_gap_threshold and challenger_cluster['size'] > champion_cluster['size']:
            logging.info(f"  - Condition MET: Q-gap is close AND challenger cluster is larger. Switching winner.")
            final_winner_info = challenger_cluster['best_candidate_info']
        else:
            logging.info(f"  - Condition FAILED: Sticking with the highest Q-value champion.")

    logging.info(f"\n--- Final Decision: Winning Candidate Index: {final_winner_info['original_idx']} ---")
    return final_winner_info['original_idx']

@draccus.wrap()
def eval_real_robot_vqh(args: Args):
    logging.info(pformat(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_safe_torch_device(args.device)
    listener, events = init_keyboard_listener()

    logging.info("Initializing robot to get hardware metadata...")
    robot_config = so100_follower.SO100FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=args.camera_config)
    robot = make_robot_from_config(robot_config)
    
    logging.info(f"Loading VQH policy from '{args.policy_path}'...")

    policy = VQHPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()
    
    robot.connect()
    logging.info("Robot hardware connection successful.")

    video_dir = pathlib.Path(args.video_out_path)
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True)
    
    episode_idx = 0
    while episode_idx < args.num_episodes:
        log_title = f"Episode {episode_idx + 1}/{args.num_episodes}"
        logging.info(f"--- {log_title} ---")
        
        policy.reset()
        q_history = collections.deque(maxlen=args.q_history_window_size)

        frames = []
        start_time = time.perf_counter()
        
        pbar = tqdm(total=int(args.episode_duration_s), desc=f"Evaluating E{episode_idx+1}")
        last_update_time = start_time
        t = 0 
        
        while (time.perf_counter() - start_time) < args.episode_duration_s:
            if events["exit_early"]:
                break

            observation_from_robot = robot.get_observation()
            
            observation_for_policy = {}
            for key in ["image", "wrist_image"]:
                if key in observation_from_robot:
                    value = observation_from_robot[key]
                    tensor = torch.from_numpy(value / 255.0).permute(2, 0, 1).to(torch.float32)
                    observation_for_policy[f"observation.images.{key}"] = tensor.to(device).unsqueeze(0)

            camera_keys = list(args.camera_config.keys())
            state_feature_keys = [k for k in robot.observation_features if k not in camera_keys]            
            state_values = [observation_from_robot[key] for key in state_feature_keys]
            state_numpy = np.array(state_values, dtype=np.float32)
            state_tensor = torch.from_numpy(state_numpy).to(torch.float32)
            observation_for_policy["observation.state"] = state_tensor.to(device).unsqueeze(0)
            observation_for_policy["task"] = args.task_description
            
            # 2. SELECT ACTION SEQUENCE USING VQH POLICY
            with torch.inference_mode():
                # Always get a default action sequence and its Q-value
                default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation_for_policy)
                default_action_chunk = default_sequence[:, :args.vqh_chunk_size, :]
                default_q_value_tensor = policy.evaluate_q_values(observation_for_policy, default_action_chunk.unsqueeze(1))
                current_q = default_q_value_tensor.item()

                logging.info(f"CURRENT Q : {current_q}")
            
                q_history.append(current_q)
                sma_q = np.mean(list(q_history))

                is_critical_action = True

                logging.debug(f"[Q-Heuristic] Step {pbar.n}: Current Q={current_q:.4f}, SMA Q={sma_q:.4f}, Threshold={sma_q + args.q_rise_threshold:.4f}, Is Critical? {is_critical_action}")

                if is_critical_action and args.num_candidate_actions > 1 and t > 1 and t < 15:
                    logging.info(f"CRITICAL STEP: Q-value rise detected. Evaluating multiple candidates with VQH.")
                    num_candidates = args.num_candidate_actions
                    batched_observation = {
                        key: value.repeat([num_candidates] + [1] * (value.dim() - 1)) if isinstance(value, torch.Tensor) else [value] * num_candidates
                        for key, value in observation_for_policy.items()
                    }
                    
                    # Sample multiple action sequences
                    all_sequences = policy.vqh.smolvla_policy.sample_action_sequence(batched_observation)
                    candidate_sequences = all_sequences.unsqueeze(0) # Add batch dim for Q-func
                    
                    # Evaluate all candidates
                    action_chunks_for_q = candidate_sequences[:, :, :args.vqh_chunk_size, :]
                    q_values = policy.evaluate_q_values(observation_for_policy, action_chunks_for_q)
                    
                    # Choose the best action using the hybrid strategy
                    best_candidate_idx = choose_best_action_hybrid(
                        q_values=q_values,
                        candidate_sequences=candidate_sequences,
                        top_k=args.num_candidate_actions,
                        q_gap_threshold=0.15 # Can be tuned
                    )
                    
                    best_action_sequence = candidate_sequences[0, best_candidate_idx]
                    best_action_sequence = best_action_sequence[:args.g_action_steps]
                    
                    log_evaluation_results(
                        q_values=q_values,
                        best_idx=best_candidate_idx,
                        best_sequence=best_action_sequence,
                        base_q_values=q_values[0],
                        time_step=t,
                        base_action_sequences=candidate_sequences.squeeze(0),
                    )
                
                else:
                    # Not a critical action, use the default sequence
                    best_action_sequence = default_sequence[0]
                    best_action_sequence = best_action_sequence[:args.n_action_steps]
                    log_default_action(sequence=best_action_sequence, time_step=t)

            # 3. EXECUTE THE CHOSEN ACTION SEQUENCE
            best_action_sequence_np = best_action_sequence.cpu().numpy()
            
            for action_in_sequence in best_action_sequence_np:
                if events["exit_early"] or (time.perf_counter() - start_time) >= args.episode_duration_s:
                    break
                    
                loop_start_time = time.perf_counter()
                
                action_dict = {key: action_in_sequence[i] for i, key in enumerate(robot.action_features)}
                robot.send_action(action_dict)
                
                obs_for_video = robot.get_observation()
                main_cam_image = obs_for_video.get("image")
                if main_cam_image is not None:
                    frames.append(main_cam_image)

                loop_duration = time.perf_counter() - loop_start_time
                busy_wait(1.0 / args.fps - loop_duration)
            
            current_time = time.perf_counter()
            pbar.update(int(current_time - last_update_time))
            last_update_time = current_time
            t += 1

            pause_start_time = time.perf_counter()
            while time.perf_counter() - pause_start_time < 1.0:
                obs_for_video = robot.get_observation()
                main_cam_image = obs_for_video.get("image")
                
                if main_cam_image is not None:
                    frames.append(main_cam_image)
                busy_wait(1.0 / args.fps)

        pbar.close()

        if events["exit_early"] and not events["rerecord_episode"]:
             logging.info("Keyboard interrupt detected, ending episode early...")
             events["exit_early"] = False
        else:
            logging.info("Episode duration reached.")

        video_path = video_dir / f"rollout_episode_{episode_idx}.mp4"
        imageio.mimsave(video_path, frames, fps=args.fps)
        logging.info(f"Episode video saved to: {video_path}")

        if events["rerecord_episode"]:
            logging.info(f"--- Re-recording Episode {episode_idx + 1} as requested. ---")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            input(">>> Please reset the environment. Press Enter to re-record the episode. <<<")
            continue

        if events["stop_recording"]:
            logging.info("Stop signal ('Esc') detected, terminating all evaluations.")
            break

        if episode_idx < args.num_episodes - 1:
            logging.info(f"Please reset the environment within {args.reset_duration_s}s. Press '→' to skip.")
            reset_start_time = time.perf_counter()
            pbar_reset = tqdm(total=int(args.reset_duration_s), desc="Resetting")
            while (time.perf_counter() - reset_start_time) < args.reset_duration_s:
                if events["exit_early"]: break
                time.sleep(0.1)
                pbar_reset.update(0.1)
            pbar_reset.close()
            events["exit_early"] = False
            input(">>> Press Enter to start the next episode after resetting <<<")
        
        episode_idx += 1

    logging.info("--- All evaluations complete ---")
    robot.disconnect()
    if not is_headless() and listener is not None:
        listener.stop()
    logging.info("Robot and keyboard listener have been shut down.")

if __name__ == "__main__":
    setup_logging()
    eval_real_robot_vqh()