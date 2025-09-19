import logging
import os
import subprocess
import sys
import numpy as np
import torch
from datetime import datetime
import dataclasses
import math
import pathlib
import cv2
import draccus
import imageio
import shutil
import time
import collections
from typing import List
from vqh_logger import setup_logging, log_evaluation_results, log_default_action

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
from vqh_training_policy import VQHPolicy
from libero.libero.envs.predicates.base_predicates import PositionUnchanged

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:
    # policy_path: str = "outputs/train/2025-09-02/18-38-39_vqh/checkpoints/025000/pretrained_model"
    # policy_path: str = "outputs/train/2025-09-07/10-15-46_vqh/checkpoints/010000/pretrained_model"
    # policy_path: str = "outputs/train/2025-09-09/14-13-39_vqh/checkpoints/005000/pretrained_model"    
    # policy_path: str = "outputs/train/2025-09-09/19-02-26_vqh/checkpoints/045000/pretrained_model"
    # policy_path: str = "outputs/train/2025-09-10/16-24-33_vqh/checkpoints/005000/pretrained_model"
    policy_path: str = "outputs/train/2025-09-13/16-24-00_vqh/checkpoints/070000/pretrained_model"
    # policy_path: str = "outputs/train/2025-09-18/23-50-10_vqh/checkpoints/004000/pretrained_model"

    """Path to the pretrained VQHPolicy on the Hugging Face Hub or local directory."""    
    task_suite_name: str = "libero_elegant"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""
    num_candidate_actions: int = 8
    """Number of candidate actions to sample at each step."""
    video_out_path: str = "data/libero_elegant_0919/eval_vqh_task_2"

    stat_script_path: str = "scripts/stat.sh" 
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random Seed (for reproducibility)"""

    foresee_steps: int = 10
    """Excute VQH Ranking Steps"""
    n_action_steps: int = 5
    """Normal Excute Action Steps"""
    g_action_steps: int = 3

    """Excute Gripper Action Steps"""
    vqh_chunk_size: int = 3
    
    """Eval Depend On Q Value"""
    q_history_window_size: int = 10
    q_rise_threshold: float = 0.2

    """Artificial Optimal Intervention Time"""
    intervention_times: List[int] = dataclasses.field(default_factory=lambda: [100, 80, 80, 80, 100, 60, 60, 100, 45, 80])

def choose_best_action_hybrid(
    q_values: torch.Tensor,
    candidate_sequences: torch.Tensor,
    top_k: int = 8,
    cluster_threshold: float = 0.2,
    q_gap_threshold: float = 0.05
) -> int:
    """
    通过一种混合策略选择最佳候选者。
    Q值优先，但当最高Q值的集群与第二高Q值的集群差距很小时，
    选择成员数量更多的那个集群中的最优动作。

    Args:
        q_values (torch.Tensor): Q值, shape (1, num_candidates).
        candidate_sequences (torch.Tensor): 动作序列, shape (1, num_candidates, seq_len, action_dim).
        top_k (int): 筛选Top-K候选者进行分析。
        cluster_threshold (float): Gripper Action的聚类阈值。
        q_gap_threshold (float): 判断两个集群Q值是否“接近”的阈值。

    Returns:
        int: 被选中的最佳候选者的原始索引。
    """
    logging.info("--- Starting Hybrid Action Selection (Q-Value First, Consensus Tie-breaker) ---")

    num_candidates = q_values.shape[1]
    if num_candidates < 2:
        logging.info("Not enough candidates, returning argmax.")
        return torch.argmax(q_values, dim=1).item()

    # 步骤 1: 筛选和聚类 (与之前相同)
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

    # 步骤 2: 分析每个集群的代表Q值和大小
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
        
    # 按Q值对集群进行降序排序
    cluster_summaries.sort(key=lambda x: x['max_q_value'], reverse=True)

    logging.info("\nCluster Analysis (sorted by Q-value):")
    for summary in cluster_summaries:
        logging.info(f"  - Cluster {summary['cluster_id']} | Size: {summary['size']} | Max Q: {summary['max_q_value']:.4f} (Candidate {summary['best_candidate_info']['original_idx']})")

    # 步骤 3: 应用“Q值优先，共识仲裁”逻辑
    champion_cluster = cluster_summaries[0]
    default_winner_info = champion_cluster['best_candidate_info']
    
    # 默认选择是Q值最高的集群中的最优候选者
    final_winner_info = default_winner_info
    
    # 检查是否存在“挑战者”集群
    if len(cluster_summaries) > 1:
        challenger_cluster = cluster_summaries[1]
        
        # 计算Q值差距
        q_gap = champion_cluster['max_q_value'] - challenger_cluster['max_q_value']
        
        logging.info(f"\nDecision Logic:")
        logging.info(f"  - Champion Cluster's Max Q: {champion_cluster['max_q_value']:.4f}")
        logging.info(f"  - Challenger Cluster's Max Q: {challenger_cluster['max_q_value']:.4f}")
        logging.info(f"  - Q-Value Gap: {q_gap:.4f} (Threshold: {q_gap_threshold})")

        # 如果Q值差距很小，并且挑战者集群更大
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

@draccus.wrap()
def eval_libero(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Loading VQHPolicy from {args.policy_path}")
    policy = VQHPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    benchmark_dict = benchmark.get_benchmark_dict()

    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. " f"Available options are: {list(benchmark_dict.keys())}")
   
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.video_out_path = os.path.join(args.video_out_path, timestamp)
  
    video_dir = pathlib.Path(args.video_out_path)
    if video_dir.exists():
        logging.warning(f"Video directory {video_dir} already exists. Clearing its contents.")
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True)
    logging.info(f"Created fresh video directory: {video_dir}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 150
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 250

    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        
        if task_id != 2:
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):  
                          
            if episode_idx != 9:
                continue

            PositionUnchanged.start_new_episode(episode_idx)
            logging.info(f"\nTask: {task_description}")
            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)            
            t = 0            
            agent_frames, wrist_frames = [], []
            done = False
            
            agent_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wrist_frames.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            
            logging.info(f"Starting episode {task_episodes+1}...")
            q_history = collections.deque(maxlen=args.q_history_window_size)

            while t < max_steps:
                try:
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agent_frames.append(agentview_image)
                    wrist_frames.append(wrist_img)

                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))

                    policy.vqh.smolvla_policy.to(torch.float32)
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    with torch.inference_mode():
                        default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                        default_action_chunk = default_sequence[:, :args.vqh_chunk_size, :]
                        default_q_value_tensor = policy.evaluate_q_values(observation, default_action_chunk.unsqueeze(1))
                        current_q = default_q_value_tensor.item()
                        
                        q_history.append(current_q)
                        sma_q = np.mean(list(q_history))

                        is_critical_action = len(q_history) >= args.q_history_window_size and \
                                             current_q > (sma_q + args.q_rise_threshold)
                        
                        logging.debug(f"[Q-Heuristic] Step {t}: Current Q={current_q:.4f}, SMA Q={sma_q:.4f}, Threshold={sma_q + args.q_rise_threshold:.4f}, Is Critical? {is_critical_action}")
                        
                        if is_critical_action and args.num_candidate_actions > 1 and t > args.intervention_times[episode_idx]:
                            logging.info(f"Critical action detected by Q-Value rise. Evaluating multiple candidates with VQH.")                                                        
                            
                            # candidate_sequences = [default_sequence]
                            # for _ in range(args.num_candidate_actions - 1):
                            #     seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                            #     candidate_sequences.append(seq)                                
                            # candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3)                        
                            # all_first_actions = candidate_sequences[:, :, :args.vqh_chunk_size, :]
                            # q_values = policy.evaluate_q_values(observation, all_first_actions)                            

                            num_candidates = args.num_candidate_actions
                            candidate_sequences, q_values = policy.sample_and_evaluate_actions(observation, num_candidates)
                            
                            avg_q_value = torch.mean(q_values).item()
                            logging.info(f"Average Q-value of all candidates: {avg_q_value:.4f}")

                            best_candidate_idx = choose_best_action_hybrid(
                                q_values=q_values,
                                candidate_sequences=candidate_sequences,
                                top_k=args.num_candidate_actions,
                                cluster_threshold=0.2,
                                q_gap_threshold=0.15
                            )
                            
                            best_action_sequence = candidate_sequences[0, best_candidate_idx]
                            best_action_sequence = best_action_sequence[:args.g_action_steps]
                            
                            base_q_values = q_values[0]
                            best_base_idx_in_group = torch.argmax(base_q_values).item()
                            best_base_action_for_log = candidate_sequences[0, best_base_idx_in_group][:args.g_action_steps]

                            log_evaluation_results(
                                q_values=q_values,
                                best_idx=best_candidate_idx,
                                best_sequence=best_action_sequence,
                                base_q_values=base_q_values,
                                best_base_action=best_base_action_for_log,
                                time_step=t,
                                base_action_sequences=candidate_sequences.squeeze(0),
                            )
                        else:
                            logging.info("Not a critical action. Using default action sequence.")
                            best_action_sequence = default_sequence[0]
                            best_action_sequence = best_action_sequence[:args.n_action_steps]
                            log_default_action(
                                sequence=best_action_sequence,
                                time_step=t
                            )                        
                    best_action_sequence_np = best_action_sequence.cpu().numpy()
                    for action_in_sequence in best_action_sequence_np:
                        if t >= max_steps: break
                        modified_action = action_in_sequence.copy()
                        gripper_action_original = modified_action[6]
                        if gripper_action_original > 0.5: modified_action[6] = 1.0
                        elif gripper_action_original < -0.5: modified_action[6] = -1.0
                        else: modified_action[6] = 0.0                        
                        obs, _, done, _ = env.step(modified_action)
                        agent_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        wrist_frames.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
                        t += 1
                        if done: break                                                
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}", exc_info=True)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            base_filename = f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}"
            output_path = pathlib.Path(args.video_out_path)
            fps = 30
            
            agent_video_path = output_path / f"{base_filename}_agentview.mp4"
            imageio.mimsave(agent_video_path, agent_frames, fps=fps)
            logging.info(f"Saved agent view video to {agent_video_path}")
            
            logging.info(f"Success: {done}")
            if total_episodes > 0:
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}")
        if total_episodes > 0:
            logging.info(f"Cumulative success rate: {float(total_successes) / float(total_episodes):.2f}")

    logging.info("--- Evaluation finished ---")
    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.2f}")

    logging.info("--- Running statistics script ---")
    try:
        result = subprocess.run(
            [args.stat_script_path, args.video_out_path],
            check=True, capture_output=True, text=True
        )
        logging.info("Statistics script finished successfully.")
        logging.info(f"Script output:\n---\n{result.stdout.strip()}\n---")
    except FileNotFoundError:
        logging.error(f"Error: The script at '{args.stat_script_path}' was not found.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing statistics script. Return code: {e.returncode}")
        logging.error(f"Stderr: {e.stderr.strip()}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite.
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    setup_logging()
    eval_libero()