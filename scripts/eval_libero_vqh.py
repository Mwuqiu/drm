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
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:        
    # policy_path: str = "outputs/train/2025-09-13/16-24-00_vqh/checkpoints/070000/pretrained_model"    # Chunk Size 为 50 的 Libero Elegant
    # policy_path: str = "outputs/train/2025-10-02/11-35-37_vqh/checkpoints/006000/pretrained_model"    # Chunk Size 为 10 的 Libero Elegant
    # policy_path: str= "outputs/train/2025-09-22/12-23-01_vqh/checkpoints/010000/pretrained_model"     # 专为 Task-2 训练的策略
    # policy_path: str = "outputs/train/2025-10-04/20-13-43_vqh/checkpoints/005000/pretrained_model"    # 专为 Task-3 训练的策略
    policy_path: str = "outputs/train/2025-09-20/15-55-26_vqh/checkpoints/008000/pretrained_model"    # 专为 Task-4 训练的策略
    # policy_path: str="outputs/train/2025-10-01/19-29-36_vqh/checkpoints/005000/pretrained_model"      # 专为 Task-5 训练的策略
    # policy_path: str= "outputs/train/2025-10-03/15-19-17_vqh/checkpoints/005000/pretrained_model"     # 专为 Task-6 训练的策略
    # policy_path: str= "outputs/train/2025-10-03/16-58-43_vqh/checkpoints/005000/pretrained_model"      # 专为 Task-7 训练的策略

    # Binary Policy
    # policy_path: str= "outputs/train/2025-10-06/14-40-48_vqh/checkpoints/005000/pretrained_model" 
    # policy_path: str= "outputs/train/2025-10-06/16-35-03_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str="outputs/train/2025-10-06/19-33-30_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str="outputs/train/2025-10-06/21-03-57_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str="outputs/train/2025-10-06/22-45-52_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str="outputs/train/2025-10-07/00-34-56_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str="outputs/train/2025-10-07/02-02-58_vqh/checkpoints/005000/pretrained_model"
    # policy_path: str="outputs/train/2025-10-07/03-30-29_vqh/checkpoints/005000/pretrained_model"

    """Path to the pretrained VQHPolicy on the Hugging Face Hub or local directory."""

    task_suite_name: str = "libero_elegant"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_elegant, libero_pragmatic, libero_object_elegant"""
    
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""
    num_candidate_actions: int = 8
    """Number of candidate actions to sample at each step."""

    video_out_path: str = "data/libero_elegant_1118/suppvideo_task-5"
    exp_specific_des: str  = "Task"
    full_intervention_mode: bool = False

    """Eval Depend On Q Value"""
    q_history_window_size: int = 10
    q_change_threshold: float = 0.05

    max_steps: int = 520
    continue_on_done: bool = False

    episode_idx_to_run: int = -1
    task_id_to_run: int = 4

    stat_script_path: str = "scripts/stat.sh"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random Seed (for reproducibility)"""

    n_action_steps: int = 10
    """Normal Excute Action Steps"""
    g_action_steps: int = 10
    """Excute Gripper Action Steps"""
    vqh_chunk_size: int = 10
    
    """Artificial Optimal Intervention Time"""
    intervention_times: List[int] = dataclasses.field(default_factory=lambda: [100, 80, 80, 80, 100, 60, 60, 100, 45, 80])

    # --- Dataset Saving Arguments ---
    save_dataset: bool = False
    dataset_repo_id: str = "libero_elegant_0920/task_4_collected"
    

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

    # --- Initialize Dataset ---
    replay_dataset = None
    temp_img_dir = None
    if args.save_dataset:
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

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # args.video_out_path = os.path.join(args.video_out_path, timestamp)
    
    args.video_out_path = os.path.join(args.video_out_path, args.exp_specific_des)
    video_dir = pathlib.Path(args.video_out_path)    
    if not video_dir.exists():
        video_dir.mkdir(parents=True)

    decision_img_dir = video_dir / "decision_step_images"
    if not decision_img_dir.exists():
        decision_img_dir.mkdir(parents=True)
    logging.info(f"将决策步骤图像保存到: {decision_img_dir}")

    # if video_dir.exists():
    #     logging.warning(f"Video directory {video_dir} already exists. Clearing its contents.")
    #     shutil.rmtree(video_dir)
    # video_dir.mkdir(parents=True)
    # logging.info(f"Created fresh video directory: {video_dir}")
    
    stats_file_path = video_dir / "episode_summary_stats.csv"
    if not stats_file_path.exists():
        with open(stats_file_path, 'w', newline='') as f:
            header = [
                "Task_ID", "Episode_ID", "Status", "Total_Steps",
                "Intervention_Count", "Avg_Intervention_Time_s", "Total_Intervention_Time_s",
                "Default_Action_Count", "Avg_Default_Time_s", "Total_Default_Time_s",
                "Total_Decision_Time_s"
            ]
            f.write(','.join(header) + '\n')


    max_steps = args.max_steps 

    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):

        if args.task_id_to_run != -1 and task_id != args.task_id_to_run:
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
            if args.episode_idx_to_run != -1 and episode_idx != args.episode_idx_to_run:
                continue

            if episode_idx != 5:
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
            episode_buffer = []
            done = False
            
            agent_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wrist_frames.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))

            intervention_count = 0
            intervention_decision_times = []
            default_decision_times = []
            
            logging.info(f"Starting episode {task_episodes+1}...")
            q_history = collections.deque(maxlen=args.q_history_window_size)

            while t < max_steps:
                try:
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img_filename = f"task_{task_id}_ep_{episode_idx}_step_{t:04d}.png"
                    img_save_path = decision_img_dir / img_filename
                    imageio.imwrite(img_save_path, agentview_image)

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

                    decision_start_time = time.time()

                    with torch.inference_mode():
                        default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                        default_action_chunk = default_sequence[:, :args.vqh_chunk_size, :]
                        default_q_value_tensor = policy.evaluate_q_values(observation, default_action_chunk.unsqueeze(1))
                        current_q = default_q_value_tensor.item()
                        
                        q_history.append(current_q)
                        sma_q = np.mean(list(q_history))

                        q_change = abs(current_q - sma_q)

                        is_critical_action = len(q_history) >= args.q_history_window_size and \
                                            q_change > args.q_change_threshold

                        logging.info(f"[Q-Heuristic] Step {t}: Current Q={current_q:.4f}, SMA Q={sma_q:.4f}, |Q Change|={q_change:.4f}, Threshold={args.q_change_threshold:.4f}, Is Critical? {is_critical_action}")

                        should_intervene = (is_critical_action or args.full_intervention_mode) and args.num_candidate_actions > 1


                        if should_intervene:
                            logging.info(f"Step {t}: Elegant Critic Intervention. Reason: {'Full Mode' if args.full_intervention_mode else 'Q-Value Heuristic'}")
                            intervention_count += 1
                            logging.info(f"Critical action detected by Q-Value rise. Evaluating multiple candidates with VQH.")                                                        
                            
                            num_candidates = args.num_candidate_actions
                            # candidate_sequences, q_values = policy.sample_and_evaluate_actions(observation, num_candidates, default_sequence)
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
                            intervention_decision_times.append(time.time() - decision_start_time)
                        else:
                            logging.info("Not a critical action. Using default action sequence.")
                            best_action_sequence = default_sequence[0]
                            best_action_sequence = best_action_sequence[:args.n_action_steps]
                            log_default_action(
                                sequence=best_action_sequence,
                                time_step=t
                            )                    
                            default_decision_times.append(time.time() - decision_start_time)

                    best_action_sequence_np = best_action_sequence.cpu().numpy()
                    for action_in_sequence in best_action_sequence_np:
                        if t >= max_steps: break                        
                        modified_action = action_in_sequence.copy()
                        gripper_action_original = modified_action[6]
                        if gripper_action_original > 0.5: modified_action[6] = 1.0
                        elif gripper_action_original < -0.5: modified_action[6] = -1.0
                        else: modified_action[6] = 0.0
                        if args.save_dataset:
                            step_data = {
                                "observation.state": state.astype(np.float32),
                                "action": modified_action.astype(np.float32),
                            }
                            next_episode_index = replay_dataset.meta.total_episodes
                            
                            for key, img_array in [("observation.images.image", agentview_image), ("observation.images.wrist_image", wrist_img)]:
                                episode_img_dir = temp_img_dir / key / f"episode_{next_episode_index:06d}"
                                episode_img_dir.mkdir(parents=True, exist_ok=True)
                                img_path = episode_img_dir / f"frame_{t:06d}.png"
                                imageio.imwrite(img_path, img_array)
                                step_data[key] = str(img_path)
                                                            
                            episode_buffer.append(step_data)

                        obs, _, done, _ = env.step(modified_action)
                        # obs, _, done, _ = env.step(action_in_sequence)

                        agent_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        wrist_frames.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
                        t += 1

                    if done:
                        task_successes += 1
                        total_successes += 1
                        if not args.continue_on_done:
                            break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}", exc_info=True)
                    break

            task_episodes += 1
            total_episodes += 1

            # 在该 episode 结束后，打印总结信息
            logging.info("--- Episode Summary ---")
            logging.info(f"  - Final Status: {'Success' if done else 'Failure'}")
            logging.info(f"  - Total Steps: {t}")
            
            # 干预统计
            total_interventions = len(intervention_decision_times)
            avg_intervention_time = np.mean(intervention_decision_times) if total_interventions > 0 else 0
            total_intervention_time = np.sum(intervention_decision_times)
            logging.info(f"  - Elegant Critic Interventions: {total_interventions}")
            logging.info(f"    - Average Intervention Time: {avg_intervention_time:.4f} seconds")
            logging.info(f"    - Total Intervention Time: {total_intervention_time:.4f} seconds")

            # 默认动作统计
            total_default_actions = len(default_decision_times)
            avg_default_time = np.mean(default_decision_times) if total_default_actions > 0 else 0
            total_default_time = np.sum(default_decision_times)
            logging.info(f"  - Default Actions: {total_default_actions}")
            logging.info(f"    - Average Default Action Time: {avg_default_time:.4f} seconds")
            logging.info(f"    - Total Default Action Time: {total_default_time:.4f} seconds")
            
            # 总体计时
            total_decision_time = total_intervention_time + total_default_time
            logging.info(f"  - Total Decision-Making Time for Episode: {total_decision_time:.4f} seconds")
            logging.info("-----------------------")

            # 将本 episode 的统计数据写入 CSV 文件
            # 1. 收集所有需要记录的数据
            status_str = "Success" if done else "Failure"
            total_interventions = len(intervention_decision_times)
            avg_intervention_time = np.mean(intervention_decision_times) if total_interventions > 0 else 0
            total_intervention_time = np.sum(intervention_decision_times)
            total_default_actions = len(default_decision_times)
            avg_default_time = np.mean(default_decision_times) if total_default_actions > 0 else 0
            total_default_time = np.sum(default_decision_times)
            total_decision_time = total_intervention_time + total_default_time
            
            # 2. 将数据格式化为一行 CSV
            stats_row = [
                task_id, episode_idx, status_str, t,
                total_interventions, f"{avg_intervention_time:.4f}", f"{total_intervention_time:.4f}",
                total_default_actions, f"{avg_default_time:.4f}", f"{total_default_time:.4f}",
                f"{total_decision_time:.4f}"
            ]
            
            # 3. 以追加模式 ('a') 打开文件并写入
            with open(stats_file_path, 'a', newline='') as f:
                f.write(','.join(map(str, stats_row)) + '\n')
            
            logging.info(f"Statistics for this episode have been saved to {stats_file_path}")
            # -->

            if args.save_dataset and episode_buffer:
                logging.info(f"Saving episode to dataset: {args.dataset_repo_id}")
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

                # Clean up temporary image files for the saved episode
                for key in ["observation.images.image", "observation.images.wrist_image"]:
                    saved_episode_img_dir = temp_img_dir / key / f"episode_{episode_index_to_save:06d}"
                    if saved_episode_img_dir.is_dir():
                        shutil.rmtree(saved_episode_img_dir)

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