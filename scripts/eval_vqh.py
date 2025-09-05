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
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
from vqh_training_policy import VQHPolicy
from libero.libero.envs.predicates.base_predicates import PositionUnchanged

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:
    policy_path: str = "outputs/train/2025-09-02/18-38-39_vqh/checkpoints/025000/pretrained_model"
    """Path to the pretrained VQHPolicy on the Hugging Face Hub or local directory."""    
    task_suite_name: str = "libero_elegant"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""
    num_candidate_actions: int = 16
    """Number of candidate actions to sample at each step."""
    video_out_path: str = "data/libero_elegant/eval_vqh_25k_task_4"
    stat_script_path: str = "scripts/stat.sh" 
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random Seed (for reproducibility)"""

    foresee_steps: int = 10
    """Excute VQH Ranking Steps"""
    n_action_steps: int = 3
    """Normal Excute Action Steps"""
    g_action_steps: int = 3

    """Excute Gripper Action Steps"""
    vqh_chunk_size: int = 3
    
    efficient_eval: bool = False
    debug_branching: bool = False

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_with_timestamp = f"evaluation_run_{timestamp}.log"

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
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有 DEBUG 及以上级别
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)   # 控制台只显示 INFO 及以上级别
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"日志将记录到文件: {log_filepath}")

def log_evaluation_results(
    q_values,
    best_idx,
    best_sequence,
    base_q_values=None,
    best_base_action=None,
    time_step=None,
    base_action_sequences=None,
):
    """
    - Uses INFO level to log summary and final decision.
    - Uses DEBUG level to log a detailed list of all candidate actions.
    """
    q_values_np = q_values.cpu().numpy().flatten()

    # --- Part 1: Summary Info (INFO Level) ---
    summary_msgs = ["\n"]
    summary_msgs.append("=" * 70)
    summary_msgs.append(f"- Time Step Now - : {time_step}")
    summary_msgs.append("[Q-Value Evaluation]")
    if base_q_values is not None and base_q_values.numel() > 0:
        base_q_str = ", ".join([f"{q:.4f}" for q in base_q_values.cpu().numpy().flatten()])
        summary_msgs.append(f"  - Base Policy Q-Values:  [{base_q_str}]")
    summary_msgs.append("  - Summary:")
    summary_msgs.append(f"    - Winning Candidate Idx: {best_idx}")
    summary_msgs.append(f"    - Winning Q-Value:       {q_values_np[best_idx]:.4f}")
    summary_msgs.append(f"    - Average Q-Value:       {np.mean(q_values_np):.4f}")
    logging.info("\n".join(summary_msgs))

    # --- Part 2: Detailed Candidate Info (DEBUG Level) ---
    details_msgs = ["-" * 70]
    details_msgs.append("[All Candidate Actions]")
    if base_action_sequences is not None and base_action_sequences.numel() > 0:
        details_msgs.append("  - Base Policy Actions:")
        for i, sequence in enumerate(base_action_sequences):
            q_value = q_values_np[i]
            details_msgs.append(f"    - Candidate {i} (Q={q_value:.4f}):")
            sequence_np = sequence.cpu().numpy()[:5]
            for step_idx, step_action in enumerate(sequence_np):
                if step_action.ndim == 0:
                    step_str = f"{step_action.item(): .4f}"
                else:
                    step_str = ", ".join([f"{x: .4f}" for x in step_action])
                details_msgs.append(f"      - Step {step_idx+1}: [{step_str}]")
    else:
        details_msgs.append("  - No candidate sequences were provided for logging.")
    logging.debug("\n".join(details_msgs))

    # --- Part 3: Final Decision Info (INFO Level) ---
    decision_msgs = ["-" * 70]
    decision_msgs.append("[Decision]")
    decision_msgs.append(f"  - ✅ Base policy action (Candidate {best_idx}) was selected.")
    decision_msgs.append("-" * 70)
    decision_msgs.append("[Group Champions' Actions]")

    if best_base_action is not None and base_q_values is not None and base_q_values.numel() > 0:
        best_base_q_val = torch.max(base_q_values).item()
        decision_msgs.append(f"  - Best Base Action (Q={best_base_q_val:.4f}):")
        best_base_np = best_base_action.cpu().numpy()
        for i, step_action in enumerate(best_base_np):
            if step_action.ndim == 0:
                step_str = f"{step_action.item(): .4f}"
            else:
                step_str = ", ".join([f"{x: .4f}" for x in step_action])
            decision_msgs.append(f"    - Step {i+1}: [{step_str}]")
    else:
        decision_msgs.append("  - No base actions were sampled.")

    decision_msgs.append("-" * 70)
    decision_msgs.append("[Overall Winning Action]")
    best_sequence_np = best_sequence.cpu().numpy()
    for i, step_action in enumerate(best_sequence_np):
        if step_action.ndim == 0:
            step_str = f"{step_action.item(): .4f}"
        else:
            step_str = ", ".join([f"{x: .4f}" for x in step_action])
        decision_msgs.append(f"  - Step {i+1}: [{step_str}]")

    decision_msgs.append("=" * 70)
    logging.info("\n".join(decision_msgs))

def log_default_action(sequence, time_step):
    """Logs the default action sequence when it's used as a fallback."""
    log_msgs = ["\n"]
    log_msgs.append("=" * 70)
    log_msgs.append(f"- Time Step Now - : {time_step}")
    log_msgs.append("[Decision]")
    log_msgs.append("  - ⚙️ Default action sequence was chosen as a fallback.")
    log_msgs.append("-" * 70)
    log_msgs.append("[Default Action Executed]")
    
    sequence_np = sequence.cpu().numpy()
    for i, step_action in enumerate(sequence_np):
        if step_action.ndim == 0:
            step_str = f"{step_action.item(): .4f}"
        else:
            step_str = ", ".join([f"{x: .4f}" for x in step_action])
        log_msgs.append(f"  - Step {i+1}: [{step_str}]")
    
    log_msgs.append("=" * 70)
    logging.info("\n".join(log_msgs))

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

    video_dir = pathlib.Path(args.video_out_path)

    # if video_dir.exists():
    #     logging.warning(f"Video directory {video_dir} already exists. Clearing its contents.")
    #     shutil.rmtree(video_dir)
    # video_dir.mkdir(parents=True)

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
        max_steps = 220

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):

        if task_id != 4: 
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
            
            if episode_idx >= 30:
                continue
    
            PositionUnchanged.start_new_episode(episode_idx)
            logging.info(f"\nTask: {task_description}")
            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)            
            
            t = 0            

            agent_frames = []
            wrist_frames = []

            episode_data_buffer = [] 
            done = False
            
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            agent_frames.append(agentview_image)

            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            wrist_frames.append(agentview_image)
            
            logging.info(f"Starting episode {task_episodes+1}...")
            
            last_gripper_action = 0.0 

            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    agent_frames.append(agentview_image)
                    wrist_frames.append(wrist_img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    policy.vqh.smolvla_policy.to(torch.float32)
                                        
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    with torch.inference_mode():
                        if args.efficient_eval:
                            default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                            current_gripper_intent = default_sequence[0, 0, -1].item()
                            gripper_threshold = 0
                            
                            is_intent_flipped = (last_gripper_action > gripper_threshold and current_gripper_intent < -gripper_threshold) or \
                                                (last_gripper_action < -gripper_threshold and current_gripper_intent > gripper_threshold)

                            gripper_actions = default_sequence[:, : args.foresee_steps, -1]
                            has_positive = torch.any(gripper_actions > 0)
                            has_negative = torch.any(gripper_actions < 0)
                            is_internal_flip = has_positive and has_negative
                            is_critical_action = is_intent_flipped or is_internal_flip

                            if is_critical_action and args.num_candidate_actions > 1:
                                logging.info("Gripper Action Detected. Evaluating multiple candidates with VQH.")
                                
                                # Sample N candidates from the base policy
                                candidate_sequences = [default_sequence]
                                for _ in range(args.num_candidate_actions - 1):
                                    seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                                    candidate_sequences.append(seq)
                                
                                candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3)
                                
                                # Prepare chunks for Q-value evaluation
                                all_first_actions = candidate_sequences[:, :, :args.vqh_chunk_size, :]
                                q_values = policy.evaluate_q_values(observation, all_first_actions)
                                avg_q_value = torch.mean(q_values).item()
                                logging.info(f"Average Q-value of all candidates: {avg_q_value:.4f}")

                                # Choose the best candidate using the hybrid strategy
                                best_candidate_idx = choose_best_action_hybrid(
                                    q_values=q_values,
                                    candidate_sequences=candidate_sequences,
                                    top_k=args.num_candidate_actions,
                                    cluster_threshold=0.2,
                                    q_gap_threshold=0.15
                                )
                                
                                # The winner is always from the base policy candidates
                                best_action_sequence = candidate_sequences[0, best_candidate_idx]
                                best_action_sequence = best_action_sequence[:args.g_action_steps]
                                
                                # Prepare logs for the base policy candidates
                                base_q_values = q_values[0]
                                best_base_idx_in_group = torch.argmax(base_q_values).item()
                                best_base_action_for_log = candidate_sequences[0, best_base_idx_in_group]
                                best_base_action_for_log = best_base_action_for_log[:args.g_action_steps]
                                
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
                                logging.info("Evaluating Default Action Sequence.")
                                best_action_sequence = default_sequence[0]
                                best_action_sequence = best_action_sequence[:args.n_action_steps]
                                log_default_action(
                                    sequence=best_action_sequence,
                                    time_step=t
                                )
                        else:
                            logging.info("Efficient eval is OFF. Evaluating multiple candidates with VQH.")
                            default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation)

                            if t > 150:
                                candidate_sequences = [default_sequence]
                                for _ in range(args.num_candidate_actions-1):
                                    seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                                    candidate_sequences.append(seq)
                                candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3)

                                all_first_actions = candidate_sequences[:, :, :args.vqh_chunk_size, :]
                                q_values = policy.evaluate_q_values(observation, all_first_actions)
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
                                logging.info("Evaluating Default Action Sequence.")
                                best_action_sequence = default_sequence[0]
                                best_action_sequence = best_action_sequence[:args.n_action_steps]
                                log_default_action(
                                    sequence=best_action_sequence,
                                    time_step=t
                                )

                    best_action_sequence_np = best_action_sequence.cpu().numpy()
                                
                    pre_sequence_state = state
                    pre_sequence_agentview = agentview_image
                    pre_sequence_wrist = wrist_img

                    for action_in_sequence in best_action_sequence_np:
                        if t >= max_steps:
                            break         


                        modified_action = action_in_sequence.copy()    

                        gripper_action_original = modified_action[6]

                        if gripper_action_original > 0.5:
                            modified_action[6] = 1.0
                        elif gripper_action_original < -0.5:
                            modified_action[6] = -1.0
                        else:
                            modified_action[6] = 0.0     

                        data_step = {
                            "agentview_image": pre_sequence_agentview,
                            "wrist_image": pre_sequence_wrist,
                            "state": pre_sequence_state,
                            "action": action_in_sequence,
                        }
                        episode_data_buffer.append(data_step)
                        
                        # obs, _, done, _ = env.step(action_in_sequence)                        
                        obs, _, done, _ = env.step(modified_action)                        

                        current_agent_frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        agent_frames.append(current_agent_frame) 
                        
                        current_wrist_frame = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        wrist_frames.append(current_wrist_frame)
                       
                        t += 1
                        if done:
                            break
                    
                    if len(best_action_sequence_np) > 0:
                        last_gripper_action = best_action_sequence_np[-1, -1]

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
            
            # wrist_video_path = output_path / f"{base_filename}_wrist.mp4"
            # imageio.mimsave(wrist_video_path, wrist_frames, fps=fps)
            # logging.info(f"Saved wrist view video to {wrist_video_path}")

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
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    cv2.destroyAllWindows()

    logging.info("--- Running statistics script ---")
    try:
        script_path = args.stat_script_path
        video_path = args.video_out_path
        
        result = subprocess.run(
            [script_path, video_path],
            check=True, 
            capture_output=True, 
            text=True
        )
        
        logging.info("Statistics script finished successfully.")
        logging.info(f"Script output:\n---\n{result.stdout.strip()}\n---")
    except FileNotFoundError:
        logging.error(f"Error: The script at '{script_path}' was not found.")
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