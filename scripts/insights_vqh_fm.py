import logging
import os
import sys
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename_with_timestamp = f"evaluation_run_{timestamp}.log"

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, log_filename_with_timestamp)

logger = logging.getLogger()
logger.setLevel(logging.INFO) 

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)

file_handler = logging.FileHandler(log_filepath, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


import collections
import dataclasses
import logging
import math
import pathlib

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

# from vqh_training_policy import VQHPolicy
from vqh_training_policy_finetune_fm_only import VQHPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    policy_path: str = "outputs/train/2025-08-07/18-53-21_vqh/checkpoints/100000/pretrained_model"
    """Path to the pretrained VQHPolicy on the Hugging Face Hub or local directory."""
    task_suite_name: str = "libero_10"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    num_candidate_actions: int = 16
    # num_calql_samples: int = 24    
    """Number of candidate actions to sample at each step."""

    video_out_path: str = "data/libero/videos_vqh_debug"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random Seed (for reproducibility)"""

    foresee_steps: int = 25
    """Excute VQH Ranking Steps"""
    n_action_steps: int = 3
    """Normal Excute Action Steps"""
    g_action_steps: int = 5
    """Excute Gripper Action Steps"""
    efficient_eval: bool = True

    debug_branching: bool = False


import torch
import logging
import numpy as np

def log_evaluation_results(
    q_values, 
    best_idx, 
    best_sequence, 
    is_special_policy_winner, 
    special_policy_samples_count,
    special_policy_q_values=None,
    base_q_values=None,
    best_special_policy_action=None, 
    best_base_action=None,
    time_step=None,   
    special_policy_name="Cal-QL",
):
    """
    一个辅助函数，用于以美观、结构化的格式打印评估结果。
    """
    q_values_np = q_values.cpu().numpy().flatten()
    
    log_msgs = ["\n"] 
    log_msgs.append("=" * 70)
    
    log_msgs.append(f"- Time Step Now - : {time_step}")
    log_msgs.append("[Q-Value Evaluation]")    
    if special_policy_q_values is not None and special_policy_q_values.numel() > 0:
        # 使用通用名称
        special_q_str = ", ".join([f"{q:.4f}" for q in special_policy_q_values.cpu().numpy().flatten()])
        log_msgs.append(f"  - {special_policy_name} Q-Values:     [{special_q_str}]")
    
    if base_q_values is not None and base_q_values.numel() > 0:
        base_q_str = ", ".join([f"{q:.4f}" for q in base_q_values.cpu().numpy().flatten()])
        log_msgs.append(f"  - Base Policy Q-Values:  [{base_q_str}]")

    log_msgs.append("  - Summary:")
    log_msgs.append(f"    - Winning Candidate Idx: {best_idx}")
    log_msgs.append(f"    - Winning Q-Value:       {q_values_np[best_idx]:.4f}")
    log_msgs.append(f"    - Average Q-Value:       {np.mean(q_values_np):.4f}")
    log_msgs.append("-" * 70)

    log_msgs.append("[Decision]")
    if is_special_policy_winner:
        # 使用通用名称
        log_msgs.append(f"  - 🏆 VQH/{special_policy_name} 策略的动作 (候选者 {best_idx}) 被选中。")
    else:
        winner_idx_in_base = best_idx - special_policy_samples_count
        log_msgs.append(f"  - ✅ 基础策略的动作 (候选者 {winner_idx_in_base}) 被选中。")
    log_msgs.append("-" * 70)

    log_msgs.append("[Group Champions' Actions]")
    if best_special_policy_action is not None and special_policy_q_values is not None:
        best_special_q_val = torch.max(special_policy_q_values).item()
        # 使用通用名称
        log_msgs.append(f"  - Best {special_policy_name} Action (Q={best_special_q_val:.4f}):")
        best_special_np = best_special_policy_action.cpu().numpy()
        for i, step_action in enumerate(best_special_np):
            step_str = ", ".join([f"{x: .4f}" for x in step_action])
            log_msgs.append(f"    - Step {i+1}: [{step_str}]")
    else:
        log_msgs.append(f"  - No {special_policy_name} actions were sampled.")

    if best_base_action is not None and base_q_values is not None:
        best_base_q_val = torch.max(base_q_values).item()
        log_msgs.append(f"  - Best Base Action (Q={best_base_q_val:.4f}):")
        best_base_np = best_base_action.cpu().numpy()
        for i, step_action in enumerate(best_base_np):
            step_str = ", ".join([f"{x: .4f}" for x in step_action])
            log_msgs.append(f"    - Step {i+1}: [{step_str}]")
            
    log_msgs.append("-" * 70)
    
    log_msgs.append("[Overall Winning Action]")
    best_sequence_np = best_sequence.cpu().numpy()
    for i, step_action in enumerate(best_sequence_np):
        step_str = ", ".join([f"{x: .4f}" for x in step_action])
        log_msgs.append(f"  - Step {i+1}: [{step_str}]")
    
    log_msgs.append("=" * 70)
    logging.info("\n".join(log_msgs))

@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
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

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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
        max_steps = 520

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        if task_id != 0:
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
            # if episode_idx != 1:
            #     continue
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
            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    last_obs_for_branching = obs.copy()

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
                        if args.efficient_eval == True:                        
                            default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                            gripper_actions = default_sequence[:, : args.foresee_steps , -1]
    
                            has_positive = torch.any(gripper_actions > 0)
                            has_negative = torch.any(gripper_actions < 0)
                            is_critical_action = has_positive and has_negative

                            if is_critical_action and args.num_candidate_actions > 1:                                                                                    
                                logging.info("Gripper Action Detected. Evaluating multiple candidates with VQH.")
                                total_candidates = args.num_candidate_actions
                                
                                num_fm_samples = total_candidates // 2
                                num_l1_samples = total_candidates - num_fm_samples

                                candidate_sequences = [default_sequence]
                                for _ in range(num_l1_samples - 1):
                                    seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                                    candidate_sequences.append(seq)                                
                                candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3) 
                                l1_first_actions = candidate_sequences[:, :, 0, :]
                                all_first_actions = l1_first_actions
                                
                                if num_fm_samples > 0:
                                    fm_action_chunks = policy.sample_fm_action(observation, num_samples=num_fm_samples)
                                    fm_first_actions = fm_action_chunks[:, :, 0, :]
                                    all_first_actions = torch.cat([fm_first_actions, l1_first_actions], dim=1)
                               
                                q_values = policy.evaluate_q_values(observation, all_first_actions)

                                avg_q_value = torch.mean(q_values).item()
                                logging.info(f"Average Q-value of all candidates: {avg_q_value:.4f}")
                                
                                best_candidate_idx = torch.argmax(q_values, dim=1).item()
                                                                
                                best_fm_action_for_log = None
                                if num_fm_samples > 0:
                                    fm_q_values = q_values[0, :num_fm_samples]
                                    best_fm_idx_in_group = torch.argmax(fm_q_values).item()
                                    best_fm_action_for_log = fm_action_chunks[0, best_fm_idx_in_group]
                                    best_fm_action_for_log = best_fm_action_for_log[:args.g_action_steps]

                                base_q_values = q_values[0, num_fm_samples:]
                                best_base_idx_in_group = torch.argmax(base_q_values).item()
                                best_base_action_for_log = candidate_sequences[0, best_base_idx_in_group]
                                best_base_action_for_log = best_base_action_for_log[:args.g_action_steps]

                                OVERRIDE_THRESHOLD = 1.0
                                TIMESTEP_THRESHOLD = 200
                                max_base_q_value = torch.max(base_q_values).item()
                                
                                if max_base_q_value > OVERRIDE_THRESHOLD or t > TIMESTEP_THRESHOLD:
                                    logging.info(f"✅ Override rule triggered: Max base Q-value ({max_base_q_value:.4f}) > {OVERRIDE_THRESHOLD} or t > {TIMESTEP_THRESHOLD}. Forcing base policy action.")                                
                                    is_fm_winner = False
                                    best_action_sequence = best_base_action_for_log
                                    best_candidate_idx = num_fm_samples + best_base_idx_in_group
                                else:
                                    logging.info(f"Standard rule applies: Max base Q-value ({max_base_q_value:.4f}) <= {OVERRIDE_THRESHOLD}.")                                    
                                    if best_candidate_idx < num_fm_samples:
                                        is_fm_winner = True
                                        best_action_sequence = fm_action_chunks[0, best_candidate_idx]
                                        best_action_sequence = best_action_sequence[:args.g_action_steps]                             
                                    else:
                                        is_fm_winner = False
                                        l1_winner_idx = best_candidate_idx - num_fm_samples
                                        best_action_sequence = candidate_sequences[0, l1_winner_idx]
                                        best_action_sequence = best_action_sequence[:args.g_action_steps]   

                                log_evaluation_results(
                                    q_values=q_values,
                                    best_idx=best_candidate_idx,
                                    best_sequence=best_action_sequence,
                                    is_special_policy_winner=is_fm_winner,
                                    special_policy_samples_count=num_fm_samples,
                                    special_policy_q_values=q_values[0, :num_fm_samples] if num_fm_samples > 0 else None,
                                    base_q_values=q_values[0, num_fm_samples:],
                                    best_special_policy_action=best_fm_action_for_log,
                                    best_base_action=best_base_action_for_log,
                                    time_step=t,
                                    special_policy_name="Flow Matching"
                                )
                            else:
                                logging.info("Evaluating Default Action Sequence.")
                                best_action_sequence = default_sequence[0]
                                best_action_sequence = best_action_sequence[:args.n_action_steps]

                        else:
                            num_l1_candidates = args.num_candidate_actions - 1
                            calql_action_chunk = policy.sample_calql_action(observation)
                            candidate_sequences = []
                            if num_l1_candidates > 0:
                                for _ in range(num_l1_candidates):
                                    seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                                    candidate_sequences.append(seq)
                                candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3)
                                l1_first_actions = candidate_sequences[:, :, 0, :] 
                                all_first_actions = torch.cat([calql_action_chunk, l1_first_actions], dim=1)
                            else:
                                all_first_actions = calql_action_chunk

                            batch_size, num_candidates, feat_dim = all_first_actions.shape
                            flat_first_actions = all_first_actions.reshape(batch_size, num_candidates, -1)
                            q_values = policy.evaluate_q_values(observation, flat_first_actions)
                            
                            
                    best_action_sequence_np = best_action_sequence.cpu().numpy()
                    pre_sequence_state = state
                    pre_sequence_agentview = agentview_image
                    pre_sequence_wrist = wrist_img

                    for action_in_sequence in best_action_sequence_np:
                        if t >= max_steps:
                            break              

                        data_step = {
                            "agentview_image": pre_sequence_agentview,
                            "wrist_image": pre_sequence_wrist,
                            "state": pre_sequence_state,
                            "action": action_in_sequence,
                        }
                        episode_data_buffer.append(data_step)


                        obs, _, done, _ = env.step(action_in_sequence)                        

                        current_agent_frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        agent_frames.append(current_agent_frame) 
                        
                        current_wrist_frame = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        wrist_frames.append(current_wrist_frame)
                       
                        t += 1
                        if done:
                            break
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}", exc_info=True) # exc_info=True 可以打印更详细的traceback
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
            
            wrist_video_path = output_path / f"{base_filename}_wrist.mp4"
            imageio.mimsave(wrist_video_path, wrist_frames, fps=fps)
            logging.info(f"Saved wrist view video to {wrist_video_path}")

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


# _get_libero_env and _quat2axisangle functions remain unchanged
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
    logging.basicConfig(level=logging.INFO)
    eval_libero()