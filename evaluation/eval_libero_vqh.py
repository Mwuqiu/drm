import logging
import os
import subprocess
import numpy as np
import torch
import dataclasses
import math
import pathlib
import draccus
import imageio
import time
import collections

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
from vqh_training_policy import VQHPolicy
from libero.libero.envs.predicates.base_predicates import PositionUnchanged

from data_utils.utils import setup_logging, log_default_action, log_evaluation_results, choose_best_action_hybrid

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:        
    policy_path: str = "outputs/train/2025-09-20/15-55-26_vqh/checkpoints/008000/pretrained_model"
    
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_elegant, libero_pragmatic, libero_object_elegant"""
    task_suite_name: str = "libero_elegant"
    
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    output_root_dir: str = "data/libero_elegant_1022/eval_task_CS10"
    experiment_name: str = "Full_Task_4"
    
    full_intervention_mode: bool = False

    q_history_window_size: int = 10
    q_change_threshold: float = 0.05
    num_candidate_actions: int = 8

    max_steps: int = 520
    continue_on_done: bool = False

    n_action_steps: int = 10
    g_action_steps: int = 10
    vqh_chunk_size: int = 10
    
    stat_script_path: str = "scripts/stat.sh"
    
    episode_idx_to_run: int = -1
    task_id_to_run: int = 4

    device: str = "cuda"
    seed: int = 7
    

@draccus.wrap()
def eval_libero(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
   
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. " f"Available options are: {list(benchmark_dict.keys())}")
   
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    
    final_output_dir = os.path.join(args.output_root_dir, args.experiment_name)
    video_dir = pathlib.Path(final_output_dir)
    if not video_dir.exists():
        video_dir.mkdir(parents=True)

    logging.info(f"Loading VQHPolicy from {args.policy_path}")
    policy = VQHPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

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

            intervention_count = 0
            intervention_decision_times = []
            default_decision_times = []
            
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
    subprocess.run([args.stat_script_path, video_dir], check=False)

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