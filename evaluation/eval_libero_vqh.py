import dataclasses
import logging
import math
import os
import pathlib
import collections
import subprocess
import sys
import draccus
import imageio
import numpy as np
import torch
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from src.vqh_training_policy import VQHPolicy 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_utils.utils import (
    setup_global_logging,
    setup_episode_logging,
    setup_task_predicates,     
    get_action_via_volatility, 
    get_action_via_advantage,    
    save_q_range_analysis,
    save_q_value_curve,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
os.environ["HF_HUB_OFFLINE"] = "1"
INTERVENTION_STRATEGY = "advantage"

@dataclasses.dataclass
class Args:
    policy_path: str = "outputs/train/2025-12-07/01-02-10_vqh/checkpoints/010000/pretrained_model"
    
    # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_elegant, libero_pragmatic, libero_object_elegant
    task_suite_name: str = "libero_elegant"
    
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    max_steps: int = 220

    output_root_dir: str = "rollouts/libero_elegant_1212"
    experiment_name: str = "Full_Task_VQH_Advan_0.05"
    stat_script_path: str = "scripts/stat.sh" 

    full_intervention_mode: bool = False
    q_history_window_size: int = 5
    q_change_threshold: float = 0.05
    num_candidate_actions: int = 16

    advantage_threshold: float = 0.05
    
    n_action_steps: int = 10
    g_action_steps: int = 10
    vqh_chunk_size: int = 10

    episode_idx_to_run: int = -1
    task_id_to_run: int = -1

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
    video_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = video_dir / "q_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading VQHPolicy from {args.policy_path}")
    policy = VQHPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):

        if args.task_id_to_run != -1 and task_id != args.task_id_to_run:
            continue

        if task_id == 3:
            max_steps = 550
        else:
            max_steps = args.max_steps     

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

            setup_task_predicates(task_id, episode_idx, enable_logging=False)
            logging.info(f"\nTask: {task_description}")

            setup_episode_logging(task_id=task_id,episode_idx=episode_idx,base_dir=video_dir)

            env.reset()
            policy.reset()

            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)            
            
            t = 0
            agentview_frames = []
            wrist_frames = []

            episode_interventions = []
            episode_default_qs = []
            episode_max_qs = []
            episode_min_qs = []
            episode_advantages = []
            episode_interventions = []
            
            done = False
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            
            logging.info(f"Starting episode {task_episodes+1}...")
            q_history = collections.deque(maxlen=args.q_history_window_size)
            intervention_snapshots = {}

            while t < max_steps:
                try:
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    agentview_frames.append(agentview_image)
                    wrist_frames.append(wrist_img)

                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))

                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    img_to_save = agentview_image.astype(np.uint8)
                    intervention_snapshots[t] = img_to_save

                    with torch.inference_mode():
                        policy.vqh.smolvla_policy.to(torch.float32)
                        
                        default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                        if INTERVENTION_STRATEGY == "volatility":
                            best_action_sequence, stats = get_action_via_volatility(
                                policy=policy,
                                observation=observation,
                                default_sequence=default_sequence,
                                q_history=q_history,
                                args=args,
                                t=t
                            )
                        elif INTERVENTION_STRATEGY == "advantage":
                            best_action_sequence, stats = get_action_via_advantage(
                                policy=policy,
                                observation=observation,
                                default_sequence=default_sequence,
                                args=args,
                                t=t,
                                advantage_threshold=args.advantage_threshold,
                                num_probe_actions=args.num_candidate_actions 
                            )

                        cur_q = stats.get('current_q', 0.0)
                        episode_default_qs.append(cur_q)
                        episode_max_qs.append(stats.get('max_q', cur_q))
                        episode_min_qs.append(stats.get('min_q', cur_q))

                        metric_val = stats.get('advantage', stats.get('q_change', 0.0))
                        episode_advantages.append(metric_val)
                        episode_interventions.append(stats.get('intervened', False))

                        best_action_sequence_np = best_action_sequence.cpu().numpy()



                    for action_in_sequence in best_action_sequence_np:
                        if t >= max_steps: break  
                        
                        agentview_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        wrist_frames.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))

                        modified_action = action_in_sequence.copy()
                        gripper_action_original = modified_action[6]
                        if gripper_action_original > 0.5: modified_action[6] = 1.0
                        elif gripper_action_original < -0.5: modified_action[6] = -1.0
                        else: modified_action[6] = 0.0                        
                        obs, _, done, _ = env.step(modified_action)
                        # obs, _, done, _ = env.step(action_in_sequence)                        
                        t += 1

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

            # --- Video Saving ---
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_filename = f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
            video_path = video_dir / video_filename
            
            writer = imageio.get_writer(video_path, fps=30)
            for img_agent, img_wrist in zip(agentview_frames, wrist_frames):
                combined = np.hstack((img_agent, img_wrist))
                writer.append_data(combined)
            writer.close()
            
            logging.info(f"Saved combined video to {video_path}")

            logging.info(f"Saving Q-value plot...")
            
            # 根据策略决定显示的名称
            # if INTERVENTION_STRATEGY == "advantage":
            #     metric_label = "Advantage (MaxQ - CurQ)"
            #     # 此时 episode_q_deltas 里存的是 advantage
            # else:
            #     metric_label = "Volatility (|Q - SMA|)"
            #     # 此时 episode_q_deltas 里存的是 delta

            # save_q_value_curve(
            #     q_values=episode_q_values,
            #     metric_values=episode_q_deltas,  # 这里传入你收集的列表（可能是 Delta 也可能是 Advantage）
            #     interventions=episode_interventions,
            #     threshold=args.q_change_threshold if INTERVENTION_STRATEGY == "volatility" else 0.05, # 注意：这里的阈值参数可能也需要根据策略传不同的
            #     task_id=task_id,
            #     episode_idx=episode_idx,
            #     task_name=task_description,
            #     save_dir=plot_dir,
            #     outcome=suffix,
            #     metric_name=metric_label # 传入新参数
            # )

            save_q_range_analysis(
                default_q_values=episode_default_qs,
                max_q_values=episode_max_qs,
                min_q_values=episode_min_qs,
                avg_q_values=None,           # 我们暂时没存平均值，不需要画
                metric_values=episode_advantages, # 这里存的是 Advantage (或 Delta)
                interventions=episode_interventions,
                threshold=args.advantage_threshold, # 传入对应的阈值
                task_id=task_id,
                episode_idx=episode_idx,
                task_name=task_description,  # 对应的任务描述字符串
                save_dir=plot_dir,           # 之前定义的保存路径
                outcome=suffix,               # "success" 或 "failure"
                intervention_snapshots=intervention_snapshots
            )

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
    setup_global_logging()
    eval_libero()