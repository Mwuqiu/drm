import collections
import dataclasses
import logging
import math
import os
import pathlib
import shutil
import sys

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
import subprocess

from libero.libero.envs.predicates.base_predicates import PositionUnchanged
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    
    # --- Hugging Face arguments ---
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_long/checkpoints/100000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_b200s100/checkpoints/100000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_object/checkpoints/100000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_spatial_task_9/checkpoints/010000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_long_task_5/checkpoints/010000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_goal/checkpoints/100000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_long_task_5/checkpoints/100000/pretrained_model"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/object_elegant_hybird_collect/checkpoints/010000/pretrained_model"

    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-09-01/13-35-06_smolvla/checkpoints/last/pretrained_model"
    policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-09-02/20-41-55_smolvla/checkpoints/060000/pretrained_model"

    # --- LIBERO environment-specific parameters --- 

    task_suite_name: str = "libero_elegant"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_elegant, libero_pragmatic"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Evaluation arguments ---

    video_out_path: str = "data/libero_elegant/eval_elegant_smolvla_60k_task_7"
    """Path to save videos."""
    stat_script_path: str = "scripts/stat.sh" 
    """Path to stat file."""
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random Seed (for reproducibility)"""
    


@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()

    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. " f"Available options are: {list(benchmark_dict.keys())}")
    
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    
    video_dir = pathlib.Path(args.video_out_path)
    if video_dir.exists():
        logging.warning(f"Video directory {video_dir} already exists. Clearing its contents.")
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True)
    logging.info(f"Created fresh video directory: {video_dir}")
    
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        max_steps = 250
        
    # --- Load Policy ---
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.n_action_steps = 10
    policy.to(args.device)
    policy.eval()

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):

        if task_id != 7:
            continue

        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ): 
            
            PositionUnchanged.start_new_episode(episode_idx)

            logging.info(f"\nTask: {task_description}")
            env.reset()
            policy.reset()

            obs = env.set_init_state(initial_states[episode_idx])

            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            t = 0
            frames = []
            wrist_frames = []
            done = False

            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            wrist_frames.append(agentview_image)
            
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    if t == 0 and episode_idx == 0 and task_id == 0:
                        print(f"正在保存诊断图片到: diagnostic_frame.png")
                        imageio.imwrite("diagnostic_frame.png", agentview_image)

                    frames.append(agentview_image)
                    wrist_frames.append(wrist_img)

                    # Prepare observations dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    with torch.inference_mode():
                        action_tensor = policy.select_action(observation)
                    action = action_tensor.cpu().numpy()[0]


                    modified_action = action.copy()    

                    gripper_action_original = modified_action[6]

                    if gripper_action_original > 0.5:
                        modified_action[6] = 1.0
                    elif gripper_action_original < -0.5:
                        modified_action[6] = -1.0
                    else:
                        modified_action[6] = 0.0 

                    # obs, _, done, _ = env.step(action)
                    obs, _, done, _ = env.step(modified_action)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
            wrist_video_path = pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}_wrist.mp4"
            fps = 30
            writer = imageio.get_writer(video_path, fps=fps)

            for image in frames:
                writer.append_data(image)
            writer.close()
            logging.info(f"Saved video to {video_path}")

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
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    from datetime import datetime
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

    
    eval_libero()