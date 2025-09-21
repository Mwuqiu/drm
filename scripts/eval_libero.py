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
from vqh_logger import setup_logging, log_evaluation_results, log_default_action

from libero.libero.envs.predicates.base_predicates import PositionUnchanged
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    
    # --- Hugging Face arguments ---
    policy_path: str="/home/wuqiu/lerobot/outputs/train/2025-09-12/16-46-10_smolvla/checkpoints/100000/pretrained_model"

    # --- LIBERO environment-specific parameters --- 
    task_suite_name: str = "libero_elegant"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_elegant, libero_pragmatic"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "data/libero_elegant_0920/eval_smolvla_task_4_oeoe_220"
    exp_specific_des: str = "Task-4-SmolVLA"


    """Path to save videos."""
    stat_script_path: str = "scripts/stat.sh" 
    """Path to stat file."""
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random Seed (for reproducibility)"""

    # --- Dataset Saving Arguments ---
    save_dataset: bool = False
    dataset_repo_id: str = "libero_elegant_0920/task_4_collected"
    episode_idx_to_run: int = -1
    

@draccus.wrap()
def eval_libero(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    benchmark_dict = benchmark.get_benchmark_dict()

    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. " f"Available options are: {list(benchmark_dict.keys())}")
    
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    args.video_out_path = os.path.join(args.video_out_path, args.exp_specific_des)    
    video_dir = pathlib.Path(args.video_out_path)
    if not video_dir.exists():
        video_dir.mkdir(parents=True)

    # if video_dir.exists():
    #     logging.warning(f"Video directory {video_dir} already exists. Clearing its contents.")
    #     shutil.rmtree(video_dir)
    # video_dir.mkdir(parents=True)
    # logging.info(f"Created fresh video directory: {video_dir}")
    
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 220
        
    # --- Load Policy ---
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.n_action_steps = 10
    policy.to(args.device)
    policy.eval()

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

            if episode_idx != 5:
                continue       

            # if args.episode_idx_to_run != -1 and episode_idx != args.episode_idx_to_run:
            #     continue

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
            episode_buffer = []

            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            wrist_frames.append(agentview_image)
            
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    frames.append(agentview_image)
                    wrist_frames.append(wrist_img)

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
                        log_default_action(sequence=action_tensor, time_step=t)
                    action = action_tensor.cpu().numpy()[0]                    
                    modified_action = action.copy()    
                    gripper_action_original = modified_action[6]
                    if gripper_action_original > 0.5:
                        modified_action[6] = 1.0
                    elif gripper_action_original < -0.5:
                        modified_action[6] = -1.0
                    else:
                        modified_action[6] = 0.0 

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
            video_path = pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
            
            writer = imageio.get_writer(video_path, fps=30)
            for image in frames:
                writer.append_data(image)
            writer.close()
            logging.info(f"Saved video to {video_path}")

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