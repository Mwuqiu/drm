import dataclasses
import logging
import math
import os
import sys
import pathlib
import shutil
import draccus
import imageio
import torch
import numpy as np
from tqdm import tqdm
import subprocess

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data_utils.utils import setup_logging, log_default_action, setup_task_predicates

os.environ["HF_HUB_OFFLINE"] = "1" 

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:
    policy_path: str="/home/wuqiu/lerobot/outputs/train/2025-12-04/15-18-52_smolvla/checkpoints/025000/pretrained_model"

    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_elegant, libero_pragmatic, libero_object_elegant"""    
    task_suite_name: str = "libero_elegant"    

    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_steps: int = 220
    
    output_root_dir: str = "rollouts/libero_elegant_1206"
    experiment_name: str = "Full_Task"

    stat_script_path: str = "scripts/stat.sh" 

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
    if not video_dir.exists():
        video_dir.mkdir(parents=True)
    
    
    # --- Load Policy ---
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.n_action_steps = 10
    policy.to(args.device)
    policy.eval()

    # --- Evaluation Loop ---
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

            env.reset()
            policy.reset()

            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            t = 0
            agentview_frames = []
            wrist_frames = []

            done = False
            has_succeeded = False

            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    agentview_frames.append(agentview_image)
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

                    obs, _, done, _ = env.step(modified_action)  
                    
                                    
                    if done and not has_succeeded:
                        has_succeeded = True
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if has_succeeded else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_path = pathlib.Path(video_dir) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
            
            writer = imageio.get_writer(video_path, fps=30)

            for img_agent, img_wrist in zip(agentview_frames, wrist_frames):
                combined = np.hstack((img_agent, img_wrist))
                writer.append_data(combined)
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
    subprocess.run([args.stat_script_path, video_dir], check=False)

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