import argparse
import collections
import datetime
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
import os
from typing import Optional

import cv2
import h5py
import imageio
import numpy as np
import robosuite as suite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, bddl_utils
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from tqdm import tqdm

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def is_noop(state, prev_state=None, pos_threshold=1e-4, gripper_threshold=1e-4):
    """
    通过比较当前和上一帧的状态（state）来判断是否为无操作。
    一个无操作满足两个条件：
    (1) 手臂末端执行器的位置和姿态变化很小。
    (2) 夹爪的关节位置变化很小。
    """
    if prev_state is None:
        return False 
    eef_change = np.linalg.norm(state[:6] - prev_state[:6])
    gripper_change = np.linalg.norm(state[6:] - prev_state[6:])
    return eef_change < pos_threshold and gripper_change < gripper_threshold


def _quat2axisangle(quat):
    # ... (this function is the same)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den



def collect_human_trajectory(
    env,
    device,
    arm,
    env_configuration,
    dataset,
    task_description,
    temp_img_dir,
):
    """
    Use a device (keyboard or SpaceNav 3D mouse) to collect a single demonstration
    and save it to the LeRobotDataset.

    Args:
        env (MujocoEnv): Environment to control.
        device (Device): Device for receiving controls.
        arm (str): Which arm to control ('right' or 'left').
        env_configuration (str): Specified environment configuration.
        dataset (LeRobotDataset): The dataset object to save the trajectory to.
        task_description (str): The language instruction for the task.
        temp_img_dir (Path): Temporary directory to store images before adding to the dataset.
    """
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
            
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = -1
    device.start_control()

    wrist_cam_name = "robot0_eye_in_hand_image"
    agent_cam_name = "agentview_image"
    cv2.namedWindow("Agent Camera View", cv2.WINDOW_AUTOSIZE)

    # Loop until we get a reset from the input or the task completes
    should_save = True
    t = 0
    record_last_n_steps = None

    # episode_buffer = []
    episode_buffer = collections.deque(maxlen=record_last_n_steps)

    prev_state = None
    env_init = True
    

    # ... 在 while True 循环之前 ...
    # 记录循环开始的时间
    start_time = time.time()

    while True:
        # 检查任务完成状态 (这部分逻辑不变)
        # if task_completion_hold_count == 0:
        #     logging.info("Task completed successfully.")
        #     break
        
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        # 获取用户输入并执行环境步骤 (这部分逻辑总是在运行)
        active_robot = env.robots[0]
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )
        if action is None:
            logging.info("Resetting episode. This trajectory will not be saved.")
            should_save = False
            break
        obs, reward, done, info = env.step(action)
        
        # 显示摄像头画面 (这部分逻辑总是在运行)
        if agent_cam_name in obs:
            agent_img_rgb = np.flipud(obs[agent_cam_name])
            agent_img_bgr = cv2.cvtColor(agent_img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Agent Camera View", agent_img_bgr)
        cv2.waitKey(1)
        env.render()

        # ===============================================================
        # 核心修改：只有在循环开始1秒后，才执行下面的数据记录逻辑
        if time.time() - start_time > 1.0:
        
            # --- Data Recording Logic ---
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
            
            if is_noop(state, prev_state):
                continue

            step_data = {
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
            }

            next_episode_index = dataset.meta.total_episodes
            for key, img_array in [
                ("observation.images.image", agent_img),
                ("observation.images.wrist_image", wrist_img),
            ]:
                episode_img_dir = temp_img_dir / key / f"episode_{next_episode_index:06d}"
                episode_img_dir.mkdir(parents=True, exist_ok=True)
                img_path = episode_img_dir / f"frame_{t:06d}.png"
                imageio.imwrite(img_path, img_array)
                step_data[key] = str(img_path)

            
            if record_last_n_steps is not None and len(episode_buffer) == record_last_n_steps:
                oldest_step = episode_buffer[0]
                for key in ["observation.images.image", "observation.images.wrist_image"]:
                    if key in oldest_step:
                        try:
                            os.remove(oldest_step[key])
                        except OSError as e:
                            logging.warning(f"Could not delete old image file {oldest_step[key]}: {e}")


            episode_buffer.append(step_data)
            t += 1
            prev_state = state

    cv2.destroyAllWindows()

    if should_save and episode_buffer:
        logging.info(f"Saving successful episode to dataset {dataset.repo_id}")
        episode_data_dict = collections.defaultdict(list)
        for step in episode_buffer:
            for key, value in step.items():
                episode_data_dict[key].append(value)

        episode_length = len(episode_buffer)
        episode_data_dict["frame_index"] = list(range(episode_length))
        episode_data_dict["timestamp"] = [i / env.control_freq for i in range(episode_length)]
        
        start_index = dataset.meta.total_frames
        episode_data_dict["index"] = list(range(start_index, start_index + episode_length))
        
        task_index = dataset.meta.get_task_index(task_description)
        if task_index is None:
            dataset.meta.add_task(task_description)
            task_index = dataset.meta.get_task_index(task_description)
        episode_data_dict["task_index"] = [task_index] * episode_length

        episode_index_to_save = dataset.meta.total_episodes
        episode_data_dict["episode_index"] = episode_index_to_save
        episode_data_dict["size"] = episode_length
        episode_data_dict["task"] = [task_description] * episode_length
        
        dataset.save_episode(episode_data=episode_data_dict)

        # Clean up temporary image files for the saved episode
        for key in ["observation.images.image", "observation.images.wrist_image"]:
            saved_episode_img_dir = temp_img_dir / key / f"episode_{episode_index_to_save:06d}"
            if saved_episode_img_dir.is_dir():
                shutil.rmtree(saved_episode_img_dir)
        return True
        
    # If not saving, clean up the directory created for the aborted episode
    elif not should_save:
        next_episode_index = dataset.meta.total_episodes
        for key in ["observation.images.image", "observation.images.wrist_image"]:
            aborted_episode_img_dir = temp_img_dir / key / f"episode_{next_episode_index:06d}"
            if aborted_episode_img_dir.is_dir():
                shutil.rmtree(aborted_episode_img_dir)
        return False
        
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="libero_elegant_collect/temp",
        help="Name of the dataset repository on the Hugging Face Hub.",
    )
    parser.add_argument("--bddl-file", type=str, required=True, help="Path to the BDDL file for the task.")
    parser.add_argument(
        "--num-demonstrations",
        type=int,
        default=10,
        help="Number of demonstration episodes to collect.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["Panda"],
        help="Which robot(s) to use in the env.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control ('right' or 'left').",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_eye_in_hand",
        help="Which camera to use for collecting demos.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller ('IK_POSE' or 'OSC_POSE').",
    )
    parser.add_argument("--device", type=str, default="spacemouse", help="'keyboard' or 'spacemouse'.")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    parser.add_argument("--push-to-hub", action="store_true", help="Whether to push the dataset to the hub.")

    args = parser.parse_args()

    # --- 1. LeRobot Dataset Initialization ---
    logging.info(f"Initializing dataset: {args.dataset_repo_id}")
    dataset_path = HF_LEROBOT_HOME / args.dataset_repo_id
    if dataset_path.exists():
        logging.info("Dataset found locally. Loading existing dataset to append data.")
        dataset = LeRobotDataset(repo_id=args.dataset_repo_id)
    else:
        logging.info("No local dataset found. Creating a new one.")
        dataset_features = {
            "observation.images.image": {"shape": (256, 256, 3), "dtype": "image"},
            "observation.images.wrist_image": {"shape": (256, 256, 3), "dtype": "image"},
            "observation.state": {"shape": (8,), "dtype": "float32"},
            "action": {"shape": (7,), "dtype": "float32"},
        }
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            features=dataset_features,
            fps=10,
            use_videos=True,
        )
    temp_img_dir = dataset.root / "temp_images"
    
    # --- 2. Robosuite Environment Setup ---
    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file), f"BDDL file not found: {args.bddl_file}"
    problem_info = bddl_utils.get_problem_info(args.bddl_file)

    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    
    logging.info(f"Task: {problem_name}")
    logging.info(f"Instruction: {language_instruction}")
    
    # Dynamically find the correct task class from libero.libero.envs
    task_class_name = None
    for key in suite.environments.REGISTERED_ENVS:
        if problem_name.lower() == key.lower():
            task_class_name = key
            break

    if not task_class_name:
        raise ValueError(f"Could not find a registered environment class for problem '{problem_name}'")

    TASK_MAPPING = suite.environments.REGISTERED_ENVS
    
    env = TASK_MAPPING[task_class_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[256, 256],
        camera_widths=[256, 256],
        reward_shaping=False,
        control_freq=10,
    )

    # env = VisualizationWrapper(env)

    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)

    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(
            vendor_id=args.vendor_id,
            product_id=args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # --- 4. Data Collection Loop ---
    collected_count = 0
    while collected_count < args.num_demonstrations:
        logging.info(f"Collecting demonstration {collected_count + 1} / {args.num_demonstrations}")
        
        was_saved = collect_human_trajectory(
            env, device, args.arm, args.config, dataset, language_instruction, temp_img_dir
        )
        if was_saved:
            collected_count += 1
            logging.info(f"Successfully collected and saved demonstration {collected_count}.")
        else:
            logging.warning("Current trajectory was aborted and not saved.")

    env.close()
    
    # --- 5. Push to Hub (Optional) ---
    if args.push_to_hub:
        if dataset.meta.total_episodes > 0:
            logging.info("Pushing dataset to the Hugging Face Hub...")
            dataset.push_to_hub()
            logging.info("Pushing complete.")
        else:
            logging.warning("No new episodes were collected. Skipping push to hub.")

    logging.info("--- Data Collection Finished ---")
    logging.info(f"Total episodes in dataset: {dataset.meta.total_episodes}")