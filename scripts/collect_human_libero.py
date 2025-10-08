import collections
import dataclasses
import logging
import math
import pathlib
import shutil

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  


@dataclasses.dataclass
class Args:
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-09-12/16-46-10_smolvla/checkpoints/100000/pretrained_model"
    policy_path: str = "/home/wuqiu/lerobot/outputs/train/libero_smolvla_object/checkpoints/100000/pretrained_model"
    task_suite_name: str = "libero_object_elegant"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/collect_human_episode"
    device: str = "cuda"
    seed: int = 7

    save_failure_dataset: bool = True
    failure_dataset_repo_id: str = "libero_elegant_0928/libero_object_task_3_append"

    human_in_loop: bool = True
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0


@draccus.wrap()
def eval_libero(args: Args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.n_action_steps = 10
    policy.to(args.device)
    policy.eval()

    keyboard_device = None
    if args.human_in_loop:
        logging.info("Human-in-the-loop mode enabled.")
        logging.info("Press 'H' to take over control from the policy.")
        logging.info("Press 'E' to end and SAVE the episode after taking over.")
        logging.info("Press 'Q' to ABANDON the episode and start a new one.")
        keyboard_device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)

    failure_dataset = None
    temp_img_dir = None
    if args.save_failure_dataset:
        logging.info(f"Handling dataset: {args.failure_dataset_repo_id}")
        dataset_path = HF_LEROBOT_HOME / args.failure_dataset_repo_id
        if dataset_path.exists():
            logging.info("Dataset found locally. Loading existing dataset to append data.")
            failure_dataset = LeRobotDataset(
                repo_id=args.failure_dataset_repo_id,
                download_videos=True,
            )
        else:
            logging.info("No local dataset found. Creating a new one.")
            failure_features = {
                "observation.images.image": {"shape": (256, 256, 3), "dtype": "image"},
                "observation.images.wrist_image": {"shape": (256, 256, 3), "dtype": "image"},
                "observation.state": {"shape": (8,), "dtype": "float32"},
                "action": {"shape": (7,), "dtype": "float32"},
            }
            failure_dataset = LeRobotDataset.create(
                repo_id=args.failure_dataset_repo_id,
                features=failure_features,
                fps=10,
                use_videos=True,
            )
        temp_img_dir = failure_dataset.root / "temp_images"

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
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 5000
        
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):

        if task_id != 3:
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
            logging.info(f"\nTask: {task_description}")
            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            t = 0
            frames = []
            done = False
            episode_buffer = []

            human_control_active = False
            episode_abandoned = False
            prev_state = None
            gripper_target_state = 1.0  
            prev_key_down = 0

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

                    if args.human_in_loop:
                        agent_view_bgr = cv2.cvtColor(agentview_image, cv2.COLOR_RGB2BGR)
                        wrist_view_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                        combined_view = np.concatenate((agent_view_bgr, wrist_view_bgr), axis=1)
                        
                        scale_percent = 200 
                        new_width = int(combined_view.shape[1] * scale_percent / 100)
                        new_height = int(combined_view.shape[0] * scale_percent / 100)
                        
                        resized_view = cv2.resize(combined_view, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("Live Robot View (Agent | Wrist)", resized_view)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('h'):
                            human_control_active = not human_control_active
                            if human_control_active:
                                logging.info("--> Human control ACTIVATED. Press 'h' again to return to policy.")

                                current_gripper_qpos = obs["robot0_gripper_qpos"][1]
                                if current_gripper_qpos < 0.5:
                                    gripper_target_state = 1.0
                                else:
                                    gripper_target_state = -1.0
                                logging.info(f"Gripper state synced. Target is now: {'CLOSED' if gripper_target_state > 0 else 'OPEN'}")

                            else:
                                logging.info("--> Policy control RESUMED.")
                                policy.reset()
                        
                        if human_control_active:
                            if key == ord('e'): # End and Save episode
                                logging.info("Episode ended by human ('e' pressed), saving trajectory.")
                                break
                            if key == ord('q'): # Abandon episode
                                logging.info("Abandoning current episode by user command ('q' pressed).")
                                episode_abandoned = True
                                break

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    if human_control_active:
                        active_robot = env.robots[0]

                        action, _ = input2action(
                            device=keyboard_device, robot=active_robot, active_arm='right'
                        )
                        if action is None:
                            action = np.zeros(7)

                        device_state = keyboard_device.get_controller_state()
                        key_down = device_state['grasp']

                        is_press_event = (key_down == 1) and (prev_key_down == 0)

                        if is_press_event:
                            gripper_target_state *= -1.0  

                        action[-1] = gripper_target_state
                        prev_key_down = key_down

                    else:
                        observation = {
                            "observation.images.image": torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                            "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                            "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                            "task": task_description,
                        }
                        with torch.inference_mode():
                            action_tensor = policy.select_action(observation)
                        action = action_tensor.cpu().numpy()[0]
                    
                    is_human_zero_action = human_control_active and (np.sum(np.abs(action[:-1])) < 1e-4)

                    if args.save_failure_dataset and not is_human_zero_action:
                        step_data = {
                            "observation.state": state.astype(np.float32),
                            "action": action.astype(np.float32),
                        }
                        next_episode_index = failure_dataset.meta.total_episodes
                        
                        for key, img_array in [("observation.images.image", agentview_image), ("observation.images.wrist_image", wrist_img)]:
                            episode_img_dir = temp_img_dir / key / f"episode_{next_episode_index:06d}"
                            episode_img_dir.mkdir(parents=True, exist_ok=True)
                            img_path = episode_img_dir / f"frame_{t:06d}.png"
                            imageio.imwrite(img_path, img_array)
                            step_data[key] = str(img_path)
                            
                        episode_buffer.append(step_data)

                    obs, _, done, _ = env.step(action)
                    prev_state = state

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break
            
            if episode_abandoned:
                logging.info(f"Skipping save for abandoned episode.")
                continue

            task_episodes += 1
            total_episodes += 1
            
            if not episode_abandoned and args.save_failure_dataset and episode_buffer and done:
                logging.info(f"Saving failed episode to dataset {args.failure_dataset_repo_id}")
                episode_data_dict = collections.defaultdict(list)
                for step in episode_buffer:
                    for key, value in step.items():
                        episode_data_dict[key].append(value)
                episode_length = len(episode_buffer)
                episode_data_dict["frame_index"] = list(range(episode_length))
                fps = failure_dataset.fps
                episode_data_dict["timestamp"] = [i / fps for i in range(episode_length)]
                start_index = failure_dataset.meta.total_frames
                episode_data_dict["index"] = list(range(start_index, start_index + episode_length))
                task_index = failure_dataset.meta.get_task_index(task_description)
                if task_index is None:
                    failure_dataset.meta.add_task(task_description)
                    task_index = failure_dataset.meta.get_task_index(task_description)
                episode_data_dict["task_index"] = [task_index] * episode_length
                
                episode_index_to_save = failure_dataset.meta.total_episodes
                episode_data_dict["episode_index"] = episode_index_to_save
                episode_data_dict["size"] = episode_length
                episode_data_dict["task"] = [task_description] * episode_length
                
                failure_dataset.save_episode(episode_data=episode_data_dict)

                for key in ["observation.images.image", "observation.images.wrist_image"]:
                    saved_episode_img_dir = temp_img_dir / key / f"episode_{episode_index_to_save:06d}"
                    if saved_episode_img_dir.is_dir():
                        shutil.rmtree(saved_episode_img_dir)

            # Save video for all non-abandoned episodes
            if not episode_abandoned:
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
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    cv2.destroyAllWindows()


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
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def is_noop(state, prev_state=None, pos_threshold=1e-4, gripper_threshold=1e-4):
    if prev_state is None:
        return False 
    eef_change = np.linalg.norm(state[:6] - prev_state[:6])
    gripper_change = np.linalg.norm(state[6:] - prev_state[6:])
    return eef_change < pos_threshold and gripper_change < gripper_threshold

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero()