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

from vqh_training_policy import VQHPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    policy_path: str = "outputs/train/2025-08-02/01-22-55_vqh/checkpoints/075000/pretrained_model"
    """Path to the pretrained VQHPolicy on the Hugging Face Hub or local directory."""
    
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""
    
    num_candidate_actions: int = 8
    """Number of candidate actions to sample at each step."""

    video_out_path: str = "data/libero/videos_vqh"

    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 8
    """Random Seed (for reproducibility)"""

    save_failure_dataset: bool = False
    """Whether to save trajectories from failed episodes to a dataset."""
    
    failure_dataset_repo_id: str = "libero_spatial_failures"
    """Repo ID for the dataset of failed trajectories."""

    foresee_steps: int = 20
    """Excute VQH Ranking Steps"""

    n_action_steps: int = 3
    """Normal Excute Action Steps"""

    g_action_steps: int = 3
    """Excute Gripper Action Steps"""

    efficient_eval: bool = True



@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Loading VQHPolicy from {args.policy_path}")
    policy = VQHPolicy.from_pretrained(args.policy_path)    
    policy.to(args.device)
    policy.eval()

    # Initialize dataset for failed trajectories if requested
    failure_dataset = None
    if args.save_failure_dataset:
        logging.info(f"Will save failure trajectories to dataset: {args.failure_dataset_repo_id}")
        failure_features = {
            "observation.images.image": {"shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), "dtype": "uint8"},
            "observation.images.wrist_image": {"shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), "dtype": "uint8"},
            "observation.state": {"shape": (7,), "dtype": "float32"},
            "action": {"shape": (7,), "dtype": "float32"},
        }
        failure_dataset = LeRobotDataset.create(
            repo_id=args.failure_dataset_repo_id,
            features=failure_features,
            fps=10,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. " f"Available options are: {list(benchmark_dict.keys())}")
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 300
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 520

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
            logging.info(f"\nTask: {task_description}")
            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            
            t = 0
            frames = []
            episode_data_buffer = [] 
            done = False
            
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(agentview_image)
            
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

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
                                logging.info("Gripper closing detected. Evaluating multiple candidates with VQH.")
                                candidate_sequences = [default_sequence]
                                for _ in range(args.num_candidate_actions - 1):
                                    seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                                    candidate_sequences.append(seq)

                                candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3) 
                                first_actions = candidate_sequences[:, :, 0, :]
                                q_values = policy.evaluate_q_values(observation, first_actions)
                                best_sequence_idx = torch.argmax(q_values, dim=1)
                                best_action_sequence = candidate_sequences[0, best_sequence_idx.item()]
                                best_action_sequence = best_action_sequence[:args.g_action_steps]
                            else:
                                best_action_sequence = default_sequence[0]
                                best_action_sequence = best_action_sequence[:args.n_action_steps]
                        else:
                            candidate_sequences = []
                            for _ in range(args.num_candidate_actions):
                                seq = policy.vqh.smolvla_policy.sample_action_sequence(observation)
                                candidate_sequences.append(seq)
                            
                            candidate_sequences = torch.stack(candidate_sequences).permute(1, 0, 2, 3) 
                            first_actions = candidate_sequences[:, :, 0, :]
                            q_values = policy.evaluate_q_values(observation, first_actions)
                            best_sequence_idx = torch.argmax(q_values, dim=1)
                            best_action_sequence = candidate_sequences[0, best_sequence_idx.item()]
                            best_action_sequence = best_action_sequence[:args.n_action_steps]

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
                        current_frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        frames.append(current_frame)                        
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

            if not done and args.save_failure_dataset:
                logging.info(f"Saving failed episode trajectory to dataset '{args.failure_dataset_repo_id}'.")
                for step_data in episode_data_buffer:
                    new_frame = {
                        "observation.images.image": step_data["agentview_image"],
                        "observation.images.wrist_image": step_data["wrist_image"],
                        "observation.state": step_data["state"],
                        "action": step_data["action"],
                    }
                    failure_dataset.add_frame(new_frame, task=task_description)
                failure_dataset.save_episode()

            # Save video and log results (logic remains the same)
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"

            fps = 30
            writer = imageio.get_writer(video_path, fps=fps)

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