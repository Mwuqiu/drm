import matplotlib.pyplot as plt
import numpy as np

class MetricsVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.suptitle('Online Fine-tuning Metrics', fontsize=16)        
        self.steps = []
        self.policy_losses = []
        self.avg_q_values = []
        self.axes[0].set_ylabel("Policy Loss")
        self.axes[0].grid(True)        
        self.axes[1].set_ylabel("Average Q-Value")
        self.axes[1].set_xlabel("Total Environment Steps")
        self.axes[1].grid(True)

    def update(self, step, logs: dict):
        # 1. 像以前一样收集数据（包含 NaN）
        self.steps.append(step)
        self.policy_losses.append(logs.get('policy_loss', np.nan))
        self.avg_q_values.append(logs.get('avg_q_value', np.nan))
        
        # --- 这是需要添加的修复代码 ---
        # 2. 在绘图前，创建只包含有效数据的“干净”列表
        
        # 清理 policy_loss 数据
        valid_loss_steps = [s for s, v in zip(self.steps, self.policy_losses) if not np.isnan(v)]
        valid_losses = [v for v in self.policy_losses if not np.isnan(v)]

        # 清理 avg_q_value 数据
        valid_q_steps = [s for s, v in zip(self.steps, self.avg_q_values) if not np.isnan(v)]
        valid_q_values = [v for v in self.avg_q_values if not np.isnan(v)]
        
        # 3. 清除旧的绘图并使用“干净”的数据重新绘制
        self.axes[0].clear()
        self.axes[0].set_ylabel("Policy Loss")
        self.axes[0].grid(True)
        self.axes[0].plot(valid_loss_steps, valid_losses, color='b')
        
        self.axes[1].clear()
        self.axes[1].set_ylabel("Average Q-Value")
        self.axes[1].set_xlabel("Total Environment Steps")
        self.axes[1].grid(True)
        self.axes[1].plot(valid_q_steps, valid_q_values, color='r', marker='.', linestyle='-')

        # -------------------------
        
        # 刷新画布
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.flush_events()
        plt.pause(0.01)


import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import collections
import random
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

from accelerate import Accelerator, DistributedDataParallelKwargs
from vqh_training_policy import VQHPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

def update_policy_only(policy: VQHPolicy, batch: dict, optimizers: dict, lr_schedulers: dict, accelerator: Accelerator) -> dict:
    policy.train()

    policy_loss, log_dict = policy.forward_policy_finetune(batch)    
    accelerator.backward(policy_loss)

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(policy.vqh.calql.policy.parameters(), 1.0)

    optimizers["actor"].step()
    optimizers["actor"].zero_grad()

    if lr_schedulers and "actor" in lr_schedulers:
        lr_schedulers["actor"].step()

    log_dict_updated = {"policy_loss": policy_loss.item()}
    if "actor_grad_norm" in log_dict:
        log_dict_updated["actor_grad_norm"] = log_dict["actor_grad_norm"]
    return log_dict_updated

import numpy as np
import torch
import logging

def log_evaluation_results(
    q_values,
    best_idx,
    best_sequence,
    is_calql,
    calql_samples_count,
    calql_q_values=None,
    base_q_values=None,
    best_calql_action=None,
    best_base_action=None,
    time_step=None,
    calql_action_sequences=None,
    base_action_sequences=None,
):
    q_values_np = q_values.cpu().numpy().flatten()

    log_msgs = ["\n"]
    log_msgs.append("=" * 70)

    log_msgs.append(f"- Time Step Now - : {time_step}")
    log_msgs.append("[Q-Value Evaluation]")
    if calql_q_values is not None and calql_q_values.numel() > 0:
        calql_q_str = ", ".join([f"{q:.4f}" for q in calql_q_values.cpu().numpy().flatten()])
        log_msgs.append(f"  - Cal-QL Q-Values:     [{calql_q_str}]")

    if base_q_values is not None and base_q_values.numel() > 0:
        base_q_str = ", ".join([f"{q:.4f}" for q in base_q_values.cpu().numpy().flatten()])
        log_msgs.append(f"  - Base Policy Q-Values:  [{base_q_str}]")

    log_msgs.append("  - Summary:")
    log_msgs.append(f"    - Winning Candidate Idx: {best_idx}")
    log_msgs.append(f"    - Winning Q-Value:       {q_values_np[best_idx]:.4f}")
    log_msgs.append(f"    - Average Q-Value:       {np.mean(q_values_np):.4f}")
    log_msgs.append("-" * 70)

    log_msgs.append("[All Candidate Actions]")
    # Log CAL-QL actions if they exist
    if calql_action_sequences is not None and calql_action_sequences.numel() > 0:
        log_msgs.append("  - Cal-QL Actions:")
        for i, sequence in enumerate(calql_action_sequences):
            q_value = q_values_np[i]
            log_msgs.append(f"    - Candidate {i} (Q={q_value:.4f}):")
            sequence_np = sequence.cpu().numpy()
            for step_idx, step_action in enumerate(sequence_np):
                
                if step_action.ndim == 0:
                    step_str = f"{step_action.item(): .4f}"
                else:
                    step_str = ", ".join([f"{x: .4f}" for x in step_action])
                log_msgs.append(f"      - Step {step_idx+1}: [{step_str}]")

    # Log Base Policy actions if they exist
    if base_action_sequences is not None and base_action_sequences.numel() > 0:
        log_msgs.append("  - Base Policy Actions:")
        start_idx = calql_samples_count
        for i, sequence in enumerate(base_action_sequences):
            q_value = q_values_np[start_idx + i]
            log_msgs.append(f"    - Candidate {start_idx + i} (Q={q_value:.4f}):")
            sequence_np = sequence.cpu().numpy()[:5]
            for step_idx, step_action in enumerate(sequence_np):
                
                if step_action.ndim == 0:
                    step_str = f"{step_action.item(): .4f}"
                else:
                    step_str = ", ".join([f"{x: .4f}" for x in step_action])
                log_msgs.append(f"      - Step {step_idx+1}: [{step_str}]")

    if (calql_action_sequences is None or calql_action_sequences.numel() == 0) and \
       (base_action_sequences is None or base_action_sequences.numel() == 0):
        log_msgs.append("  - No candidate sequences were provided for logging.")

    log_msgs.append("-" * 70)

    log_msgs.append("[Decision]")
    if is_calql:
        log_msgs.append(f"  - 🏆 VQH/Cal-QL 策略的动作 (候选者 {best_idx}) 被选中。")
    else:
        winner_idx_in_base = best_idx - calql_samples_count
        log_msgs.append(f"  - ✅ 基础策略的动作 (候选者 {winner_idx_in_base}) 被选中。")
    log_msgs.append("-" * 70)

    log_msgs.append("[Group Champions' Actions]")
    if best_calql_action is not None and calql_q_values is not None:
        best_calql_q_val = torch.max(calql_q_values).item()
        log_msgs.append(f"  - Best CAL-QL Action (Q={best_calql_q_val:.4f}):")
        best_calql_np = best_calql_action.cpu().numpy()
        for i, step_action in enumerate(best_calql_np):            
            if step_action.ndim == 0:
                step_str = f"{step_action.item(): .4f}"
            else:
                step_str = ", ".join([f"{x: .4f}" for x in step_action])
            log_msgs.append(f"    - Step {i+1}: [{step_str}]")
    else:
        log_msgs.append("  - No CAL-QL actions were sampled.")

    if best_base_action is not None and base_q_values is not None:
        best_base_q_val = torch.max(base_q_values).item()
        log_msgs.append(f"  - Best Base Action (Q={best_base_q_val:.4f}):")
        best_base_np = best_base_action.cpu().numpy()
        for i, step_action in enumerate(best_base_np):            
            if step_action.ndim == 0:
                step_str = f"{step_action.item(): .4f}"
            else:
                step_str = ", ".join([f"{x: .4f}" for x in step_action])
            log_msgs.append(f"    - Step {i+1}: [{step_str}]")

    log_msgs.append("-" * 70)

    log_msgs.append("[Overall Winning Action]")
    best_sequence_np = best_sequence.cpu().numpy()
    for i, step_action in enumerate(best_sequence_np):        
        if step_action.ndim == 0:
            step_str = f"{step_action.item(): .4f}"
        else:
            step_str = ", ".join([f"{x: .4f}" for x in step_action])
        log_msgs.append(f"  - Step {i+1}: [{step_str}]")

    log_msgs.append("=" * 70)
    logging.info("\n".join(log_msgs))

@dataclasses.dataclass
class Args:
    # policy_path: str = "outputs/train/2025-08-05/18-42-09_vqh/checkpoints/020000/pretrained_model"
    policy_path: str = "outputs/train/2025-08-12/09-27-20_vqh/checkpoints/010000/pretrained_model" 
    """Path to the pretrained VQHPolicy on the Hugging Face Hub or local directory."""
    task_suite_name: str = "libero_10"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    num_candidate_actions: int = 8
    """Number of candidate actions to sample at each step."""

    video_out_path: str = "data/libero/videos_vqh_debug"
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
    
    efficient_eval: bool = True
    debug_branching: bool = False

class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, observation):
        self.buffer.append(observation)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

@draccus.wrap()
def eval_libero(args: Args) -> None:

    visualizer = MetricsVisualizer()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.DEBUG)

    accelerator = Accelerator(log_with="wandb")
    logging.info(f"Loading VQHPolicy from {args.policy_path}")
    policy = VQHPolicy.from_pretrained(args.policy_path)
    policy.to(accelerator.device)

    optimizers, lr_schedulers = policy.prepare_optimizers()
    policy, actor_optimizer, critic_optimizer, temp_optimizer, vlm_optimizer = accelerator.prepare(
        policy, optimizers["actor"], optimizers["critic"], optimizers["temperature"], optimizers["vlm"]
    )

    optimizers = {
        "actor": actor_optimizer,
        "critic": critic_optimizer,
        "temperature": temp_optimizer,
        "vlm": vlm_optimizer,
    }

    replay_buffer = SimpleReplayBuffer(capacity=10000)
    finetune_batch_size = 32
    update_every_n_steps = 50
    evaluate_every_n_steps = 500
    start_finetuning_after = 500
    total_steps_so_far = 0

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = 150

    total_episodes = 100
    finish_episodes = 0

    while finish_episodes < total_episodes:
        task_id = 0
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        finish_episodes += 1

        for episode_idx in tqdm(range(min(args.num_trials_per_task, len(initial_states))), desc=f"Task {task_id}", leave=False):
            env.reset()
            policy.reset()

            episode_idx = 1
            task_episodes = 1

            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                
            t = 0
            done = False
            logging.info(f"Starting episode {task_episodes+1}...")

            realtime_vis_freq = 5
            while t < max_steps and not done:
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                state = np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))
                
                current_observation = {
                    "observation.images.image": torch.from_numpy(agentview_image / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                    "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0).permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                    "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                    "task": task_description,
                }
                
                replay_buffer.add(current_observation)
                policy.eval()
                with torch.inference_mode():
                    default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(current_observation)
                    gripper_actions = default_sequence[:, : args.foresee_steps, -1]
                    is_critical_action = torch.any(gripper_actions > 0) and torch.any(gripper_actions < 0)
                    if is_critical_action:
                        replay_buffer.add(current_observation)
                        best_action_sequence = policy.sample_calql_action(current_observation, num_samples = 1)                        
                        q_values = policy.evaluate_q_values(current_observation, best_action_sequence)
                        logging.info("Now Q-Values: %s", q_values)
                        best_action_sequence = best_action_sequence.squeeze()
                    else:
                        best_action_sequence = default_sequence[0, :args.n_action_steps]

                best_action_sequence_np = best_action_sequence.cpu().numpy()

                for action_in_sequence in best_action_sequence_np:
                    if t >= max_steps:
                        break
                    obs, _, done, _ = env.step(action_in_sequence)     

                    if t % realtime_vis_freq == 0:
                        current_agent_frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        current_wrist_frame = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        agent_view_live = current_agent_frame
                        wrist_view_live = current_wrist_frame
                        agent_view_bgr = cv2.cvtColor(agent_view_live, cv2.COLOR_RGB2BGR)
                        wrist_view_bgr = cv2.cvtColor(wrist_view_live, cv2.COLOR_RGB2BGR)
                        combined_view = np.concatenate((agent_view_bgr, wrist_view_bgr), axis=1)
                        scale_percent = 300 
                        original_width = combined_view.shape[1]
                        original_height = combined_view.shape[0]
                        new_width = int(original_width * scale_percent / 100)
                        new_height = int(original_height * scale_percent / 100)
                        new_dim = (new_width, new_height)
                        resized_view = cv2.resize(combined_view, new_dim, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow("Live Robot View (Agent | Wrist)", resized_view)
                        cv2.waitKey(1)

                    t += 1                    
                    total_steps_so_far += 1
                    if len(replay_buffer) > start_finetuning_after and total_steps_so_far % update_every_n_steps == 0:
                        logging.info(f"--- Performing policy-only finetuning at step {t} ---")
                        sampled_obs_list = replay_buffer.sample(finetune_batch_size)
                        
                        batch = {
                            "observation.images.image": torch.cat([s["observation.images.image"] for s in sampled_obs_list], dim=0).to(accelerator.device),
                            "observation.images.wrist_image": torch.cat([s["observation.images.wrist_image"] for s in sampled_obs_list], dim=0).to(accelerator.device),
                            "observation.state": torch.cat([s["observation.state"] for s in sampled_obs_list], dim=0).to(accelerator.device),
                            "task": [s["task"] for s in sampled_obs_list],
                            "action": torch.zeros(
                                (finetune_batch_size, policy.config.vqh_chunk_size, policy.config.action_feature.shape[0]),
                                device=accelerator.device
                            ),
                            "reward": torch.zeros(finetune_batch_size, device=accelerator.device),
                            "mc_returns": torch.zeros(finetune_batch_size, device=accelerator.device),
                            "tdm": torch.ones(finetune_batch_size, device=accelerator.device),
                        }
                        finetune_logs = update_policy_only(policy, batch, optimizers, lr_schedulers, accelerator)

                        if total_steps_so_far % evaluate_every_n_steps == 0:
                            with torch.no_grad():
                                policy.eval()
                                eval_obs_list = replay_buffer.sample(finetune_batch_size)
                                eval_batch = {
                                    "observation.images.image": torch.cat([s["observation.images.image"] for s in eval_obs_list], dim=0).to(accelerator.device),
                                    "observation.images.wrist_image": torch.cat([s["observation.images.wrist_image"] for s in eval_obs_list], dim=0).to(accelerator.device),
                                    "observation.state": torch.cat([s["observation.state"] for s in eval_obs_list], dim=0).to(accelerator.device),
                                    "task": [s["task"] for s in eval_obs_list],
                                }
                                actions_to_evaluate = policy.sample_calql_action(eval_batch, num_samples=1)                         
                                q_values = policy.evaluate_q_values(eval_batch, actions_to_evaluate)                            
                                avg_q_value = q_values.mean().item()
                                finetune_logs['avg_q_value'] = avg_q_value
                                print("avg_q_value : ", avg_q_value)

                        accelerator.log(finetune_logs, step=total_steps_so_far)
                        visualizer.update(total_steps_so_far, finetune_logs)
                    if done:
                        logging.info(f"Episode finished at step {t}. Breaking action sequence loop.")
                        break
                    
                if done:
                    task_successes += 1
                    total_successes += 1
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


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
    logging.basicConfig(level=logging.DEBUG)
    eval_libero()