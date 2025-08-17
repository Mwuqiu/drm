import collections
import dataclasses
import logging
import math
import os
import pathlib
import queue
import random
import threading
import time

import cv2
import draccus
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from torch.optim import Adam
from tqdm import tqdm
from termcolor import colored
from collections import deque

from accelerate import Accelerator
from vqh_training_policy import VQHPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # policy_path: str = "outputs/train/2025-08-12/09-27-20_vqh/checkpoints/010000/pretrained_model" 
    # policy_path: str = "outputs/train/2025-08-05/18-42-09_vqh/checkpoints/010000/pretrained_model"
    # policy_path: str = "outputs/train/2025-08-14/18-50-08_vqh/checkpoints/007500/pretrained_model"
    policy_path : str = "outputs/train/2025-08-15/10-01-17_vqh/checkpoints/045000/pretrained_model"
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 100
    num_candidate_actions: int = 8
    video_out_path: str = "data/libero/videos_vqh_debug"
    device: str = "cuda"
    seed: int = 7
    foresee_steps: int = 10
    n_action_steps: int = 3

    replay_buffer_capacity: int = 1000
    finetune_batch_size: int = 64
    start_finetuning_after: int = 10
    update_every_n_updates: int = 1  # Learner 每收到 N 次数据就更新
    evaluate_every_n_updates: int = 20 # Learner 每 N 次更新后评估一次
    param_push_freq: int = 10 # Learner 每 N 次更新后发送一次模型参数
    actor_send_freq: int = 5   # Actor 每 N 步发送一次数据


class ThreadSafeReplayBuffer:
    """线程安全的 Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()

    def add(self, observation):
        with self.lock:
            self.buffer.append(observation)

    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            return random.sample(self.buffer, batch_size)

    def __len__(self):
        with self.lock:
            return len(self.buffer)

class MetricsVisualizer:
    # ... (与之前完全相同的 MetricsVisualizer 类代码)
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
        self.axes[1].set_xlabel("Total Learner Updates")
        self.axes[1].grid(True)

    def update(self, step, logs: dict):
        self.steps.append(step)
        self.policy_losses.append(logs.get('policy_loss', np.nan))
        self.avg_q_values.append(logs.get('avg_q_value', np.nan))
        valid_loss_steps = [s for s, v in zip(self.steps, self.policy_losses) if not np.isnan(v)]
        valid_losses = [v for v in self.policy_losses if not np.isnan(v)]
        valid_q_steps = [s for s, v in zip(self.steps, self.avg_q_values) if not np.isnan(v)]
        valid_q_values = [v for v in self.avg_q_values if not np.isnan(v)]
        self.axes[0].clear(); self.axes[0].set_ylabel("Policy Loss"); self.axes[0].grid(True)
        self.axes[0].plot(valid_loss_steps, valid_losses, color='b')
        self.axes[1].clear(); self.axes[1].set_ylabel("Average Q-Value"); self.axes[1].set_xlabel("Total Learner Updates"); self.axes[1].grid(True)
        self.axes[1].plot(valid_q_steps, valid_q_values, color='r', marker='.', linestyle='-')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); self.fig.canvas.flush_events(); plt.pause(0.01)

def update_policy_only(policy: VQHPolicy, batch: dict, optimizers: dict, lr_schedulers: dict, accelerator: Accelerator) -> dict:
    # ... (与之前完全相同的 update_policy_only 函数代码)
    policy.train()
    policy_loss, log_dict = policy.forward_policy_finetune(batch)    

    print(f"policy_loss: {policy_loss}")
    print(f"policy_loss requires_grad: {policy_loss.requires_grad}")
    print(f"policy_loss grad_fn: {policy_loss.grad_fn}")

    accelerator.backward(policy_loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(policy.vqh.calql.policy.parameters(), 1.0)
    optimizers["actor"].step()
    optimizers["actor"].zero_grad()
    if lr_schedulers and "actor" in lr_schedulers:
        lr_schedulers["actor"].step()
    log_dict.update({"policy_loss": policy_loss.item()})
    return log_dict
    
def _get_libero_env(task, resolution, seed):
    # ... (您原来的 _get_libero_env 函数)
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    # ... (您原来的 _quat2axisangle 函数)
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def learner_fn(
    policy: VQHPolicy,
    optimizer: dict,
    lr_schedulers: dict,
    replay_buffer: ThreadSafeReplayBuffer,
    transition_queue: queue.Queue,
    parameter_queue: queue.Queue,
    stop_event: threading.Event,
    accelerator: Accelerator,
    log_queue: queue.Queue,
    args: Args,
):
    """负责从队列获取数据，更新模型，评估并发送新参数"""
    total_updates = 0
    data_packages_received = 0
    while not stop_event.is_set():
        try:
            # 从Actor获取一批 transitions
            transitions = transition_queue.get(timeout=1.0)
            data_packages_received += 1
            for transition in transitions:
                replay_buffer.add(transition)
        except queue.Empty:
            continue
            
        logging.info(
            f"Learner Heartbeat: "
            f"Buffer size: {len(replay_buffer)}/{args.start_finetuning_after}, "
            f"Updates done: {total_updates}"
        )

        if len(replay_buffer) < args.start_finetuning_after:
            time.sleep(1)
            continue
        
        # 每收到 N 次数据包后，进行一次更新
        if total_updates % args.update_every_n_updates == 0:
            # --- 模型更新 ---
            sampled_obs_list = replay_buffer.sample(args.finetune_batch_size)
            if not sampled_obs_list: continue
            
            batch = { # ... 构造训练 batch ...
                "observation.images.image": torch.cat([s["observation.images.image"] for s in sampled_obs_list], dim=0),
                "observation.images.wrist_image": torch.cat([s["observation.images.wrist_image"] for s in sampled_obs_list], dim=0),
                "observation.state": torch.cat([s["observation.state"] for s in sampled_obs_list], dim=0),
                "task": [s["task"] for s in sampled_obs_list],
                "action": torch.zeros((args.finetune_batch_size, policy.config.vqh_chunk_size, policy.config.action_feature.shape[0]), device=accelerator.device),
                "reward": torch.zeros(args.finetune_batch_size, device=accelerator.device),
            }
            finetune_logs = update_policy_only(policy, batch, optimizer, lr_schedulers, accelerator)

            # --- 模型评估与可视化 ---
            if total_updates % args.evaluate_every_n_updates == 0 and False:
                with torch.no_grad():
                    policy.eval()
                    eval_batch = { # 构造评估 batch
                        "observation.images.image": torch.cat([s["observation.images.image"] for s in sampled_obs_list], dim=0),
                        "observation.images.wrist_image": torch.cat([s["observation.images.wrist_image"] for s in sampled_obs_list], dim=0),
                        "observation.state": torch.cat([s["observation.state"] for s in sampled_obs_list], dim=0),
                        "task": [s["task"] for s in sampled_obs_list],
                    }
                    actions_to_evaluate = policy.sample_calql_action(eval_batch, num_samples=1)
                    q_values = policy.evaluate_q_values(eval_batch, actions_to_evaluate)
                    avg_q_value = q_values.mean().item()
                    finetune_logs['avg_q_value'] = avg_q_value
                
            accelerator.log(finetune_logs, step=total_updates)
            log_queue.put((total_updates, finetune_logs))
            # visualizer.update(total_updates, finetune_logs)

            # --- 发送新参数 ---
            if total_updates % args.param_push_freq == 0:
                cpu_state_dict = {k: v.cpu() for k, v in policy.vqh.calql.policy.state_dict().items()}
                # cpu_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
                parameter_queue.put(cpu_state_dict)
                logging.info(colored(f"Learner pushed new parameters after {total_updates} updates.", "green", attrs=["bold"]))

        total_updates += 1

def actor_fn(
    policy: VQHPolicy,
    transition_queue: queue.Queue,
    parameter_queue: queue.Queue,
    stop_event: threading.Event,
    accelerator: Accelerator,
    args: Args,
    high_q_frame_dir: str, 
):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = 100
    
    task_id = 0
    episode_idx = 1

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    realtime_vis_freq = 1
    episode_count = 0
    total_episode_count = 1000

    total_q_value_successes = 0

    packages_sent = 0 

    recent_successes = deque(maxlen=10)
   

    while not stop_event.is_set() and episode_count < total_episode_count:
        env.reset()
        policy.reset()

        obs = env.set_init_state(initial_states[episode_idx])
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        
        t = 0        
        done = False

        logging.info(f"Actor starting episode #{episode_count + 1}...")
        
        transitions_to_send = []
        episode_q_values = []
        

        while t < max_steps and not done:
            try:
                new_params = parameter_queue.get_nowait()
                # policy.load_state_dict(new_params)
                policy.vqh.calql.policy.load_state_dict(new_params)
                logging.info(colored("Actor updated with new parameters from learner.", "cyan", attrs=["bold"]))
            except queue.Empty:
                pass

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
            
            # transitions_to_send.append(current_observation)
            policy.eval()
            with torch.inference_mode():
                default_sequence = policy.vqh.smolvla_policy.sample_action_sequence(current_observation)
                gripper_actions = default_sequence[:, : args.foresee_steps, -1]
                is_critical_action = torch.any(gripper_actions > 0) and torch.any(gripper_actions < 0)
                if is_critical_action:
                    transitions_to_send.append(current_observation)
                    best_action_sequence = policy.sample_calql_action(current_observation, num_samples = 1)                        
                    q_values = policy.evaluate_q_values(current_observation, best_action_sequence)
                    q_values = q_values.cpu().numpy()[0]
                    q_value_scalar = q_values.mean().item()

                    logging.info("Now Q-Values: %s", q_values)
                    episode_q_values.append(q_values)

                    # if q_value_scalar > 0.5:
                    #     filename = f"step_{t}_q_{q_value_scalar:.4f}.png"
                    #     save_path = os.path.join(high_q_frame_dir, filename)
                    #     image_to_save = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                    #     cv2.imwrite(save_path, image_to_save)
                    #     logging.info(colored(f"✅ High Q-value frame saved to {save_path}", "yellow"))

                    best_action_sequence = best_action_sequence.squeeze()
                    # best_action_sequence = best_action_sequence.unsqueeze(0)
                else:
                    best_action_sequence = default_sequence[0, :args.n_action_steps]
            
            best_action_sequence_np = best_action_sequence.cpu().numpy()
            # print("shape of best_action_sequence_np : ", best_action_sequence_np.shape)

            for action_in_sequence in best_action_sequence_np:
                if t >= max_steps: break
                obs, _, done, info = env.step(action_in_sequence)
                t += 1
                
                if t % realtime_vis_freq == 0:
                    current_agent_frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    current_wrist_frame = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agent_view_live = current_agent_frame
                    wrist_view_live = current_wrist_frame
                    agent_view_bgr = cv2.cvtColor(agent_view_live, cv2.COLOR_RGB2BGR)
                    wrist_view_bgr = cv2.cvtColor(wrist_view_live, cv2.COLOR_RGB2BGR)
                    combined_view = np.concatenate((agent_view_bgr, wrist_view_bgr), axis=1)
                    scale_percent = 200 
                    original_width = combined_view.shape[1]
                    original_height = combined_view.shape[0]
                    new_width = int(original_width * scale_percent / 100)
                    new_height = int(original_height * scale_percent / 100)
                    new_dim = (new_width, new_height)
                    resized_view = cv2.resize(combined_view, new_dim, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Live Robot View (Agent | Wrist)", resized_view)
                    cv2.waitKey(1)

                if done or info.get('timed_out', False):
                    break
            
            if len(transitions_to_send) >= args.actor_send_freq:
                transition_queue.put(transitions_to_send)
                packages_sent += 1
                logging.info(f"Actor sent data package #{packages_sent} to learner.")
                transitions_to_send = []


        episode_count += 1

        is_q_value_success = False
        avg_episode_q = 0.0
        if episode_q_values:
            avg_episode_q = np.mean(episode_q_values)
            if avg_episode_q > 0:
                is_q_value_success = True
                total_q_value_successes += 1

        recent_successes.append(1 if is_q_value_success else 0)
        num_recent_episodes = len(recent_successes)
        if num_recent_episodes > 0:
            rolling_success_rate = sum(recent_successes) / num_recent_episodes
        else:
            rolling_success_rate = 0.0

        logging.info(f"--- Episode #{episode_idx + 1} Summary ---")
        logging.info(f"Q-Value Based Success (Avg Q > 0): {is_q_value_success} (Avg Q: {avg_episode_q:.4f})")
        # logging.info(colored(f"Rolling 10-Episode Success Rate: {rolling_success_rate:.0%} ({sum(recent_successes)}/{num_recent_episodes})", "yellow", attrs=["bold"]))
        logging.info("-" * 40)
    
    if episode_count > 0:
        q_value_success_rate = total_q_value_successes / episode_count
        logging.info(f"=== Final Q-Value Success Rate: {q_value_success_rate:.2%} ({total_q_value_successes}/{episode_count}) ===")

    stop_event.set()
    logging.info("Actor has finished all its episodes.")

@draccus.wrap()
def main(args: Args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    high_q_frame_dir = "high_q_frames"
    os.makedirs(high_q_frame_dir, exist_ok=True)
    logging.info(f"High Q-value frames will be saved to: {high_q_frame_dir}")
    
    
    accelerator = Accelerator(log_with="wandb")
    
    logging.info(f"Loading VQHPolicy from {args.policy_path}")
    policy = VQHPolicy.from_pretrained(args.policy_path)

    new_actor_lr = 1e-2
    logging.info(f"Overriding Actor LR. Original: {policy.config.actor_lr}, New: {new_actor_lr}")
    policy.config.actor_lr = new_actor_lr
    
    optimizers, lr_schedulers = policy.prepare_optimizers()
    
    policy, optimizers["actor"], optimizers["critic"], optimizers["temperature"], optimizers["vlm"] = accelerator.prepare(
        policy, optimizers["actor"], optimizers["critic"], optimizers["temperature"], optimizers["vlm"]
    )
    
    replay_buffer = ThreadSafeReplayBuffer(args.replay_buffer_capacity)
    visualizer = MetricsVisualizer()
    
    transition_queue = queue.Queue(maxsize=10)
    parameter_queue = queue.Queue(maxsize=1)
    log_queue = queue.Queue()
    stop_event = threading.Event()
    
    learner_thread = threading.Thread(
        target=learner_fn,
        name="LearnerThread",
        args=(policy, optimizers, lr_schedulers, replay_buffer, transition_queue, parameter_queue, stop_event, accelerator, log_queue, args), # 移除 visualizer, 加入 log_queue
    )
    
    actor_thread = threading.Thread(
        target=actor_fn,
        name="ActorThread",
        args=(policy, transition_queue, parameter_queue, stop_event, accelerator, args, high_q_frame_dir),
    )
    
    logging.info("Starting Actor and Learner threads...")
    learner_thread.start()
    actor_thread.start()


    try:
        while learner_thread.is_alive():
            try:
                step, logs = log_queue.get(timeout=1.0)
                visualizer.update(step, logs)
                
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected. Shutting down threads gracefully...")

    try:
        actor_thread.join()
        learner_thread.join()
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected. Shutting down threads gracefully...")
        stop_event.set()
    finally:
        if actor_thread.is_alive(): actor_thread.join()
        if learner_thread.is_alive(): learner_thread.join()
        logging.info("All threads have been joined. Program finished.")
        
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()