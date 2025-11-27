# eval_real_world.py (Final Version)

import logging
import time
import shutil
import pathlib
from dataclasses import dataclass, field
from pprint import pformat

import cv2
import draccus
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.common.policies.factory import make_policy
from lerobot.common.robots import make_robot_from_config, so100_follower
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@dataclass
class Args:
    # "pick up the carrot and put it in the basket"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-11-02/17-02-42_smolvla/checkpoints/010000/pretrained_model"

    # "stack the red squares on top of the green ones"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-11-03/15-09-33_smolvla/checkpoints/010000/pretrained_model"

    # "stack the blue squares on top of the yellow ones"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-11-03/21-52-24_smolvla/checkpoints/007500/pretrained_model"

    # "Put the bowl on the plate"
    # policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-11-04/16-41-10_smolvla/checkpoints/010000/pretrained_model"

    # "Put the corn on the plate"
    policy_path: str = "/home/wuqiu/lerobot/outputs/train/2025-11-04/21-43-29_smolvla/checkpoints/005000/pretrained_model"
    robot_type: str = "so100_follower"
    robot_port: str = "/dev/follower"
    robot_id: str = "follower_arm"
    camera_config: dict = field(default_factory=lambda: {
        "wrist_image": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=30),
        "image": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30)
    })
    task_description: str = "Place the corn on the plate"
    num_episodes: int = 50
    episode_duration_s: float = 50.0
    fps: int = 30
    reset_duration_s: float = 20.0
    video_out_path: str = "data/eval_real_world_cop_5k"
    device: str = "cuda"
    seed: int = 42


@draccus.wrap()
def eval_real_robot(args: Args):
    init_logging()
    logging.info(pformat(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_safe_torch_device(args.device)
    listener, events = init_keyboard_listener()
    logging.info("键盘监听器已启动。按 '→' 可提前结束当前操作，按 'Esc' 停止所有评估。")
    logging.info("正在初始化机器人以获取硬件元数据...")
    robot_config = so100_follower.SO100FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=args.camera_config)
    robot = make_robot_from_config(robot_config)
    logging.info("机器人对象创建成功。")
    logging.info(f"正在从 '{args.policy_path}' 加载策略...")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    logging.info("策略加载成功。")
    robot.connect()
    logging.info("机器人硬件连接成功。")
    video_dir = pathlib.Path(args.video_out_path)

    if not video_dir.exists():
        video_dir.mkdir(parents=True)

    # if video_dir.exists():
        # shutil.rmtree(video_dir)
    # video_dir.mkdir(parents=True)
    
    for episode_idx in range(args.num_episodes):
        log_title = f"第 {episode_idx + 1}/{args.num_episodes} 次评估"
        logging.info(f"--- {log_title} ---")
        
        policy.reset()

        frames = []
        start_time = time.perf_counter()
        
        pbar = tqdm(total=int(args.episode_duration_s), desc=f"评估中 E{episode_idx+1}")
        last_update_time = start_time
        while (time.perf_counter() - start_time) < args.episode_duration_s:
            if events["exit_early"]:
                break
            
            loop_start_time = time.perf_counter()
            observation_from_robot = robot.get_observation()
            
            main_cam_image = observation_from_robot.get("image")
            if main_cam_image is not None:
                frames.append(main_cam_image)

            observation_for_policy = {}

            for key in ["image", "wrist_image"]:
                if key in observation_from_robot:
                    value = observation_from_robot[key]
                    tensor = torch.from_numpy(value / 255.0).permute(2, 0, 1).to(torch.float32)
                    observation_for_policy[f"observation.images.{key}"] = tensor.to(device).unsqueeze(0)

            camera_keys = list(args.camera_config.keys())
            state_feature_keys = [k for k in robot.observation_features if k not in camera_keys]            
            
            state_values = [observation_from_robot[key] for key in state_feature_keys]
            
            state_numpy = np.array(state_values, dtype=np.float32)
            state_tensor = torch.from_numpy(state_numpy).to(torch.float32)

            observation_for_policy["observation.state"] = state_tensor.to(device).unsqueeze(0)
            observation_for_policy["task"] = args.task_description
            
            with torch.inference_mode():
                action_tensor = policy.select_action(observation_for_policy)
            
            action_numpy = action_tensor.cpu().numpy()[0]
            action_dict = {key: action_numpy[i] for i, key in enumerate(robot.action_features)}
            robot.send_action(action_dict)
            loop_duration = time.perf_counter() - loop_start_time
            busy_wait(1.0 / args.fps - loop_duration)
            current_time = time.perf_counter()
            pbar.update(int(current_time - last_update_time))
            last_update_time = current_time

        pbar.close()

        if events["exit_early"]:
            logging.info("检测到键盘输入，提前终止本轮评估...")
            events["exit_early"] = False
        else:
            logging.info("本轮评估时间到。")

        video_path = video_dir / f"rollout_episode_{episode_idx}.mp4"
        imageio.mimsave(video_path, frames, fps=args.fps)
        logging.info(f"视频已保存到: {video_path}")

        success_input = ""
        while success_input not in ['s', 'f']:
            success_input = input(f"\n>>> Episode {episode_idx + 1} finished. Was it a success? (s/f): ").lower().strip()
            
        if success_input == 's':
            try:
                new_video_path = video_path.with_stem(f"{video_path.stem}_success")
                video_path.rename(new_video_path)
                logging.info(f"Marked as SUCCESS. Renamed video to: {new_video_path}")
            except Exception as e:
                logging.error(f"Could not rename video file: {e}")
        else:
            logging.info("Marked as FAILURE. Video name unchanged.")


        if events["stop_recording"]:
            logging.info("检测到 'Esc' 停止信号，将终止所有评估...")
            break

        if episode_idx < args.num_episodes - 1:
            logging.info(f"请在 {args.reset_duration_s} 秒内重置环境。按 '→' 可跳过等待。")
            reset_start_time = time.perf_counter()
            
            pbar_reset = tqdm(total=int(args.reset_duration_s), desc="重置倒计时")
            last_reset_update_time = reset_start_time

            while (time.perf_counter() - reset_start_time) < args.reset_duration_s:
                if events["exit_early"]:
                    pbar_reset.n = pbar_reset.total
                    pbar_reset.refresh()
                    break
                
                current_reset_time = time.perf_counter()
                pbar_reset.update(int(current_reset_time - last_reset_update_time))
                last_reset_update_time = current_reset_time
                time.sleep(0.1)

            pbar_reset.close()
            if events["exit_early"]:
                logging.info("跳过重置等待。")
                events["exit_early"] = False
            else:
                logging.info("重置时间到。")            
            input(">>> 重置完成后，请按 Enter 键开始下一次评估 <<<")

    logging.info("--- 所有评估已完成 ---")
    robot.disconnect()
    if not is_headless() and listener is not None:
        listener.stop()
    logging.info("机器人和键盘监听器已关闭。")

if __name__ == "__main__":
    eval_real_robot()