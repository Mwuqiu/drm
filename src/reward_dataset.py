import json
import logging
import shutil
import torch
from pathlib import Path
from typing import Callable
import datasets
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from lerobot.common.datasets.utils import hf_transform_to_torch, DEFAULT_FEATURES
from PIL import Image


# from vlm_labeler import label_episode

class RewardLeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        discount: float = 0.99,
        num_pos: int = 3,
        next_obs_offset: int = 10,
        vel_threshold: float = 0.015,
        reward_weights: dict = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            episodes=episodes,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        self.discount = discount
        self.num_pos = num_pos
        self.next_obs_offset = next_obs_offset

        # print("Constructing reward data for the entire dataset...")
        # self.reward_dataset = self.construct_reward_data()
        # print("Reward data constructed.")

        # ReinboT Reward Param
        self.vel_threshold = vel_threshold
        self.reward_weights = reward_weights
        if self.reward_weights is None:
            self.reward_weights = {"w1": 1, "w2": 0, "w3": 0, "w4": 0}

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train", keep_in_memory=True)
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train", keep_in_memory=True)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def construct_reward_data(self) -> datasets.Dataset:
        reward_keys = ["reward", "mc_returns", "tdm", "done"]
        reward_data = {k: [] for k in reward_keys}
        for i in tqdm(range(self.num_episodes), desc="Calculating rewards"):
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start
            reward = torch.zeros(ep_len, dtype=torch.float32)
            if ep_len > self.num_pos:
                reward[: -self.num_pos] = -1.0
            td_mask = torch.ones(ep_len, dtype=torch.bool)
            td_mask[-self.num_pos :] = False
            mc_ret = torch.zeros_like(reward)
            prev_ret = 0.0
            for t in range(ep_len - 1, -1, -1):
                mc_ret[t] = reward[t] + self.discount * prev_ret
                prev_ret = mc_ret[t]
            done = torch.zeros(ep_len, dtype=torch.bool)
            done[-1] = True
            for k, v in zip(reward_keys, [reward, mc_ret, td_mask, done]):
                reward_data[k].append(v)
        for k, v in reward_data.items():
            reward_data[k] = torch.cat(v, dim=0)
        reward_hf_dataset = datasets.Dataset.from_dict(reward_data, split="train")
        reward_hf_dataset.set_transform(hf_transform_to_torch)
        return reward_hf_dataset

    def _calculate_orb_similarity(self, img1, img2) -> float:
        """基于 ORB 特征匹配计算相似度"""
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if not matches:
            return 0.0

        num_keypoints = min(len(kp1), len(kp2))
        return len(matches) / num_keypoints if num_keypoints > 0 else 0.0

    def _calculate_ssim_similarity(self, img1, img2) -> float:
        """Calculates structural similarity index (SSIM) with an adaptive window size."""
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        h, w, _ = img1_np.shape

        # The ssim function's default window size is 7. If the image's smaller side
        # is less than 7, we must provide a smaller, odd-numbered window size.
        smaller_side = min(h, w)

        # If the image is too small for a meaningful comparison, return 0 similarity.
        # The smallest possible odd window size is 3.
        if smaller_side < 3:
            return 0.0

        # Determine a safe window size: it must be odd and smaller than the image dimensions.
        win_size = min(7, smaller_side)
        if win_size % 2 == 0:
            win_size -= 1

        return ssim(img1_np, img2_np, channel_axis=-1, data_range=255, win_size=win_size)

    def _find_subgoals(self, states: torch.Tensor, actions: torch.Tensor) -> list[int]:
        """
        Finds sub-goals by detecting a change in gripper state that is followed
        by a period of stability, thus correctly identifying the end of a
        noisy transition.
        """
        gripper_stability_period = 25

        ep_len = len(states)
        subgoal_indices = []

        gripper_actions = actions[:, -1]

        t = 1
        while t < ep_len:
            gripper_state_changed = gripper_actions[t].sign() != gripper_actions[t - 1].sign()

            if gripper_state_changed:
                is_stable = True
                new_gripper_state_sign = gripper_actions[t].sign()
                end_of_stability_window = min(t + gripper_stability_period, ep_len - 1)

                for k in range(t + 1, end_of_stability_window + 1):
                    if gripper_actions[k].sign() != new_gripper_state_sign:
                        is_stable = False
                        break

                if is_stable:
                    subgoal_indices.append(t)
                    t += gripper_stability_period
                else:
                    t += 1
            else:
                t += 1
        return subgoal_indices

    def construct_dense_reward_data(self) -> datasets.Dataset:  
        stage_bonus_value = 2.5 
        proprio_mse_scale = 50.0
        img_mse_scale = 10.0

        r3_grace_period = 5
        reward_keys = ["reward", "r1", "r2", "r3", "r4", "vel_norm", "gripper_action", "mc_returns", "tdm", "done"]
        reward_data = {k: [] for k in reward_keys}
        w = self.reward_weights

        IMPRECISE_PLACEMENT_THRESHOLD_FRAMES = 10

        for i in tqdm(range(self.num_episodes), desc="Cal Dense Reward"):
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start

            if ep_len <= 2:
                continue

            episode_data = self.hf_dataset[ep_start:ep_end]
            ep_states = torch.stack(episode_data["observation.state"])
            ep_actions = torch.stack(episode_data["action"])
            ep_image_features = torch.stack([torch.tensor(img) for img in episode_data["observation.images.image"]])

            ep_gripper_action = ep_actions[:, -1]

            subgoal_indices = self._find_subgoals(ep_states, ep_actions)
            

            subgoal_rewards = torch.zeros(ep_len, dtype=torch.float32)
            
            # --- START: 应用你的新发现，过滤不精确的最终放置 ---
            if subgoal_indices:
                last_subgoal_frame = subgoal_indices[-1]
                frames_after_last_subgoal = (ep_len - 1) - last_subgoal_frame
                if frames_after_last_subgoal > IMPRECISE_PLACEMENT_THRESHOLD_FRAMES:
                    print(f"\n[Heuristic Applied] Episode {i}: Final subgoal at frame {last_subgoal_frame} deemed imprecise "
                          f"({frames_after_last_subgoal} frames until termination). Removing it from reward consideration.")
                    subgoal_indices.pop()
            # --- END: 应用你的新发现 ---

            ramp_steps = 10
            for subgoal_idx in subgoal_indices:
                # Apply the main bonus on the subgoal frame itself
                if subgoal_idx < ep_len:
                    subgoal_rewards[subgoal_idx] = stage_bonus_value
                
                # Apply linearly increasing reward for frames leading to the subgoal
                if ramp_steps > 0:
                    for k in range(1, ramp_steps + 1):
                        ramp_idx = subgoal_idx - k
                        if ramp_idx >= 0:
                            # A simple linear ramp-up. The reward is proportional to how close it is to the subgoal.
                            # The max bonus is half of the stage_bonus_value to make it distinct.
                            ramp_value = (stage_bonus_value / 2.0) * (ramp_steps - k + 1) / ramp_steps
                            # Use max() to not overwrite a potentially larger reward from an overlapping ramp
                            # or the main subgoal bonus itself.
                            subgoal_rewards[ramp_idx] = max(subgoal_rewards[ramp_idx].item(), ramp_value)

                # >>> START OF ADDED CODE <<<
                # Apply linearly decreasing reward for frames after the subgoal
                if ramp_steps > 0:
                    for k in range(1, ramp_steps + 1):
                        ramp_idx = subgoal_idx + k
                        # Ensure we don't go beyond the episode length
                        if ramp_idx < ep_len:
                            # The reward value decreases as we move further from the subgoal
                            ramp_value = (stage_bonus_value / 2.0) * (ramp_steps - k + 1) / ramp_steps
                            # Use max() to avoid overwriting a larger reward (e.g., from the next subgoal's ramp)
                            subgoal_rewards[ramp_idx] = max(subgoal_rewards[ramp_idx].item(), ramp_value)
                # >>> END OF ADDED CODE <<<

            num_subgoals = len(subgoal_indices)

            ep_rewards = torch.zeros(ep_len, dtype=torch.float32)
            ep_r1 = torch.zeros(ep_len, dtype=torch.float32)
            ep_r2 = torch.zeros(ep_len, dtype=torch.float32)
            ep_r3 = torch.zeros(ep_len, dtype=torch.float32)
            ep_r4 = torch.zeros(ep_len, dtype=torch.float32)
            ep_vel_norm = torch.zeros(ep_len, dtype=torch.float32)

            current_subgoal_progress_idx = 0

            frames_since_subgoal_switch = float("inf")

            joint_vel = torch.zeros_like(ep_states)
            joint_vel[1:] = ep_states[1:] - ep_states[:-1]
            joint_acc = torch.zeros_like(ep_states)
            joint_acc[1:] = joint_vel[1:] - joint_vel[:-1]
            action_vel = torch.zeros_like(ep_actions)
            action_vel[1:] = ep_actions[1:] - ep_actions[:-1]
            action_acc = torch.zeros_like(ep_actions)
            action_acc[1:] = action_vel[1:] - action_vel[:-1]

            for t in range(ep_len):

                # total_reward = 0.0
                # frames_since_subgoal_switch += 1
                # if current_subgoal_progress_idx < num_subgoals and t == subgoal_indices[current_subgoal_progress_idx]:
                #     total_reward = stage_bonus_value
                #     current_subgoal_progress_idx += 1
                
                r1 = subgoal_rewards[t]
                # r1 = total_reward
                r2 = 0.0
                r3 = 0.0
                r4 = 0.0
                 
                # if current_subgoal_progress_idx < num_subgoals and t >= subgoal_indices[current_subgoal_progress_idx]:
                #     frames_since_subgoal_switch = 0
                #     current_subgoal_progress_idx += 1

                # target_idx_in_list = min(current_subgoal_progress_idx, num_subgoals - 1)
                # target_subgoal_local_idx = subgoal_indices[target_idx_in_list]

                # s_t = ep_states[t]
                # s_star = ep_states[target_subgoal_local_idx]
                # o_t_feat = ep_image_features[t]
                # o_star_feat = ep_image_features[target_subgoal_local_idx]

                # mse_proprio = F.mse_loss(s_t, s_star)
                # mse_img_features = F.mse_loss(o_t_feat.float(), o_star_feat.float())
                # distance_reward = (torch.exp(-proprio_mse_scale * mse_proprio) + 
                #              torch.exp(-img_mse_scale * mse_img_features))
                # stage_reward = stage_bonus_value * current_subgoal_progress_idx
                # r1 = stage_reward + distance_reward
            
        
                # mse_proprio = F.mse_loss(s_t, s_star)
                # mse_img_features = F.mse_loss(o_t_feat.float(), o_star_feat.float())                
                # r1 = distance_reward

                # r2 = (current_subgoal_progress_idx + 1) / num_subgoals
                # r3 = 0.0
                # if frames_since_subgoal_switch > r3_grace_period:
                #     r3 = -torch.sum(joint_vel[t, :-1] ** 2) - torch.sum(joint_acc[t, :-1] ** 2) - torch.sum(action_vel[t, :-1] ** 2) - torch.sum(action_acc[t, :-1] ** 2)
                # r4 = 0.0
                
                total_reward = w["w1"] * r1 + w["w2"] * r2 + w["w3"] * r3 + w["w4"] * r4

                current_vel_norm = torch.linalg.norm(joint_vel[t])
                ep_vel_norm[t] = current_vel_norm

                ep_rewards[t] = total_reward
                ep_r1[t] = w["w1"] * r1
                ep_r2[t] = w["w2"] * r2
                ep_r3[t] = w["w3"] * r3
                ep_r4[t] = w["w4"] * r4

            mc_ret = torch.zeros_like(ep_rewards)
            prev_ret = 0.0
            for t in range(ep_len - 1, -1, -1):
                mc_ret[t] = ep_rewards[t] + self.discount * prev_ret
                prev_ret = mc_ret[t]

            done = torch.zeros(ep_len, dtype=torch.bool)
            done[-1] = True
            td_mask = torch.ones(ep_len, dtype=torch.bool)
            td_mask[-1] = False

            all_values = [ep_rewards, ep_r1, ep_r2, ep_r3, ep_r4, ep_vel_norm, ep_gripper_action, mc_ret, td_mask, done]
            for k, v in zip(reward_keys, all_values):
                reward_data[k].append(v)

        for k, v in reward_data.items():
            if v:
                reward_data[k] = torch.cat(v, dim=0)
            else:
                reward_data[k] = torch.tensor([])
        reward_hf_dataset = datasets.Dataset.from_dict(reward_data, split="train")
        reward_hf_dataset.set_transform(hf_transform_to_torch)
        return reward_hf_dataset

    def process_and_save_with_rewards(self, target_repo_id: str) -> BaseLeRobotDataset:
        logging.info(f"Starting reward processing for '{self.repo_id}'. Target: '{target_repo_id}'.")
        logging.info("Step 1: Calculating dense rewards for the loaded dataset...")
        reward_hf_dataset = self.construct_reward_data()
        logging.info("Reward calculation complete.")
        logging.info("Step 2: Defining features for the new dataset...")
        source_meta = self.meta
        new_features = source_meta.features.copy()
        reward_feature_definitions = {
            "reward": {"dtype": "float32", "shape": (1,)},
            "mc_returns": {"dtype": "float32", "shape": (1,)},
            "tdm": {"dtype": "bool", "shape": (1,)},
            "done": {"dtype": "bool", "shape": (1,)},
        }
        new_features.update(reward_feature_definitions)
        logging.info(f"Step 3: Creating new target dataset structure: {target_repo_id}...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=new_features,
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )
        logging.info("Step 4: Populating the new dataset...")
        current_episode_being_processed = -1
        default_keys = set(DEFAULT_FEATURES.keys())
        camera_keys = self.meta.camera_keys
        for i in tqdm(range(len(self.hf_dataset)), desc="Building new dataset"):
            frame_data = self.hf_dataset[i]
            reward_data = reward_hf_dataset[i]
            ep_idx = frame_data["episode_index"].item()
            if ep_idx != current_episode_being_processed:
                if current_episode_being_processed != -1:
                    target_ds.save_episode()
                current_episode_being_processed = ep_idx
            new_frame = {}
            for key, value in frame_data.items():
                if key in default_keys:
                    continue
                if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                    new_frame[key] = value.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                elif hasattr(value, "numpy"):
                    new_frame[key] = value.numpy()
                else:
                    new_frame[key] = value
            for key, value in reward_data.items():
                numpy_value = value.numpy()
                if numpy_value.shape == ():
                    new_frame[key] = numpy_value.reshape(1)
                else:
                    new_frame[key] = numpy_value
            task_string = source_meta.tasks[frame_data["task_index"].item()]
            target_ds.add_frame(new_frame, task=task_string)
        if target_ds.episode_buffer["size"] > 0:
            target_ds.save_episode()
        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready locally at: {target_ds.root.resolve()}")
        return target_ds

    def process_and_save_with_dense_rewards(self, target_repo_id: str) -> BaseLeRobotDataset:
        logging.info(f"Starting reward processing for '{self.repo_id}'. Target: '{target_repo_id}'.")

        logging.info("Step 1: Calculating dense rewards for the loaded dataset...")
        reward_hf_dataset = self.construct_dense_reward_data()
        logging.info("Reward calculation complete.")
        logging.info("Step 2: Defining features for the new dataset...")
                
        source_meta = self.meta
        new_features = source_meta.features.copy()

        reward_feature_definitions = {
            "reward": {"dtype": "float32", "shape": (1,)},
            "r1": {"dtype": "float32", "shape": (1,)},
            "r2": {"dtype": "float32", "shape": (1,)},
            "r3": {"dtype": "float32", "shape": (1,)},
            "r4": {"dtype": "float32", "shape": (1,)},
            "vel_norm": {"dtype": "float32", "shape": (1,)},
            "gripper_action": {"dtype": "float32", "shape": (1,)},
            "mc_returns": {"dtype": "float32", "shape": (1,)},
            "tdm": {"dtype": "bool", "shape": (1,)},
            "done": {"dtype": "bool", "shape": (1,)},
        }
        new_features.update(reward_feature_definitions)

        logging.info(f"Step 3: Creating new target dataset structure: {target_repo_id}...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=new_features,
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )

        logging.info("Step 4: Populating the new dataset...")
        current_episode_being_processed = -1
        default_keys = set(DEFAULT_FEATURES.keys())

        camera_keys = self.meta.camera_keys

        for i in tqdm(range(len(self.hf_dataset)), desc="Building new dataset"):
            frame_data = self.hf_dataset[i]
            reward_data = reward_hf_dataset[i]
            ep_idx = frame_data["episode_index"].item()

            if ep_idx != current_episode_being_processed:
                if current_episode_being_processed != -1:
                    target_ds.save_episode()
                current_episode_being_processed = ep_idx

            new_frame = {}
            for key, value in frame_data.items():
                if key in default_keys:
                    continue
                if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                    new_frame[key] = value.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                elif hasattr(value, "numpy"):
                    # 处理其他张量（例如 state, actions）
                    new_frame[key] = value.numpy()
                else:
                    new_frame[key] = value

            for key, value in reward_data.items():
                numpy_value = value.numpy()
                if numpy_value.shape == ():
                    new_frame[key] = numpy_value.reshape(1)
                else:
                    new_frame[key] = numpy_value

            task_string = source_meta.tasks[frame_data["task_index"].item()]
            target_ds.add_frame(new_frame, task=task_string)

        if target_ds.episode_buffer["size"] > 0:
            target_ds.save_episode()

        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready locally at: {target_ds.root.resolve()}")
        return target_ds
    
    def process_and_save_subgoal_episodes(self, target_repo_id: str, ramp_steps: int = 20) -> BaseLeRobotDataset:
        logging.info(f"Starting subgoal episode extraction for '{self.repo_id}'. Target: '{target_repo_id}'.")

        annotations_path = Path("annotations_libero_long.json")
        if not annotations_path.exists():
            raise FileNotFoundError(f"Error: annotations.json not found in the current directory '{Path.cwd()}'")
        
        with annotations_path.open('r') as f:
            successful_subgoals_data = json.load(f)
        logging.info("Successfully loaded annotations.json.")

        source_meta = self.meta
        new_features = source_meta.features.copy()

        reward_feature_definitions = {
            "reward": {"dtype": "float32", "shape": (1,)},
            "mc_returns": {"dtype": "float32", "shape": (1,)},
            "tdm": {"dtype": "bool", "shape": (1,)},
            "done": {"dtype": "bool", "shape": (1,)},
        }
        new_features.update(reward_feature_definitions)

        logging.info(f"Creating new target dataset structure: {target_repo_id}...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=new_features,
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )

        stage_bonus_value = 2.5
        
        default_keys = set(DEFAULT_FEATURES.keys())
        camera_keys = self.meta.camera_keys

        for i in tqdm(range(self.num_episodes), desc="Extracting Subgoal Episodes"):            
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start
            
            if ep_len <= ramp_steps * 2:
                continue

            episode_data_full = self.hf_dataset[ep_start:ep_end]
            ep_states = torch.stack(episode_data_full["observation.state"])
            ep_actions = torch.stack(episode_data_full["action"])
            
            subgoal_indices_original = self._find_subgoals(ep_states, ep_actions)

            if not subgoal_indices_original:
                continue

            successful_frames_for_ep = successful_subgoals_data.get(str(i), [])

            for idx, subgoal_idx_original in enumerate(subgoal_indices_original):
                should_receive_reward = subgoal_idx_original in successful_frames_for_ep

                start_frame = subgoal_idx_original - ramp_steps
                if start_frame < 0 or (start_frame + 2 * ramp_steps) > ep_len:
                    continue 

                end_frame = start_frame + 2 * ramp_steps -1
                sub_episode_len = 2 * ramp_steps

                sub_episode_rewards = torch.zeros(sub_episode_len, dtype=torch.float32)
                subgoal_idx_local = ramp_steps
                
                if should_receive_reward:
                    if subgoal_idx_local - 1 >= 0:
                        sub_episode_rewards[subgoal_idx_local - 1] = stage_bonus_value - 1
                    
                    sub_episode_rewards[subgoal_idx_local] = stage_bonus_value

                    if subgoal_idx_local + 1 < sub_episode_len:
                        sub_episode_rewards[subgoal_idx_local + 1] = stage_bonus_value - 1

                # start_frame = max(0, subgoal_idx_original - ramp_steps)
                # end_frame = min(ep_len - 1, subgoal_idx_original + ramp_steps)
    
                # sub_episode_len = end_frame - start_frame + 1                
                # sub_episode_rewards = torch.zeros(sub_episode_len, dtype=torch.float32)

                # if should_receive_reward:
                #     subgoal_idx_local = subgoal_idx_original - start_frame
                #     # Main bonus
                #     sub_episode_rewards[subgoal_idx_local] = stage_bonus_value
                #     # Ramp-up reward
                #     for k in range(1, ramp_steps + 1):
                #         ramp_idx = subgoal_idx_local - k
                #         if ramp_idx >= 0:
                #             ramp_value = (stage_bonus_value / 2.0) * (ramp_steps - k + 1) / ramp_steps
                #             sub_episode_rewards[ramp_idx] = max(sub_episode_rewards[ramp_idx].item(), ramp_value)
                #     # Ramp-down reward
                #     for k in range(1, ramp_steps + 1):
                #         ramp_idx = subgoal_idx_local + k
                #         if ramp_idx < sub_episode_len:
                #             ramp_value = (stage_bonus_value / 2.0) * (ramp_steps - k + 1) / ramp_steps
                #             sub_episode_rewards[ramp_idx] = max(sub_episode_rewards[ramp_idx].item(), ramp_value)

                # --- Calculate other reward-related fields for the sub-episode ---
                mc_ret = torch.zeros_like(sub_episode_rewards)
                prev_ret = 0.0
                for t in range(sub_episode_len - 1, -1, -1):
                    mc_ret[t] = sub_episode_rewards[t] + self.discount * prev_ret
                    prev_ret = mc_ret[t]
                
                done = torch.zeros(sub_episode_len, dtype=torch.bool)
                done[-1] = True
                
                td_mask = torch.ones(sub_episode_len, dtype=torch.bool)
                td_mask[-1] = False
                
                task_string = source_meta.tasks[episode_data_full["task_index"][0].item()]
                for j in range(sub_episode_len):
                    original_episode_local_index = start_frame + j
                    absolute_index = ep_start + original_episode_local_index
                    frame_data = self.hf_dataset[absolute_index]
                    
                    new_frame = {}
                    for key, value in frame_data.items():
                        if key in default_keys:
                            continue
                        if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                            new_frame[key] = value.permute(1, 2, 0).numpy()
                        elif hasattr(value, "numpy"):
                            new_frame[key] = value.numpy()
                        else:
                            new_frame[key] = value
                    
                    new_frame["reward"] = np.array([sub_episode_rewards[j].item()], dtype=np.float32)
                    new_frame["mc_returns"] = np.array([mc_ret[j].item()], dtype=np.float32)
                    new_frame["done"] = np.array([done[j].item()])
                    new_frame["tdm"] = np.array([td_mask[j].item()])
                    
                    target_ds.add_frame(new_frame, task=task_string)
                
                target_ds.save_episode()

        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready locally at: {target_ds.root.resolve()}")
        return target_ds
    
    def process_and_annotate_full_episodes(self, target_repo_id: str) -> BaseLeRobotDataset:
        logging.info(f"Starting full episode annotation for '{self.repo_id}'. Target: '{target_repo_id}'.")

        annotations_path = Path("annotations.json")
        if not annotations_path.exists():
            raise FileNotFoundError(f"Error: annotations.json not found in the current directory '{Path.cwd()}'")
        
        with annotations_path.open('r') as f:
            successful_subgoals_data = json.load(f)
        logging.info("Successfully loaded annotations.json.")

        source_meta = self.meta
        new_features = source_meta.features.copy()

        reward_feature_definitions = {
            "reward": {"dtype": "float32", "shape": (1,)},
            "mc_returns": {"dtype": "float32", "shape": (1,)},
            "tdm": {"dtype": "bool", "shape": (1,)},
            "done": {"dtype": "bool", "shape": (1,)},
        }
        new_features.update(reward_feature_definitions)

        logging.info(f"Creating new target dataset structure: {target_repo_id}...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=new_features,
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )

        stage_bonus_value = 1.0
        
        default_keys = set(DEFAULT_FEATURES.keys())
        camera_keys = self.meta.camera_keys

        for i in tqdm(range(self.num_episodes), desc="Annotating Full Episodes"):            
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start
            
            if ep_len <= 1:
                continue

            episode_data_full = self.hf_dataset[ep_start:ep_end]
            ep_states = torch.stack(episode_data_full["observation.state"])
            ep_actions = torch.stack(episode_data_full["action"])
            
            subgoal_indices_original = self._find_subgoals(ep_states, ep_actions)

            episode_is_successful = False
            if subgoal_indices_original:
                successful_frames_for_ep = successful_subgoals_data.get(str(i), [])
                if any(sg in successful_frames_for_ep for sg in subgoal_indices_original):
                    episode_is_successful = True

            episode_rewards = torch.zeros(ep_len, dtype=torch.float32)

            terminal_reward_frames = 5
            if episode_is_successful:
                episode_rewards[-terminal_reward_frames:] = stage_bonus_value
                logging.debug(f"Episode {i} is successful. Applying terminal reward.")
            
            # ------------------------
            mc_ret = torch.zeros_like(episode_rewards)
            prev_ret = 0.0
            for t in range(ep_len - 1, -1, -1):
                mc_ret[t] = episode_rewards[t] + self.discount * prev_ret
                prev_ret = mc_ret[t]
            
            done = torch.zeros(ep_len, dtype=torch.bool)
            done[-1] = True 
            td_mask = torch.ones(ep_len, dtype=torch.bool)
            td_mask[-1] = False 
            task_string = source_meta.tasks[episode_data_full["task_index"][0].item()]
            
            for j in range(ep_len):
                absolute_index = ep_start + j
                frame_data = self.hf_dataset[absolute_index]

                new_frame = {}
                for key, value in frame_data.items():
                    if key in default_keys:
                        continue

                    if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                        new_frame[key] = value.permute(1, 2, 0).numpy()
                    elif hasattr(value, "numpy"):
                        new_frame[key] = value.numpy()
                    else:
                        new_frame[key] = value
                
                new_frame["reward"] = np.array([episode_rewards[j].item()], dtype=np.float32)
                new_frame["mc_returns"] = np.array([mc_ret[j].item()], dtype=np.float32)
                new_frame["done"] = np.array([done[j].item()])
                new_frame["tdm"] = np.array([td_mask[j].item()])
                
                target_ds.add_frame(new_frame, task=task_string)
            
            target_ds.save_episode()

        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready locally at: {target_ds.root.resolve()}")
        return target_ds

    def get_sampling_indices_with_subgoals(self, ep_states, ep_actions, ep_len, interval_threshold=50) -> list[int]:            
        if ep_len <= 0:
            return []        
        
        # final_indices = [ep_len - 1]
        # subgoal_indices_used = []
        # return final_indices, subgoal_indices_used
        
        indices_to_sample = set()

        indices_to_sample.add(0)
        indices_to_sample.add(ep_len - 1)

        subgoal_indices = self._find_subgoals(ep_states, ep_actions)
        subgoal_indices_used = []

        for idx in subgoal_indices:
            subgoal_indices_used.append(min(idx + 10, ep_len - 1))
            indices_to_sample.add(min(idx + 10, ep_len - 1))
                 
        sorted_indices = sorted(list(indices_to_sample))
        final_indices = []

        if not sorted_indices:
            for i in range(0, ep_len, interval_threshold):
                final_indices.append(i)
            if ep_len - 1 not in final_indices:
                final_indices.append(ep_len - 1)
            return sorted(list(set(final_indices)))

        final_indices.append(sorted_indices[0])
        for i in range(1, len(sorted_indices)):
            prev_idx = sorted_indices[i-1]
            curr_idx = sorted_indices[i]
            if curr_idx - prev_idx > interval_threshold:
                mid_point = prev_idx + (curr_idx - prev_idx) // 2
                final_indices.append(mid_point)
            final_indices.append(curr_idx)

        final_indices.append(ep_len - 1)
        return sorted(list(set(final_indices))), subgoal_indices_used
    
    def process_and_label_with_vlm(self, target_repo_id: str, terminal_reward_frames: int = 5) -> BaseLeRobotDataset:
        """
        Processes episodes by labeling them with a VLM (e.g., GPT-4o) and creates a new dataset
        with sparse terminal rewards based on the 'success' or 'fail' labels.
        """
        if label_episode is None:
            raise ImportError("The 'label_episode' function from 'vlm_labeler.py' is not available.")

        logging.info(f"Starting VLM-based labeling and processing for '{self.repo_id}'. Target: '{target_repo_id}'.")

        vlm_output_root = Path("labeling_output")
        vlm_output_root.mkdir(exist_ok=True)
        logging.info(f"VLM artifacts (sampled frames, labels) will be saved to: {vlm_output_root.resolve()}")
        
        source_meta = self.meta
        new_features = source_meta.features.copy()
        reward_feature_definitions = {
            "reward": {"dtype": "float32", "shape": (1,)},
            "mc_returns": {"dtype": "float32", "shape": (1,)},
            "tdm": {"dtype": "bool", "shape": (1,)},
            "done": {"dtype": "bool", "shape": (1,)},
        }
        new_features.update(reward_feature_definitions)

        logging.info(f"Creating new target dataset structure: {target_repo_id}...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=new_features,
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )

        stage_bonus_value = 1.0
        default_keys = set(DEFAULT_FEATURES.keys())
        camera_keys = self.meta.camera_keys
        
        successful_episodes = []
        failed_episodes = []
        
        for i in tqdm(range(self.num_episodes), desc="Labeling Episodes with VLM"):
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start
            
            if ep_len <= 1:
                logging.warning(f"Skipping episode {i} due to insufficient length ({ep_len}).")
                failed_episodes.append(f"{i} (skipped)")
                continue

            episode_data_full = self.hf_dataset[ep_start:ep_end]
            ep_states = torch.stack(episode_data_full["observation.state"])
            ep_actions = torch.stack(episode_data_full["action"])

            sampled_indices, detected_subgoals = self.get_sampling_indices_with_subgoals(
                ep_states, ep_actions, ep_len
            )
            
            episode_output_dir = vlm_output_root / f"episode_{i}"
            if episode_output_dir.exists():
                shutil.rmtree(episode_output_dir)
            episode_output_dir.mkdir()

            for frame_idx in sampled_indices:
                image_tensor_main = episode_data_full["observation.images.image"][frame_idx]
                image_np_main = image_tensor_main.permute(1, 2, 0).numpy()
                image_tensor_wrist = episode_data_full["observation.images.wrist_image"][frame_idx]
                image_np_wrist = image_tensor_wrist.permute(1, 2, 0).numpy()
                base_filename = f"frame_{frame_idx:04d}{'_key_event' if frame_idx in detected_subgoals else ''}"
                filename_main = f"{base_filename}_main.png"
                filename_wrist = f"{base_filename}_wrist.png"
                filepath_main = episode_output_dir / filename_main
                filepath_wrist = episode_output_dir / filename_wrist
                pil_img_main = Image.fromarray((image_np_main * 255).astype(np.uint8))
                pil_img_main.save(filepath_main)
                pil_img_wrist = Image.fromarray((image_np_wrist * 255).astype(np.uint8))
                pil_img_wrist.save(filepath_wrist)
            
            logging.info(f"Episode {i}: Calling VLM for labeling...")
            try:
                vlm_result = label_episode(
                    episode_id=i,
                    episode_dir=str(episode_output_dir),
                    sampled_indices=sampled_indices,
                    detected_subgoals=detected_subgoals
                )
                # logging.info(f"Episode {i}: VLM label is '{vlm_result.get('label', 'N/A')}'. Reason: {vlm_result.get('reason', 'N/A')}")
                episode_is_successful = (vlm_result.get("label") == "success")
            except Exception as e:
                logging.error(f"Episode {i}: VLM labeling failed with error: {e}")
                episode_is_successful = False

            if episode_is_successful:
                successful_episodes.append(i)
            else:
                failed_episodes.append(i)

            episode_rewards = torch.zeros(ep_len, dtype=torch.float32)
            if episode_is_successful:
                start_reward_idx = max(0, ep_len - terminal_reward_frames)
                episode_rewards[start_reward_idx:] = stage_bonus_value
                logging.info(f"Episode {i} marked as SUCCESS. Applied terminal reward.")
            else:
                logging.info(f"Episode {i} marked as FAIL/UNKNOWN. No reward applied.")

            mc_ret = torch.zeros_like(episode_rewards)
            prev_ret = 0.0
            for t in range(ep_len - 1, -1, -1):
                mc_ret[t] = episode_rewards[t] + self.discount * prev_ret
                prev_ret = mc_ret[t]
            
            done = torch.zeros(ep_len, dtype=torch.bool)
            done[-1] = True 
            td_mask = torch.ones(ep_len, dtype=torch.bool)
            td_mask[-1] = False 
            task_string = source_meta.tasks[episode_data_full["task_index"][0].item()]
            
            for j in range(ep_len):
                absolute_index = ep_start + j
                frame_data = self.hf_dataset[absolute_index]
                new_frame = {}
                for key, value in frame_data.items():
                    if key in default_keys:
                        continue
                    if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                        new_frame[key] = value.permute(1, 2, 0).numpy()
                    elif hasattr(value, "numpy"):
                        new_frame[key] = value.numpy()
                    else:
                        new_frame[key] = value
                new_frame["reward"] = np.array([episode_rewards[j].item()], dtype=np.float32)
                new_frame["mc_returns"] = np.array([mc_ret[j].item()], dtype=np.float32)
                new_frame["done"] = np.array([done[j].item()])
                new_frame["tdm"] = np.array([td_mask[j].item()])
                target_ds.add_frame(new_frame, task=task_string)
            
            target_ds.save_episode()
            
        logging.info("\n" + "="*50 + "\n")
        logging.info("--- VLM Labeling Summary ---")
        logging.info(f"Total Episodes Processed: {self.num_episodes}")
        logging.info(f"Successful Episodes ({len(successful_episodes)}): {successful_episodes if successful_episodes else 'None'}")
        logging.info(f"Failed/Unknown Episodes ({len(failed_episodes)}): {failed_episodes if failed_episodes else 'None'}")
        logging.info("\n" + "="*50)
        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready locally at: {target_ds.root.resolve()}")
        return target_ds
    

        # >>> START: 新增的方法 <<<
    
    def process_and_label_from_json(self, target_repo_id: str, annotations_path: str, terminal_reward_frames: int = 5) -> BaseLeRobotDataset:
        logging.info(f"Starting labeling process from JSON file for '{self.repo_id}'. Target: '{target_repo_id}'.")

        annotations_file = Path(annotations_path)
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotation file not found at: {annotations_file.resolve()}")
        with annotations_file.open('r') as f:
            annotations = json.load(f)
        logging.info(f"Successfully loaded annotations from {annotations_file.resolve()}")

        source_meta = self.meta
        new_features = source_meta.features.copy()
        reward_feature_definitions = {
            "reward": {"dtype": "float32", "shape": (1,)},
            "mc_returns": {"dtype": "float32", "shape": (1,)},
            "tdm": {"dtype": "bool", "shape": (1,)},
            "done": {"dtype": "bool", "shape": (1,)},
        }
        new_features.update(reward_feature_definitions)

        logging.info(f"Creating new target dataset structure: {target_repo_id}...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=new_features,
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )

        stage_bonus_value = 1.0  # 成功奖励值
        default_keys = set(DEFAULT_FEATURES.keys())
        camera_keys = self.meta.camera_keys
        
        successful_episodes = []
        failed_episodes = []
        
        # 遍历所有 episode
        for i in tqdm(range(self.num_episodes), desc="Labeling Episodes from JSON"):
            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start
            
            if ep_len <= 1:
                logging.warning(f"Skipping episode {i} due to insufficient length ({ep_len}).")
                failed_episodes.append(f"{i} (skipped)")
                continue

            episode_is_successful = annotations.get(str(i), False)

            if episode_is_successful:
                successful_episodes.append(i)
            else:
                failed_episodes.append(i)

            episode_rewards = torch.zeros(ep_len, dtype=torch.float32)
            if episode_is_successful:
                start_reward_idx = max(0, ep_len - terminal_reward_frames)
                episode_rewards[start_reward_idx:] = stage_bonus_value

            mc_ret = torch.zeros_like(episode_rewards)
            prev_ret = 0.0
            for t in range(ep_len - 1, -1, -1):
                mc_ret[t] = episode_rewards[t] + self.discount * prev_ret
                prev_ret = mc_ret[t]
            
            done = torch.zeros(ep_len, dtype=torch.bool)
            done[-1] = True 
            td_mask = torch.ones(ep_len, dtype=torch.bool)
            td_mask[-1] = False 
            
            episode_data_full = self.hf_dataset[ep_start:ep_end]
            task_string = source_meta.tasks[episode_data_full["task_index"][0].item()]
            
            # 将原始数据和新的奖励数据写入新数据集中
            for j in range(ep_len):
                absolute_index = ep_start + j
                frame_data = self.hf_dataset[absolute_index]
                new_frame = {}
                for key, value in frame_data.items():
                    if key in default_keys:
                        continue
                    if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                        new_frame[key] = value.permute(1, 2, 0).numpy()
                    elif hasattr(value, "numpy"):
                        new_frame[key] = value.numpy()
                    else:
                        new_frame[key] = value

                new_frame["reward"] = np.array([episode_rewards[j].item()], dtype=np.float32)
                new_frame["mc_returns"] = np.array([mc_ret[j].item()], dtype=np.float32)
                new_frame["done"] = np.array([done[j].item()])
                new_frame["tdm"] = np.array([td_mask[j].item()])
                target_ds.add_frame(new_frame, task=task_string)
            
            target_ds.save_episode()
            
        logging.info("\n" + "="*50 + "\n")
        logging.info("--- JSON Labeling Summary ---")
        logging.info(f"Total Episodes Processed: {self.num_episodes}")
        logging.info(f"Successful Episodes ({len(successful_episodes)}): {successful_episodes if successful_episodes else 'None'}")
        logging.info(f"Failed/Skipped Episodes ({len(failed_episodes)}): {failed_episodes if failed_episodes else 'None'}")
        logging.info("\n" + "="*50)
        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready locally at: {target_ds.root.resolve()}")
        return target_ds

    def filter_episodes_from_json(self, target_repo_id: str, annotations_path: str) -> BaseLeRobotDataset:
        """
        Filters a dataset to keep only the episodes marked as 'true' in a JSON annotation file.
        This method creates a new, clean LeRobotDataset with the exact same features as the source,
        without adding any reward information.
        """
        logging.info(f"Starting to filter for successful episodes from '{self.repo_id}'. Target: '{target_repo_id}'.")
        logging.info("Mode: Pure filtering, no rewards will be added.")

        # 1. 加载并解析JSON标注文件
        annotations_file = Path(annotations_path)
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotation file not found at: {annotations_file.resolve()}")
        with annotations_file.open('r') as f:
            annotations = json.load(f)
        
        successful_episode_indices = {int(ep_idx) for ep_idx, is_successful in annotations.items() if is_successful}
        logging.info(f"Found {len(successful_episode_indices)} successful episodes marked as 'true' in '{annotations_path}'.")

        if not successful_episode_indices:
            logging.warning("No successful episodes found. The resulting dataset will be empty.")

        # 2. 使用源数据集的原始特征创建新数据集
        source_meta = self.meta
        
        logging.info(f"Creating new target dataset '{target_repo_id}' with original features...")
        target_ds = BaseLeRobotDataset.create(
            repo_id=target_repo_id,
            fps=source_meta.fps,
            features=source_meta.features,  # <-- 使用原始特征，不添加任何reward
            robot_type=source_meta.robot_type,
            use_videos=(source_meta.video_path is not None),
            image_writer_threads=4,
        )

        default_keys = set(DEFAULT_FEATURES.keys())
        camera_keys = self.meta.camera_keys
        
        # 3. 遍历源数据集，只复制成功的 episode
        for i in tqdm(range(self.num_episodes), desc="Filtering Successful Episodes"):
            if i not in successful_episode_indices:
                continue

            ep_start = self.episode_data_index["from"][i].item()
            ep_end = self.episode_data_index["to"][i].item()
            ep_len = ep_end - ep_start
            
            episode_data_full = self.hf_dataset[ep_start:ep_end]
            task_string = source_meta.tasks[episode_data_full["task_index"][0].item()]
            
            # 逐帧复制数据到新数据集中
            for j in range(ep_len):
                absolute_index = ep_start + j
                frame_data = self.hf_dataset[absolute_index]
                new_frame = {}
                # 仔细复制每一帧的数据，不添加任何新键
                for key, value in frame_data.items():
                    if key in default_keys:
                        continue
                    if isinstance(value, torch.Tensor) and value.ndim == 3 and key in camera_keys:
                        new_frame[key] = value.permute(1, 2, 0).numpy()
                    elif hasattr(value, "numpy"):
                        new_frame[key] = value.numpy()
                    else:
                        new_frame[key] = value
                
                # 不添加任何 reward 相关的键
                target_ds.add_frame(new_frame, task=task_string)
            
            target_ds.save_episode()
            
        logging.info("\n" + "="*50)
        logging.info("--- Pure Filtering Summary ---")
        logging.info(f"Total episodes in new dataset: {target_ds.num_episodes}")
        logging.info(f"Features in new dataset: {list(target_ds.features.keys())}")
        logging.info("="*50)
        logging.info(f"\nProcessing complete! New dataset '{target_repo_id}' is ready at: {target_ds.root.resolve()}")
        return target_ds

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
       
        key_mapping = {}
        item = self.hf_dataset[idx]

        ep_idx = item["episode_index"].item()
        ep_end = self.episode_data_index["to"][ep_idx].item()

        offset = self.next_obs_offset
        vqh_idx = min(idx + offset, ep_end - 1)
        vqh_obs_item = self.hf_dataset[vqh_idx]

        # --- START: Logic for current observation ---
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
            
        # --- START: Added code to get VQH (next observation) data ---
        vqh_images = {key: vqh_obs_item[key] for key in self.meta.camera_keys if key in vqh_obs_item}
        vqh_item = {**vqh_images}
        for k, v in vqh_item.items():
            item[f"{k}.vqh"] = v

        next_state = vqh_obs_item["observation.state"]
        item["observation.state.vqh"] = next_state
        # --- END: Added code to get VQH data ---

        # --- START: Modified image transformation logic ---
        camera_keys = self.meta.camera_keys
        all_keys_to_transform = list(camera_keys) + [f"{key}.vqh" for key in camera_keys]
        if self.image_transforms is not None:
            for cam_key in all_keys_to_transform:
                if cam_key in item:
                    item[cam_key] = self.image_transforms(item[cam_key])
        # --- END: Modified image transformation logic ---

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        return item
