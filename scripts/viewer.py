import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.gridspec import GridSpec

from tqdm import tqdm
import numpy as np
import torch
import logging
import argparse
import traceback

from reward_dataset import RewardLeRobotDataset
from vqh_training_policy import VQHPolicy, TrainVQHConfig, VQHPolicyConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot_patch import make_dataset, make_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.factory import resolve_delta_timestamps

class EpisodeViewerApp:
    def __init__(self, dataset, policy=None):
        self.dataset = dataset
        self.policy = policy

        self._is_programmatic_update = False

        if self.policy:
            self.device = next(self.policy.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.current_episode_idx = 0
        self.num_episodes = self.dataset.num_episodes

        self.images = []
        self.wrist_images = [] # Added for wrist images
        self.data = {}
        self.current_frame_idx = 0
        self.max_frame_idx = 0
        self.fig = plt.figure(figsize=(12, 10)) # Adjusted figure size for two images

        # Modified GridSpec and layout for two images
        if self.policy:
            self.fig.set_figheight(10)
            self.plot_keys = ["Trajectory Reward", "Q Value [mc_returns]", "Predicted Q Value"]
            gs = GridSpec(4, 2, height_ratios=[12, 2, 2, 1], width_ratios=[1, 1])
            self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, hspace=0.6, wspace=0.1)
            plot_colors = {
                "Trajectory Reward": "blue",
                "Q Value [mc_returns]": "green",
                "Predicted Q Value": "magenta",
            }
            last_plot_key = "Predicted Q Value"
        else:
            self.fig.set_figheight(6)
            self.plot_keys = ["Trajectory Reward", "Q Value [mc_returns]"]
            gs = GridSpec(3, 2, height_ratios=[12, 2, 1], width_ratios=[1, 1])
            self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.95, hspace=0.5, wspace=0.1)
            plot_colors = {
                "Trajectory Reward": "blue",
                "Q Value [mc_returns]": "green",
            }
            last_plot_key = "Q Value [mc_returns]"

        # Main image axis
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_img.axis("off")
        self.info_text_overlay = self.ax_img.text(
            0.02, 0.98, "", transform=self.ax_img.transAxes, color="white", fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6)
        )

        # Wrist image axis
        self.ax_wrist_img = self.fig.add_subplot(gs[0, 1])
        self.ax_wrist_img.axis("off")


        self.data_axes = {}
        self.indicator_lines = {}
        self.line_artists = {}

        for i, key in enumerate(self.plot_keys):
            # Plots now span both columns
            ax = self.fig.add_subplot(gs[i + 1, :])
            (line,) = ax.plot([], [], color=plot_colors[key], alpha=0.9)
            self.line_artists[key] = line
            ax.set_title(f"{key}", fontsize=10)
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            self.indicator_lines[key] = ax.axvline(x=0, color="black", linestyle=":")
            self.data_axes[key] = ax

        self.data_axes[last_plot_key].set_xlabel("Frame")

        # Image displays
        self.img_display = self.ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        self.wrist_img_display = self.ax_wrist_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))


        ax_prev_ep = plt.axes([0.1, 0.01, 0.12, 0.035])
        ax_next_ep = plt.axes([0.23, 0.01, 0.12, 0.035])
        self.btn_prev_ep = Button(ax_prev_ep, "Prev Episode")
        self.btn_next_ep = Button(ax_next_ep, "Next Episode")
        self.btn_prev_ep.on_clicked(lambda e: self.switch_episode(-1))
        self.btn_next_ep.on_clicked(lambda e: self.switch_episode(1))

        ax_prev_fr = plt.axes([0.65, 0.01, 0.1, 0.035])
        ax_next_fr = plt.axes([0.76, 0.01, 0.1, 0.035])
        self.btn_prev_fr = Button(ax_prev_fr, "Previous")
        self.btn_next_fr = Button(ax_next_fr, "Next")
        self.btn_prev_fr.on_clicked(self.prev_frame)
        self.btn_next_fr.on_clicked(self.next_frame)

        ax_episode_input = plt.axes([0.40, 0.01, 0.2, 0.035])
        self.text_box_episode = TextBox(ax_episode_input, "GoTo:", initial=str(self.current_episode_idx))
        self.text_box_episode.on_submit(self.submit_episode_jump)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.load_and_display_episode(0)

    def load_and_display_episode(self, episode_idx):
        print(f"\nLoading data for episode {episode_idx}...")
        self.current_episode_idx = episode_idx
        self.current_frame_idx = 0

        if hasattr(self, 'text_box_episode'):
            self._is_programmatic_update = True
            self.text_box_episode.set_val(str(self.current_episode_idx))
            self._is_programmatic_update = False

        self.load_episode_data(episode_idx)
        self._update_plots_for_new_episode()
        self.update_all_elements()

    @torch.no_grad()
    def _compute_predicted_q_values(self, episode_data):
        if not self.policy:
            return [0.0] * len(episode_data.get("observation.images.image", []))
        
        print("Computing predicted Q-values for the episode in batches...")
        
        batch_for_stacking = {key: val for key, val in episode_data.items() if val}
        if not batch_for_stacking or "action" not in batch_for_stacking or "observation.images.image" not in batch_for_stacking:
            print("Warning: Missing essential data (action or image) for Q-value prediction. Skipping.")
            return [float('nan')] * len(episode_data.get("observation.images.image", []))

        batch_size = 32 
        num_samples = len(batch_for_stacking["action"])
        all_predicted_q_values = []

        for i in tqdm(range(0, num_samples, batch_size), desc="Predicting Q-values"):
            end_i = min(i + batch_size, num_samples)
            
            mini_batch_data = {
                key: torch.stack(val[i:end_i]).to(self.device)
                for key, val in batch_for_stacking.items()
                if key != 'task' and isinstance(val, list) and len(val) == num_samples
            }
            if 'task' in batch_for_stacking:
                mini_batch_data['task'] = batch_for_stacking['task'][i:end_i]

            if not mini_batch_data:
                continue

            ground_truth_action = self.policy.prepare_action(mini_batch_data)
            candidate_actions = ground_truth_action.unsqueeze(1)
            
            # predicted_q_values = self.policy.evaluate_q_values(mini_batch_data, candidate_actions)
            predicted_q_values = self.policy.evaluate_q_values_with_elite_logic(mini_batch_data, candidate_actions)
                                
            predicted_q_values = predicted_q_values.squeeze(1)
            
            all_predicted_q_values.extend(predicted_q_values.cpu().tolist())

        return all_predicted_q_values

    def load_episode_data(self, episode_idx):
        start_idx = self.dataset.episode_data_index["from"][episode_idx]
        end_idx = self.dataset.episode_data_index["to"][episode_idx]

        self.images = []
        self.wrist_images = []
        self.data = {key: [] for key in self.plot_keys}

        episode_batch_data = {
            "observation.images.image": [],
            "observation.images.wrist_image":[],
            "observation.state": [],
            "task":[],
            "action": [],
            "mc_returns": [],
            "reward": [],
            "r1": [], "r2": [], "r3": [], "r4": [],
            "vel_norm": [], "gripper_action": []
        }
        data_keys_to_load = list(episode_batch_data.keys())

        image_key = "observation.images.image"
        wrist_image_key = "observation.images.wrist_image" 

        for i in tqdm(range(start_idx, end_idx), desc=f"Fetching ep {episode_idx} data"):
            sample = self.dataset[i]
            if sample:
                for key in data_keys_to_load:
                    if key in sample:
                        tensor = sample[key]
                        if isinstance(tensor, torch.Tensor):
                            episode_batch_data[key].append(tensor.cpu())
                        else:
                             episode_batch_data[key].append(tensor)

        self.images = episode_batch_data.get(image_key, [])
        self.wrist_images = episode_batch_data.get(wrist_image_key, [])

        if self.policy and self.images:
            predicted_qs = self._compute_predicted_q_values(episode_batch_data)
            self.data["Predicted Q Value"] = predicted_qs

        data_keys_map = {
            "Trajectory Reward": "reward", "Q Value [mc_returns]": "mc_returns",
            "r1": "r1", "r2": "r2", "r3": "r3", "r4": "r4",
            "vel_norm": "vel_norm", "gripper_action": "gripper_action"
        }

        for plot_key, sample_key in data_keys_map.items():
            if sample_key == "mc_returns" and self.policy:
                mc_returns_tensors = episode_batch_data.get("mc_returns", [])
                if not mc_returns_tensors:
                    self.data[plot_key] = []
                    continue

                mc_returns_raw = torch.stack(mc_returns_tensors).to(self.device)

                if "mc_returns" in self.policy.normalize_targets.stats:
                    stats = self.policy.normalize_targets.stats["mc_returns"]
                    mc_mean = torch.tensor(stats["mean"], device=self.device, dtype=mc_returns_raw.dtype)
                    mc_std = torch.tensor(stats["std"], device=self.device, dtype=mc_returns_raw.dtype)
                    normalized_mc_returns = (mc_returns_raw - mc_mean) / (mc_std + 1e-8)
                    self.data[plot_key] = normalized_mc_returns.cpu().tolist()
                else:
                    self.data[plot_key] = [t.item() for t in mc_returns_tensors]

                continue
            self.data[plot_key] = [t.item() for t in episode_batch_data.get(sample_key, [])]

        if not self.images:
            self.max_frame_idx = 0
            print(f"Warning: No images with key '{image_key}' found in episode {episode_idx}.")
        else:
            self.max_frame_idx = len(self.images) - 1

    def _update_plots_for_new_episode(self):
        for key in self.plot_keys:
            ax = self.data_axes[key]
            line = self.line_artists[key]

            new_data = self.data.get(key, [])
            if new_data:
                line.set_data(range(len(new_data)), new_data)
                ax.set_xlim(0, self.max_frame_idx)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            else:
                line.set_data([], [])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        if self.data_axes.get("vel_norm") and hasattr(self.dataset, "vel_threshold"):
            for child in self.data_axes["vel_norm"].get_children():
                if isinstance(child, plt.Line2D) and child.get_linestyle() == "-.":
                    child.remove()
            self.data_axes["vel_norm"].axhline(y=self.dataset.vel_threshold, color="magenta", linestyle="-.", label=f"Threshold ({self.dataset.vel_threshold})")
            self.data_axes["vel_norm"].legend(loc="upper right")

    def switch_episode(self, direction: int):
        new_episode_idx = self.current_episode_idx + direction
        if 0 <= new_episode_idx < self.num_episodes:
            self.load_and_display_episode(new_episode_idx)
        else:
            print(f"Already at {'first' if direction == -1 else 'last'} episode.")

    def submit_episode_jump(self, text):
        if self._is_programmatic_update:
            return

        try:
            target_episode_idx = int(text)
            if 0 <= target_episode_idx < self.num_episodes:
                self.load_and_display_episode(target_episode_idx)
            else:
                print(f"Error: Episode index {target_episode_idx} is out of range. Please enter a number between 0 and {self.num_episodes - 1}.")
        except ValueError:
            print(f"Error: Invalid input. '{text}' is not a valid episode number.")

        self._is_programmatic_update = True
        self.text_box_episode.set_val(str(self.current_episode_idx))
        self._is_programmatic_update = False

    def on_key_press(self, event):
        if event.key == "right":
            self.next_frame(None)
        elif event.key == "left":
            self.prev_frame(None)
        elif event.key == "pagedown":
            self.switch_episode(1)
        elif event.key == "pageup":
            self.switch_episode(-1)

    def update_all_elements(self):
        # Update main image
        if not self.images or self.current_frame_idx >= len(self.images):
            self.img_display.set_data(np.zeros((224, 224, 3), dtype=np.uint8))
            self.ax_img.set_title(f"Episode {self.current_episode_idx} | No Image Data")
            self.info_text_overlay.set_text("")
        else:
            tensor_img = self.images[self.current_frame_idx]
            if tensor_img.dim() == 4:
                img_to_show = tensor_img[0].permute(1, 2, 0).cpu().numpy()
            else:
                img_to_show = tensor_img.permute(1, 2, 0).cpu().numpy()

            if img_to_show.min() < 0:
                img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())
            img_to_show = np.clip(img_to_show, 0, 1)

            self.img_display.set_data(img_to_show)
            self.ax_img.set_title(f"Episode {self.current_episode_idx} | Frame {self.current_frame_idx} / {self.max_frame_idx}")

            overlay_text_parts = []
            overlay_plot_keys = ["Trajectory Reward", "Q Value [mc_returns]", "Predicted Q Value"]
            for key in overlay_plot_keys:
                if key in self.data and self.current_frame_idx < len(self.data[key]):
                    overlay_text_parts.append(f"{key}: {self.data[key][self.current_frame_idx]:.3f}")
            overlay_text = "\n".join(overlay_text_parts)
            self.info_text_overlay.set_text(overlay_text)

        # Update wrist image
        if not self.wrist_images or self.current_frame_idx >= len(self.wrist_images):
            self.wrist_img_display.set_data(np.zeros((224, 224, 3), dtype=np.uint8))
            self.ax_wrist_img.set_title("No Wrist Image Data")
        else:
            tensor_img = self.wrist_images[self.current_frame_idx]
            if tensor_img.dim() == 4:
                img_to_show = tensor_img[0].permute(1, 2, 0).cpu().numpy()
            else:
                img_to_show = tensor_img.permute(1, 2, 0).cpu().numpy()

            if img_to_show.min() < 0:
                img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())
            img_to_show = np.clip(img_to_show, 0, 1)

            self.wrist_img_display.set_data(img_to_show)
            self.ax_wrist_img.set_title("Wrist Camera")

        # Update plot indicators
        for key in self.indicator_lines:
            self.indicator_lines[key].set_xdata([self.current_frame_idx])

        self.fig.canvas.draw_idle()

    def next_frame(self, event):
        if self.current_frame_idx < self.max_frame_idx:
            self.current_frame_idx += 1
            self.update_all_elements()

    def prev_frame(self, event):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_all_elements()

    def show(self):
        plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visualize dataset episodes with optional policy predictions.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID of the dataset.")
    parser.add_argument(
        "--policy_path", 
        type=str, 
        default=None, 
        help="Optional path to the policy model folder. If provided, the viewer will display Q-value predictions."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print("Initializing dataset and policy for the viewer app...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading pre-processed dataset: '{args.repo_id}'")

        policy_cfg = VQHPolicyConfig()
        ds_meta = LeRobotDatasetMetadata(args.repo_id)
        delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
        full_dataset = RewardLeRobotDataset(repo_id=args.repo_id, delta_timestamps=delta_timestamps, next_obs_offset=3, video_backend='pyav')

        policy = None
        if args.policy_path:
            logging.info("--- Policy loading is enabled ---")
            logging.info("Loading dataset statistics for policy normalization.")
            dataset_stats = full_dataset.stats
            logging.info(f"Loading policy from: {args.policy_path}")
            policy = VQHPolicy.from_pretrained(args.policy_path, dataset_stats=dataset_stats)
            policy.to(device)
            policy.eval()
        app = EpisodeViewerApp(full_dataset, policy=policy)
        app.show()

    except Exception as e:
        print("\n--- ERROR ---")
        print(f"Failed to load or run the viewer: {e}")
        traceback.print_exc()
