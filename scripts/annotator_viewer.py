import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import torch
import logging
import argparse
import json
from pathlib import Path
import time

# 确保 reward_dataset.py 在同一目录或可访问
from reward_dataset import RewardLeRobotDataset

class SubgoalAnnotatorApp:
    def __init__(self, dataset, annotations_path, context_frames=10, playback_fps=10):
        self.dataset = dataset
        self.annotations_path = Path(annotations_path)
        self.context_frames = context_frames
        self.playback_interval = 1.0 / playback_fps
        
        self.current_episode_idx = 0
        self.num_episodes = self.dataset.num_episodes

        # --- Annotation & Frame Data ---
        self.annotations = self._load_annotations_from_disk()
        self.candidate_subgoals = []
        self.current_subgoal_cursor = 0
        self.frame_window_cache = []
        self.center_frame_in_episode = 0
        self.is_playing = False

        # --- UI Setup (与V3版本相同) ---
        self.fig, self.ax_img = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)
        self.img_display = self.ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        self.ax_img.axis("off")
        
        ax_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
        self.frame_slider = Slider(ax=ax_slider, label='Frame Context', valmin=-self.context_frames, valmax=self.context_frames, valinit=0, valstep=1)
        self.frame_slider.on_changed(self.update_from_slider)

        btn_width, btn_height, btn_y_pos = 0.1, 0.05, 0.05
        
        ax_prev_ep = plt.axes([0.05, btn_y_pos, btn_width, btn_height])
        self.btn_prev_ep = Button(ax_prev_ep, '<< Episode')
        self.btn_prev_ep.on_clicked(lambda e: self.switch_episode(-1))

        ax_next_ep = plt.axes([0.16, btn_y_pos, btn_width, btn_height])
        self.btn_next_ep = Button(ax_next_ep, 'Episode >>')
        self.btn_next_ep.on_clicked(lambda e: self.switch_episode(1))

        ax_play = plt.axes([0.31, btn_y_pos, btn_width, btn_height])
        self.btn_play = Button(ax_play, 'Play (▶)')
        self.btn_play.on_clicked(self.toggle_playback)

        ax_reject = plt.axes([0.46, btn_y_pos, btn_width, 0.075])
        self.btn_reject = Button(ax_reject, 'Reject (←)', color='salmon')
        self.btn_reject.on_clicked(self.reject_subgoal)

        ax_accept = plt.axes([0.57, btn_y_pos, btn_width, 0.075])
        self.btn_accept = Button(ax_accept, 'Accept (→)', color='lightgreen')
        self.btn_accept.on_clicked(self.accept_subgoal)

        ax_save = plt.axes([0.85, btn_y_pos, btn_width, btn_height])
        self.btn_save = Button(ax_save, 'Save & Exit')
        self.btn_save.on_clicked(self.save_and_exit)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.load_episode_data()
        self.load_subgoal()

    def _load_annotations_from_disk(self):
        if self.annotations_path.exists():
            print(f"Loading existing annotations from {self.annotations_path}")
            with open(self.annotations_path, 'r') as f:
                try: return {int(k): v for k, v in json.load(f).items()}
                except json.JSONDecodeError:
                    print("Warning: Annotation file is corrupted or empty. Starting fresh.")
                    return {}
        print("No existing annotations file found. Starting fresh.")
        return {}

    def _save_annotations_to_disk(self):
        try:
            with open(self.annotations_path, 'w') as f:
                json.dump(dict(sorted(self.annotations.items())), f, indent=4)
        except Exception as e:
            print(f"\n--- ERROR: FAILED TO SAVE ANNOTATIONS: {e} ---")

    def save_and_exit(self, event=None):
        self.is_playing = False
        self._save_annotations_to_disk()
        print(f"\nFinal annotations saved. Exiting.")
        plt.close(self.fig)

    def load_episode_data(self):
        self.is_playing = False
        print(f"\nLoading Episode {self.current_episode_idx}...")
        ep_start = self.dataset.episode_data_index["from"][self.current_episode_idx].item()
        ep_end = self.dataset.episode_data_index["to"][self.current_episode_idx].item()
        episode_data = self.dataset.hf_dataset[ep_start:ep_end]
        ep_states = torch.stack(episode_data["observation.state"])
        ep_actions = torch.stack(episode_data["action"])
        self.candidate_subgoals = self.dataset._find_subgoals(ep_states, ep_actions)
        if self.current_episode_idx not in self.annotations:
            self.annotations[self.current_episode_idx] = []
        self.current_subgoal_cursor = 0
        if not self.candidate_subgoals:
            print("No candidate subgoals found in this episode.")

    def load_subgoal(self):
        self.is_playing = False
        self.frame_window_cache.clear()
        if not self.candidate_subgoals or self.current_subgoal_cursor >= len(self.candidate_subgoals):
            self.display_frame(None)
            return

        self.center_frame_in_episode = self.candidate_subgoals[self.current_subgoal_cursor]
        ep_start = self.dataset.episode_data_index["from"][self.current_episode_idx].item()
        ep_len = self.dataset.episode_data_index["to"][self.current_episode_idx].item() - ep_start
        start_frame = max(0, self.center_frame_in_episode - self.context_frames)
        end_frame = min(ep_len, self.center_frame_in_episode + self.context_frames + 1)
        for frame_offset in range(start_frame, end_frame):
            self.frame_window_cache.append(self.dataset.hf_dataset[ep_start + frame_offset]['observation.images.image'])
        
        self.frame_slider.set_val(0)
        self.update_from_slider(0)

    def display_frame(self, img_tensor, relative_offset=None):
        if img_tensor is None:
            title = f"Episode {self.current_episode_idx}"
            if not self.candidate_subgoals: title += " (No subgoals found)"
            elif self.current_subgoal_cursor >= len(self.candidate_subgoals): title += " (All subgoals reviewed)"
            self.ax_img.set_title(title, fontsize=12)
            self.img_display.set_data(np.zeros((224, 224, 3)))
            self.fig.canvas.draw_idle()
            return
            
        img_to_show = np.clip(img_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)
        self.img_display.set_data(img_to_show)

        status = "ACCEPTED" if self.center_frame_in_episode in self.annotations.get(self.current_episode_idx, []) else "PENDING"
        offset_str = f"({relative_offset:+d})" if relative_offset is not None else ""
        title = (f"Ep {self.current_episode_idx} | Subgoal {self.current_subgoal_cursor + 1}/{len(self.candidate_subgoals)} "
                 f"(Frame {self.center_frame_in_episode}) {offset_str} | Status: {status}")
        self.ax_img.set_title(title, fontsize=12)
        self.fig.canvas.draw_idle()

    def update_from_slider(self, val):
        relative_offset = int(val)
        center_in_cache = self.center_frame_in_episode - max(0, self.center_frame_in_episode - self.context_frames)
        idx_in_cache = center_in_cache + relative_offset
        if 0 <= idx_in_cache < len(self.frame_window_cache):
            self.display_frame(self.frame_window_cache[idx_in_cache], relative_offset)

    def _update_annotation(self, accept: bool):
        if not self.candidate_subgoals or self.current_subgoal_cursor >= len(self.candidate_subgoals): return

        self.is_playing = False
        frame_idx = self.center_frame_in_episode
        accepted_set = set(self.annotations.get(self.current_episode_idx, []))
        action_str = "Accepted" if accept else "Rejected"
        if accept: accepted_set.add(frame_idx)
        else: accepted_set.discard(frame_idx)
        self.annotations[self.current_episode_idx] = sorted(list(accepted_set))
        
        print(f"  {action_str} subgoal at frame {frame_idx}. Annotations auto-saved.")
        self._save_annotations_to_disk()
        
        self.current_subgoal_cursor += 1

        # --- START: 自动跳转的核心改动 ---
        if self.current_subgoal_cursor >= len(self.candidate_subgoals):
            # 如果当前 episode 的子目标已全部标注完
            print("Episode complete. Automatically moving to the next episode...")
            self.switch_episode(1) # 自动跳转到下一个 episode
        else:
            # 否则，加载当前 episode 的下一个子目标
            self.load_subgoal()
        # --- END: 自动跳转的核心改动 ---
        
    def accept_subgoal(self, event=None): self._update_annotation(accept=True)
    def reject_subgoal(self, event=None): self._update_annotation(accept=False)

    def switch_episode(self, direction: int):
        new_idx = self.current_episode_idx + direction
        if 0 <= new_idx < self.num_episodes:
            self.current_episode_idx = new_idx
            self.load_episode_data()
            self.load_subgoal()
        elif direction > 0:
             print("\nAll episodes have been annotated! You are a hero! Please Save & Exit.")
             # 清空视图，防止误操作
             self.candidate_subgoals = [] 
             self.display_frame(None)
        
    def toggle_playback(self, event=None):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.label.set_text('Play (▶)')
            return

        self.is_playing = True
        self.btn_play.label.set_text('Stop (■)')
        start_time, frame_count, current_val = time.time(), 0, self.frame_slider.val
        
        while self.is_playing:
            current_val = (current_val + 1)
            if current_val > self.context_frames: current_val = -self.context_frames
            self.frame_slider.set_val(current_val)
            
            frame_count += 1
            next_frame_time = start_time + frame_count * self.playback_interval
            plt.pause(max(0, next_frame_time - time.time()))
            if not self.is_playing: break
        
        self.btn_play.label.set_text('Play (▶)')
        self.is_playing = False

    def on_key_press(self, event):
        if event.key == "right": self.accept_subgoal()
        elif event.key == "left": self.reject_subgoal()
        elif event.key == 'pagedown': self.switch_episode(1)
        elif event.key == 'pageup': self.switch_episode(-1)
        elif event.key == ' ': self.toggle_playback()
        elif event.key == ',' or event.key == '<':
            self.is_playing = False
            if (new_val := self.frame_slider.val - 1) >= -self.context_frames: self.frame_slider.set_val(new_val)
        elif event.key == '.' or event.key == '>':
            self.is_playing = False
            if (new_val := self.frame_slider.val + 1) <= self.context_frames: self.frame_slider.set_val(new_val)

    def show(self):
        print("\n--- Annotator Controls (v4 - Auto-Next) ---")
        print("  - Inspect:  Use slider, or press < , > to step frame-by-frame.")
        print("  - Playback: Press [Space Bar] or the [Play] button.")
        print("  - Annotate: Press [←] to Reject, [→] to Accept.")
        print("  - Navigate: [PageUp]/[PageDown] for manual episode switch.")
        print("  - Feature:  Finishing an episode automatically loads the next one.")
        print("  - Safety:   Annotations are saved automatically after every action.")
        print("---------------------------------------------")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual annotation tool for robot subgoals (V4).")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID of the dataset.")
    parser.add_argument("--annotations_path", type=str, default="./annotations.json", help="Path for auto-saving annotations.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        dataset = RewardLeRobotDataset(repo_id=args.repo_id)
        app = SubgoalAnnotatorApp(dataset, args.annotations_path)
        app.show()
    except Exception as e:
        import traceback
        print(f"\n--- ERROR ---")
        print(f"Failed to run the annotator: {e}")
        traceback.print_exc()