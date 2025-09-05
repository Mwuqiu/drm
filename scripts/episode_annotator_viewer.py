import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import torch
import logging
import argparse
import json
from pathlib import Path
import time
# 假设您的数据加载类 RewardLeRobotDataset 保持不变
from reward_dataset import RewardLeRobotDataset

class EpisodeAnnotatorApp:
    # MODIFIED: 移除了 context_frames, 因为我们现在要看整个 episode
    def __init__(self, dataset, annotations_path, playback_fps=30):
        self.dataset = dataset
        self.annotations_path = Path(annotations_path)
        self.playback_interval = 1.0 / playback_fps
        
        self.current_episode_idx = 0
        self.num_episodes = self.dataset.num_episodes

        # --- Annotation & Frame Data ---
        self.annotations = self._load_annotations_from_disk()
        self.episode_frame_cache = [] 
        self.is_playing = False

        self.fig, self.ax_img = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)
        self.img_display = self.ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        self.ax_img.axis("off")
        
        ax_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
        self.frame_slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=1, valinit=0, valstep=1)
        self.frame_slider.on_changed(self.update_display)

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

        # MODIFIED: 修改按钮标签和功能
        ax_incorrect = plt.axes([0.46, btn_y_pos, btn_width, 0.075])
        self.btn_incorrect = Button(ax_incorrect, 'Incorrect (←)', color='salmon')
        self.btn_incorrect.on_clicked(self.mark_incorrect)

        ax_correct = plt.axes([0.57, btn_y_pos, btn_width, 0.075])
        self.btn_correct = Button(ax_correct, 'Correct (→)', color='lightgreen')
        self.btn_correct.on_clicked(self.mark_correct)

        ax_save = plt.axes([0.85, btn_y_pos, btn_width, btn_height])
        self.btn_save = Button(ax_save, 'Save & Exit')
        self.btn_save.on_clicked(self.save_and_exit)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.load_episode_data() # MODIFIED: 只需要加载 episode 数据

    def _load_annotations_from_disk(self):
        if self.annotations_path.exists():
            print(f"Loading existing annotations from {self.annotations_path}")
            with open(self.annotations_path, 'r') as f:
                try: 
                    # MODIFIED: 值可以是 bool, str 或任何 JSON 支持的类型
                    return {int(k): v for k, v in json.load(f).items()}
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

    # MODIFIED: 大幅简化, 只加载整个 episode 的图像帧
    def load_episode_data(self):
        self.is_playing = False
        print(f"\nLoading Episode {self.current_episode_idx}...")
        
        self.episode_frame_cache.clear()
        
        ep_start = self.dataset.episode_data_index["from"][self.current_episode_idx].item()
        ep_end = self.dataset.episode_data_index["to"][self.current_episode_idx].item()
        
        # 加载所有帧的图像数据到缓存
        for i in range(ep_start, ep_end):
            self.episode_frame_cache.append(self.dataset.hf_dataset[i]['observation.images.image'])
            
        if not self.episode_frame_cache:
            print("Warning: Episode has no frames.")
            self.update_display(0)
            return

        # 更新滑动条的范围和值
        self.frame_slider.ax.set_xlim(0, len(self.episode_frame_cache) - 1)
        self.frame_slider.valmax = len(self.episode_frame_cache) - 1
        self.frame_slider.set_val(0)
        
        self.update_display(0)

    # REMOVED: `load_subgoal` 方法被完全移除

    # MODIFIED: 重命名并修改了显示逻辑
    def update_display(self, frame_idx_float):
        frame_idx = int(frame_idx_float)
        
        if not self.episode_frame_cache:
            self.ax_img.set_title(f"Episode {self.current_episode_idx} (No frames or loading failed)", fontsize=12)
            self.img_display.set_data(np.zeros((224, 224, 3)))
            self.fig.canvas.draw_idle()
            return
            
        img_tensor = self.episode_frame_cache[frame_idx]
        img_to_show = np.clip(img_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)
        self.img_display.set_data(img_to_show)

        # 更新标题以显示 episode 状态
        status = self.annotations.get(self.current_episode_idx)
        if status is True:
            status_str = "CORRECT"
        elif status is False:
            status_str = "INCORRECT"
        else:
            status_str = "PENDING"
            
        title = (f"Ep {self.current_episode_idx + 1}/{self.num_episodes} | "
                 f"Frame {frame_idx + 1}/{len(self.episode_frame_cache)} | Status: {status_str}")
        self.ax_img.set_title(title, fontsize=12)
        self.fig.canvas.draw_idle()
        
    # REMOVED: `update_from_slider` 的功能合并到了 `update_display` 中

    # NEW: 核心标注逻辑
    def _annotate_episode(self, is_correct: bool):
        self.is_playing = False
        action_str = "Correct" if is_correct else "Incorrect"
        self.annotations[self.current_episode_idx] = is_correct
        
        print(f"  Episode {self.current_episode_idx} marked as {action_str}. Annotations auto-saved.")
        self._save_annotations_to_disk()
        
        # 自动切换到下一个 episode
        self.switch_episode(1)
        
    def mark_correct(self, event=None): self._annotate_episode(is_correct=True)
    def mark_incorrect(self, event=None): self._annotate_episode(is_correct=False)

    def switch_episode(self, direction: int):
        new_idx = self.current_episode_idx + direction
        if 0 <= new_idx < self.num_episodes:
            self.current_episode_idx = new_idx
            self.load_episode_data()
        elif direction > 0:
             print("\nAll episodes have been annotated! You are a hero! Please Save & Exit.")
             # 清空缓存并显示结束信息
             self.episode_frame_cache.clear()
             self.ax_img.set_title("All episodes reviewed!", fontsize=16)
             self.img_display.set_data(np.zeros((224, 224, 3)))
             self.fig.canvas.draw_idle()
        
    # MODIFIED: 调整播放逻辑以适应整个 episode
    def toggle_playback(self, event=None):
        if not self.episode_frame_cache: return
        if self.is_playing:
            self.is_playing = False
            self.btn_play.label.set_text('Play (▶)')
            return

        self.is_playing = True
        self.btn_play.label.set_text('Stop (■)')
        start_time = time.time()
        frame_count = 0
        
        # ... inside toggle_playback method
        while self.is_playing:
            current_val = int(self.frame_slider.val)
            next_val = current_val + 1

            # 如果下一帧会超出范围,则停止播放
            if next_val >= len(self.episode_frame_cache):
                self.is_playing = False
                # 可以在这里选择是否将滑块停在最后一帧
                self.frame_slider.set_val(current_val) 
            else:
                self.frame_slider.set_val(next_val)

            frame_count += 1
            next_frame_time = start_time + frame_count * self.playback_interval
            plt.pause(max(0.001, next_frame_time - time.time()))
            # The 'if not self.is_playing: break' is now naturally handled by the loop condition
        
        self.btn_play.label.set_text('Play (▶)')
        self.is_playing = False

    # MODIFIED: 调整快捷键
    def on_key_press(self, event):
        if event.key == "right": self.mark_correct()
        elif event.key == "left": self.mark_incorrect()
        elif event.key == 'pagedown': self.switch_episode(1)
        elif event.key == 'pageup': self.switch_episode(-1)
        elif event.key == ' ': self.toggle_playback()
        elif event.key == ',' or event.key == '<':
            self.is_playing = False
            new_val = self.frame_slider.val - 1
            if new_val >= 0: self.frame_slider.set_val(new_val)
        elif event.key == '.' or event.key == '>':
            self.is_playing = False
            new_val = self.frame_slider.val + 1
            if new_val <= self.frame_slider.valmax: self.frame_slider.set_val(new_val)

    # MODIFIED: 更新帮助信息
    def show(self):
        print("\n--- Episode Annotator ---")
        print("  - Review:   Use slider or press < , > to scrub through the episode.")
        print("  - Playback: Press [Space Bar] or the [Play] button.")
        print("  - Annotate: Press [←] to mark as INCORRECT, [→] to mark as CORRECT.")
        print("  - Navigate: [PageUp]/[PageDown] for manual episode switching.")
        print("  - Feature:  After annotating, it automatically loads the next episode.")
        print("  - Safety:   Annotations are saved automatically after every action.")
        print("---------------------------------------------")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual annotation tool for entire episodes.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID of the dataset.")
    parser.add_argument("--annotations_path", type=str, default="./episode_annotations.json", help="Path for auto-saving annotations.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        dataset = RewardLeRobotDataset(repo_id=args.repo_id)
        app = EpisodeAnnotatorApp(dataset, args.annotations_path)
        app.show()
    except Exception as e:
        import traceback
        print(f"\n--- ERROR ---")
        print(f"Failed to run the annotator: {e}")
        traceback.print_exc()