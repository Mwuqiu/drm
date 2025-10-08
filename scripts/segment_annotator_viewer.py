import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox # <-- 1. Added TextBox import
import numpy as np
import torch
import logging
import argparse
import json
from pathlib import Path
import time
from reward_dataset import RewardLeRobotDataset

class FixedLengthAnnotatorApp:
    def __init__(self, dataset, annotations_path, segment_length=10, playback_fps=30):
        self.dataset = dataset
        self.annotations_path = Path(annotations_path)
        self.segment_length = segment_length
        self.playback_interval = 1.0 / playback_fps
        
        self.current_episode_idx = 0
        self.num_episodes = self.dataset.num_episodes

        self.annotations = self._load_annotations_from_disk()
        self.episode_frame_cache = []
        self.wrist_episode_frame_cache = [] 
        self.is_playing = False
        self.active_segment_index = None

        self.segment_widgets = []

        self.fig, (self.ax_img, self.ax_wrist_img) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.subplots_adjust(bottom=0.4, top=0.9)

        self.img_display = self.ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        self.ax_img.axis("off")
        self.ax_img.set_title("Main Camera")

        self.wrist_img_display = self.ax_wrist_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        self.ax_wrist_img.axis("off")
        self.ax_wrist_img.set_title("Wrist Camera")
        
        ax_preview_slider = plt.axes([0.15, 0.3, 0.7, 0.03])
        self.preview_slider = Slider(ax=ax_preview_slider, label='Preview', valmin=0, valmax=1, valinit=0, valstep=1)
        self.preview_slider.on_changed(self.update_image_display)

        # --- Button Layout Adjusted for New Widgets ---
        btn_width, btn_height, btn_y_pos = 0.1, 0.05, 0.02
        
        # Episode Navigation
        ax_prev_ep = plt.axes([0.02, btn_y_pos, btn_width, btn_height])
        self.btn_prev_ep = Button(ax_prev_ep, '<< Ep (PgUp)')
        self.btn_prev_ep.on_clicked(lambda e: self.switch_episode(-1))

        ax_next_ep = plt.axes([0.13, btn_y_pos, btn_width, btn_height])
        self.btn_next_ep = Button(ax_next_ep, 'Ep >> (PgDn)')
        self.btn_next_ep.on_clicked(lambda e: self.switch_episode(1))

        # --- 2. Added Jump to Episode Widgets ---
        ax_ep_text = plt.axes([0.25, btn_y_pos, 0.1, btn_height])
        self.ep_textbox = TextBox(ax_ep_text, "Go to Ep #:", initial="1", textalignment="center")
        self.ep_textbox.on_submit(self.on_jump_submit)

        ax_jump_ep = plt.axes([0.36, btn_y_pos, 0.05, btn_height])
        self.btn_jump_ep = Button(ax_jump_ep, 'Go')
        self.btn_jump_ep.on_clicked(self.on_jump_submit)
        # --- End of New Widgets ---

        ax_view_full = plt.axes([0.43, btn_y_pos, 0.14, btn_height])
        self.btn_view_full = Button(ax_view_full, 'View Full Episode')
        self.btn_view_full.on_clicked(self._view_full_episode)
        
        ax_add_segment = plt.axes([0.59, btn_y_pos, 0.14, 0.075])
        self.btn_add_segment = Button(ax_add_segment, 'Add Segment', color='lightgreen')
        self.btn_add_segment.on_clicked(self.add_new_segment_ui)

        ax_save = plt.axes([0.88, btn_y_pos, 0.1, btn_height])
        self.btn_save = Button(ax_save, 'Save & Exit')
        self.btn_save.on_clicked(self.save_and_exit)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.load_episode_data()

    # --- 3. New method to handle jumping to a specific episode ---
    def on_jump_submit(self, event):
        """Callback for the Go button and TextBox to jump to an episode."""
        try:
            # User will input 1-based number, convert to 0-based index
            target_ep_num = int(self.ep_textbox.text)
            target_ep_idx = target_ep_num - 1

            if 0 <= target_ep_idx < self.num_episodes:
                self.jump_to_episode(target_ep_idx)
            else:
                print(f"Invalid episode number: {target_ep_num}. Please enter a number between 1 and {self.num_episodes}.")
                self.ep_textbox.set_val("") # Clear invalid input
        except ValueError:
            print(f"Invalid input: '{self.ep_textbox.text}'. Please enter a number.")
            self.ep_textbox.set_val("") # Clear invalid input

    def jump_to_episode(self, target_idx: int):
        """Jumps the view to a specific episode index."""
        if self.current_episode_idx == target_idx:
            return # No need to reload if we are already there
            
        self._save_annotations_to_disk()
        self.current_episode_idx = target_idx
        self.load_episode_data()
    # --- End of new methods ---

    def _clear_segment_widgets(self):
        for widgets in self.segment_widgets:
            widgets['slider_ax'].remove()
            widgets['view_button_ax'].remove()
            widgets['delete_button_ax'].remove()
        self.segment_widgets.clear()
        self.fig.canvas.draw_idle()

    def _render_segment_widgets(self):
        self._clear_segment_widgets()
        segments = self.annotations.get(self.current_episode_idx, [])
        for i, (start, _) in enumerate(segments):
            self._create_segment_slider_ui(i, start_frame=start)
        self.update_title()

    def _create_segment_slider_ui(self, index, start_frame):
        num_existing = len(self.segment_widgets)
        y_pos = 0.25 - (num_existing * 0.05)

        ep_len = len(self.episode_frame_cache)
        slider_max = max(0, ep_len - self.segment_length)
        
        slider_ax = plt.axes([0.15, y_pos, 0.55, 0.03])
        pos_slider = Slider(slider_ax, f'Seg {index} Start', 0, slider_max, valinit=start_frame, valstep=1)
        
        view_ax = plt.axes([0.74, y_pos, 0.08, 0.03])
        view_button = Button(view_ax, 'View')
        
        delete_ax = plt.axes([0.84, y_pos, 0.04, 0.03])
        delete_button = Button(delete_ax, 'X', color='salmon')

        pos_slider.on_changed(lambda val, idx=index: self._on_position_change(val, idx))
        view_button.on_clicked(lambda event, idx=index: self._view_segment(idx))
        delete_button.on_clicked(lambda event, idx=index: self._delete_segment_ui(idx))
        
        self.segment_widgets.append({
            'slider': pos_slider, 'slider_ax': slider_ax,
            'view_button': view_button, 'view_button_ax': view_ax,
            'delete_button': delete_button, 'delete_button_ax': delete_ax
        })

    def _on_position_change(self, start_frame_float, index):
        start = int(start_frame_float)
        end = start + self.segment_length - 1
        self.annotations[self.current_episode_idx][index] = [start, end]
        if self.active_segment_index == index:
            self._view_segment(index, preserve_val=True)
        self.update_title()
        self._save_annotations_to_disk()

    def _delete_segment_ui(self, index):
        self.annotations[self.current_episode_idx].pop(index)
        if self.active_segment_index == index:
            self._view_full_episode()
        elif self.active_segment_index is not None and self.active_segment_index > index:
             self.active_segment_index -= 1
        self._save_annotations_to_disk()
        self._render_segment_widgets()

    def _view_segment(self, index, preserve_val=False):
        self.active_segment_index = index
        start, end = self.annotations.get(self.current_episode_idx)[index]
        
        current_val = self.preview_slider.val
        self.preview_slider.ax.set_xlim(start, end)
        
        if preserve_val and start <= current_val <= end:
             self.preview_slider.set_val(current_val)
        else:
             self.preview_slider.set_val(start)
        
        self.update_title()

    def _view_full_episode(self, event=None):
        self.active_segment_index = None
        ep_len = len(self.episode_frame_cache)
        self.preview_slider.ax.set_xlim(0, max(1, ep_len - 1))
        self.update_title()

    def add_new_segment_ui(self, event=None):
        if self.current_episode_idx not in self.annotations:
            self.annotations[self.current_episode_idx] = []
        
        current_frame = int(self.preview_slider.val)
        max_start_frame = len(self.episode_frame_cache) - self.segment_length
        start = min(current_frame, max_start_frame)
        end = start + self.segment_length - 1
        self.annotations[self.current_episode_idx].append([start, end])
        
        self._save_annotations_to_disk()
        self._render_segment_widgets()

    def load_episode_data(self):
        self.is_playing = False
        print(f"\nLoading Episode {self.current_episode_idx}...")
        
        self.episode_frame_cache.clear()
        self.wrist_episode_frame_cache.clear()
        
        ep_start = self.dataset.episode_data_index["from"][self.current_episode_idx].item()
        ep_end = self.dataset.episode_data_index["to"][self.current_episode_idx].item()
        
        for i in range(ep_start, ep_end):
            complete_frame = self.dataset[i]
            self.episode_frame_cache.append(complete_frame['observation.images.image'])
            self.wrist_episode_frame_cache.append(complete_frame['observation.images.wrist_image'])

        ep_len = len(self.episode_frame_cache)
        self.preview_slider.valmax = max(1, ep_len - 1)
        self.preview_slider.set_val(0)
        
        # After loading new data, update the textbox to the current episode number
        self.ep_textbox.set_val(str(self.current_episode_idx + 1))
        
        self._view_full_episode()
        self._render_segment_widgets()
        self.update_image_display(0)

    def update_title(self):
        frame_idx = int(self.preview_slider.val)
        mode = f"Segment {self.active_segment_index}" if self.active_segment_index is not None else "Full Episode"
        
        title = (f"Ep {self.current_episode_idx + 1}/{self.num_episodes} | Frame {frame_idx + 1}/{len(self.episode_frame_cache)} "
                 f"| View Mode: {mode}")
        
        segments = self.annotations.get(self.current_episode_idx, [])
        if segments:
            title += f"\nSegments: {segments}"
        
        self.fig.suptitle(title, fontsize=10)
        self.fig.canvas.draw_idle()

    def update_image_display(self, frame_idx_float):
        frame_idx = int(frame_idx_float)
        
        if not self.episode_frame_cache or frame_idx >= len(self.episode_frame_cache):
            self.img_display.set_data(np.zeros((224, 224, 3)))
        else:
            img_tensor = self.episode_frame_cache[frame_idx]
            img_to_show = np.clip(img_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)
            self.img_display.set_data(img_to_show)
            
        if not self.wrist_episode_frame_cache or frame_idx >= len(self.wrist_episode_frame_cache):
            self.wrist_img_display.set_data(np.zeros((224, 224, 3)))
        else:
            wrist_img_tensor = self.wrist_episode_frame_cache[frame_idx]
            wrist_img_to_show = np.clip(wrist_img_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)
            self.wrist_img_display.set_data(wrist_img_to_show)

        self.update_title()

    def switch_episode(self, direction: int):
        self._save_annotations_to_disk()
        new_idx = self.current_episode_idx + direction
        if 0 <= new_idx < self.num_episodes:
            self.current_episode_idx = new_idx
            self.load_episode_data()

    def _load_annotations_from_disk(self):
        if self.annotations_path.exists():
            with self.annotations_path.open('r') as f:
                try: return {int(k): v for k, v in json.load(f).items()}
                except json.JSONDecodeError: return {}
        return {}

    def _save_annotations_to_disk(self):
        try:
            with self.annotations_path.open('w') as f:
                if self.current_episode_idx in self.annotations:
                    self.annotations[self.current_episode_idx] = sorted(self.annotations[self.current_episode_idx])
                json.dump(dict(sorted(self.annotations.items())), f, indent=4)
        except Exception as e:
            print(f"\n--- ERROR SAVING: {e} ---")

    def save_and_exit(self, event=None):
        self._save_annotations_to_disk()
        plt.close(self.fig)

    def on_key_press(self, event):
        if event.key == 'pagedown':
            self.switch_episode(1)
            return
        if event.key == 'pageup':
            self.switch_episode(-1)
            return

        if event.key == 'left' or event.key == 'right':
            if self.active_segment_index is not None:
                try:
                    bounds = self.annotations[self.current_episode_idx][self.active_segment_index]
                    min_val, max_val = bounds[0], bounds[1]
                except (KeyError, IndexError):
                    min_val, max_val = self.preview_slider.valmin, self.preview_slider.valmax
            else:
                min_val, max_val = self.preview_slider.valmin, self.preview_slider.valmax

            current_val = self.preview_slider.val
            if event.key == 'left':
                new_val = max(min_val, current_val - 1)
            else:
                new_val = min(max_val, current_val + 1)
            
            self.preview_slider.set_val(new_val)

    def show(self):
        print("\n--- Fixed-Length Segment Annotator (v2) ---")
        print(f"  Segment Length is fixed to: {self.segment_length} frames")
        print("  - Add:      Click [Add Segment] to create a new reward segment.")
        print("  - Position: Drag the 'Seg X Start' slider to position the segment.")
        print("  - Inspect:  Click [View] to lock the top Preview slider to that segment.")
        print("  - Reset:    Click [View Full Episode] to unlock the Preview slider.")
        print("  - Navigate: [PageUp]/[PageDown] for episodes, [<-]/[->] for frames.")
        print("  - Jump:     Type episode number in 'Go to Ep #:' box and press Enter or click [Go].")
        print("----------------------------------------------------------")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An interactive tool to annotate fixed-length reward segments.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID of the dataset.")
    parser.add_argument("--annotations_path", type=str, default="task_annotations/libero_elegant/task_3_segment_annotations.json", help="Path for auto-saving segment annotations.")
    parser.add_argument("--segment_length", type=int, default=25, help="The fixed length of each reward segment.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        dataset = RewardLeRobotDataset(repo_id=args.repo_id, video_backend='pyav')
        app = FixedLengthAnnotatorApp(dataset, args.annotations_path, segment_length=args.segment_length)
        app.show()
    except Exception as e:
        import traceback
        print(f"\n--- ERROR ---")
        print(f"Failed to run the annotator: {e}")
        traceback.print_exc()