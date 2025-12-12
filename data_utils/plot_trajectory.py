import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main(log_dir, video_dir):
    npy_files = glob.glob(os.path.join(log_dir, "*.npy"))
    
    status_map = {}
    pattern = re.compile(r"episode_(\d+)_.*_(success|failure)\.mp4")
    for f in glob.glob(os.path.join(video_dir, "*.mp4")):
        m = pattern.search(os.path.basename(f))
        if m: status_map[int(m.group(1))] = (m.group(2) == "success")

    success_data = []
    fail_data = []

    print(f"Analyzing {len(npy_files)} logs...")

    for filepath in npy_files:
        filename = os.path.basename(filepath)
        match = re.search(r"episode_(\d+)_geom_pos\.npy", filename)
        if not match: continue
        episode_idx = int(match.group(1))
        
        if episode_idx not in status_map: continue
        is_success = status_map[episode_idx]
        
        try:
            data = np.load(filepath)
            if data.ndim != 3 or data.shape[1] != 2 or data.shape[0] == 0: continue
            
            last_frame = data[-1]
            item_x, item_y = float(last_frame[0, 0]), float(last_frame[0, 1])
            cont_x, cont_y = float(last_frame[1, 0]), float(last_frame[1, 1])

            if abs(cont_x) < 1e-5 and abs(cont_y) < 1e-5: continue

            dx = item_x - cont_x
            dy = item_y - cont_y
            
            if is_success:
                success_data.append([dx, dy, episode_idx])
            else:
                fail_data.append([dx, dy, episode_idx])

        except Exception: pass

    if not success_data:
        print("没有找到成功数据，无法分析。")
        return
    
    s_arr = np.array(success_data)
    
    min_x, max_x = np.min(s_arr[:, 0]), np.max(s_arr[:, 0])
    min_y, max_y = np.min(s_arr[:, 1]), np.max(s_arr[:, 1])

    box_x_min, box_x_max = min_x , max_x 
    box_y_min, box_y_max = min_y , max_y 

    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.scatter(0, 0, c='black', marker='s', s=120, label='Container Center (0,0)', zorder=10)
    
    if fail_data:
        f_arr = np.array(fail_data)
        ax.scatter(f_arr[:, 0], f_arr[:, 1], c='#8b0000', marker='x', s=60, alpha=0.8, label='Failure Offset', zorder=6)
        for dx, dy, ep_id in f_arr:
            ax.text(dx, dy, str(int(ep_id)), color='#8b0000', fontsize=9, weight='bold', ha='left', va='bottom', zorder=7)

    ax.scatter(s_arr[:, 0], s_arr[:, 1], c='#006400', marker='o', s=50, alpha=0.6, label='Success Offset', zorder=4)
    for dx, dy, ep_id in s_arr:
        ax.text(dx, dy, str(int(ep_id)), color='#004d00', fontsize=8, ha='right', va='top', zorder=5, alpha=0.8)

    width = box_x_max - box_x_min
    height = box_y_max - box_y_min
    rect = patches.Rectangle((box_x_min, box_y_min), width, height, 
                             linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', zorder=8,
                             label='Calculated Success Zone')
    ax.add_patch(rect)

    print("\n" + "="*60)
    print("🎯 最终判定逻辑配置 (IsPreciselyOn):")
    print("   逻辑：只判断 (Item位置 - Container位置) 是否落在此区间内")
    print("="*60)
    print(f"SUCCESS_ZONES = {{")
    print(f"    \"target_object_name\": {{")
    print(f"        \"x_range\": ({box_x_min:.4f}, {box_x_max:.4f}),")
    print(f"        \"y_range\": ({box_y_min:.4f}, {box_y_max:.4f})")
    print(f"    }},")
    print(f"    \"default\": {{ \"x_range\": (-0.1, 0.1), \"y_range\": (-0.1, 0.1) }}")
    print(f"}}")
    print("="*60 + "\n")

    ax.set_title("Relative Position Analysis with Episode IDs (Centered at Container)")
    ax.set_xlabel("Relative X (Item - Container)")
    ax.set_ylabel("Relative Y (Item - Container)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    limit = max(abs(box_x_max), abs(box_x_min), abs(box_y_max), abs(box_y_min)) * 2.5
    limit = max(limit, 0.08) 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    output_path = "relative_zone_analysis_labeled.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved labeled plot to {output_path}")

if __name__ == "__main__":
    LOG_DIR = os.path.expanduser("~/drm/experiments/task_2_logs")
    VIDEO_DIR = os.path.expanduser("~/drm/rollouts/libero_elegant_1205/Full_Task")
    main(LOG_DIR, VIDEO_DIR)