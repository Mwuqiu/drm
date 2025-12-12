import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
# Y轴显示的范围（度）
Y_LIMIT = 90 
# 是否只标注失败的 Episode？(设为 False 则标注所有)
LABEL_FAILURES_ONLY = False
# =======================================

def get_signed_angle_diff(angle_a, angle_b):
    diff = angle_a - angle_b
    diff = (diff + 180) % 360 - 180
    return diff

def main(log_dir, video_dir):
    # 1. 寻找数据
    npy_files = glob.glob(os.path.join(log_dir, "*_geom_euler.npy"))
    
    # 2. 解析状态
    status_map = {}
    pattern = re.compile(r"episode_(\d+)_.*_(success|failure)\.mp4")
    for f in glob.glob(os.path.join(video_dir, "*.mp4")):
        m = pattern.search(os.path.basename(f))
        if m: status_map[int(m.group(1))] = (m.group(2) == "success")

    print(f"Analyzing trajectory curves from {len(npy_files)} logs...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True) # 画布加宽一点
    titles = ["Roll Difference (X)", "Pitch Difference (Y)", "Yaw Difference (Z)"]

    success_count = 0
    fail_count = 0
    max_step = 0 # 用于记录最长的步数，方便设置坐标轴
    
    # 统计阈值用
    min_deviations = [[], [], []] 

    for filepath in npy_files:
        filename = os.path.basename(filepath)
        match = re.search(r"episode_(\d+)_geom_euler\.npy", filename)
        if not match: continue
        episode_idx = int(match.group(1))
        
        if episode_idx not in status_map: continue
        is_success = status_map[episode_idx]
        
        try:
            data = np.load(filepath)
            if data.ndim != 3 or data.shape[1] != 2: continue
            
            # 提取数据
            item_euler = data[:, 0, :]   
            target_euler = data[:, 1, :] 

            # 计算差值
            diffs = get_signed_angle_diff(item_euler, target_euler)
            steps = np.arange(len(diffs))
            max_step = max(max_step, len(diffs))

            # 设置样式
            if is_success:
                color = 'green'
                alpha = 0.4
                linewidth = 1.2
                linestyle = '-'
                success_count += 1
                
                # 标签样式 (成功)
                label_color = '#006400' # 深绿
                label_weight = 'normal'
                label_size = 7
                zorder = 5
                
                for i in range(3):
                    min_deviations[i].append(np.min(np.abs(diffs[:, i])))
            else:
                color = 'red'
                alpha = 0.3 
                linewidth = 1.0
                linestyle = '-' # 失败也用实线，看得更清楚
                fail_count += 1
                
                # 标签样式 (失败 - 更醒目)
                label_color = '#8b0000' # 深红
                label_weight = 'bold'
                label_size = 9
                zorder = 10

            # 绘制 3 个子图
            for i in range(3):
                ax = axes[i]
                val = diffs[:, i]
                
                # 画线
                ax.plot(steps, val, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
                
                # === [核心修改] 添加 Episode ID 标签 ===
                # 只在终点处标记
                if not LABEL_FAILURES_ONLY or (LABEL_FAILURES_ONLY and not is_success):
                    # 为了防止太密，可以在 Y 轴方向加一点点随机抖动，或者直接标
                    last_step = steps[-1]
                    last_val = val[-1]
                    
                    ax.text(last_step + 2, last_val, str(episode_idx), 
                            color=label_color, 
                            fontsize=label_size, 
                            fontweight=label_weight, 
                            va='center', # 垂直居中
                            zorder=zorder+1)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # --- 后处理与美化 ---
    suggested_thresholds = []

    threshes = [10, 15, 5]

    for i in range(3):
        ax = axes[i]
        
        thresh = threshes[i]


        suggested_thresholds.append(thresh)

        # 辅助线
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axhspan(-thresh, thresh, color='blue', alpha=0.05, label=f'Suggested Threshold (+/- {thresh:.1f}°)')
        ax.axhline(thresh, color='blue', linestyle=':', alpha=0.5)
        ax.axhline(-thresh, color='blue', linestyle=':', alpha=0.5)

        # 范围
        ax.set_ylim(-Y_LIMIT, Y_LIMIT)
        ax.set_ylabel("Angle Diff (Deg)")
        ax.set_title(f"{titles[i]} (Green=Success, Red=Failure)")
        ax.legend(loc='upper left') # 图例挪到左边，防止遮挡右边的标签
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # [关键] 扩展 X 轴右侧，给标签留出空间
        ax.set_xlim(0, max_step * 1.1) 

    axes[2].set_xlabel("Time Step (Simulation Steps)")
    plt.tight_layout()
    
    output_path = "orientation_curves_labeled.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved labeled curves to {output_path}")
    
    print("✅ 推荐配置:")
    print(f"   ROLL_TOLERANCE_DEG  = {suggested_thresholds[0]:.1f}")
    print(f"   PITCH_TOLERANCE_DEG = {suggested_thresholds[1]:.1f}")
    print(f"   YAW_TOLERANCE_DEG   = {suggested_thresholds[2]:.1f}")

if __name__ == "__main__":
    # --- 请确认路径 ---
    LOG_DIR = os.path.expanduser("~/drm/experiments/task_5_logs")
    VIDEO_DIR = os.path.expanduser("~/drm/rollouts/libero_elegant_1129/Full_Task_5")
    main(LOG_DIR, VIDEO_DIR)