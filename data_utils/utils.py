import logging
import os
import sys
import shutil
import collections

import imageio
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from libero.libero.envs.predicates.base_predicates import (
    PositionUnchanged, 
    PitchRollStable, 
    IsPreciselyOn, 
    IsOrientationAligned
)

def save_q_range_analysis(
    default_q_values,
    max_q_values,
    min_q_values,
    avg_q_values,
    metric_values,
    interventions,
    threshold,
    task_id,
    episode_idx,
    task_name,
    save_dir,
    outcome,
    intervention_snapshots=None
):
    plt.switch_backend('Agg')
    os.makedirs(save_dir, exist_ok=True)

    if intervention_snapshots and len(intervention_snapshots) > 0:
        real_time_steps = sorted(list(intervention_snapshots.keys()))
    else:
        real_time_steps = [i * 10 for i in range(len(default_q_values))]

    data_len = min(len(default_q_values), len(real_time_steps))
    real_time_steps = real_time_steps[:data_len]
    logical_indices = np.arange(data_len) 

    default_q = default_q_values[:data_len]
    max_q = max_q_values[:data_len]
    metric_val = metric_values[:data_len]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 2, 2], hspace=0.15)
    
    ax_vis = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax_vis)
    ax2 = plt.subplot(gs[2], sharex=ax_vis)

    ax_vis.set_xlim(-0.5, data_len - 0.5)
    ax_vis.set_ylim(0, 1)
    ax_vis.axis('off')
    ax_vis.set_title(f"Episode Analysis: {task_name} (Ep {episode_idx})", fontsize=16, y=1.05, fontweight='bold')

    y_center = 0.55
    y_text = 0.15

    if intervention_snapshots:
        for i, t in zip(logical_indices, real_time_steps):
            if t in intervention_snapshots:
                img = intervention_snapshots[t]
                
                ax_vis.vlines(x=i, ymin=y_text, ymax=y_center, 
                             color='red', linestyle='-', linewidth=1, alpha=0.3)
                
                imagebox = OffsetImage(img, zoom=0.25) 
                ab = AnnotationBbox(imagebox, (i, y_center), 
                                    xycoords='data', 
                                    boxcoords="offset points", 
                                    pad=0.2,
                                    box_alignment=(0.5, 0.5), 
                                    bboxprops=dict(edgecolor='red', linewidth=1.5))
                ax_vis.add_artist(ab)
                
                ax_vis.text(i, y_text, f"T={t}", 
                           ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='darkred',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
                ax_vis.axvline(x=i, color='gray', linestyle=':', alpha=0.2)

    if min_q_values is not None:
        ax1.fill_between(logical_indices, min_q_values[:data_len], max_q, color='gray', alpha=0.15)
        
    ax1.plot(logical_indices, default_q, label='Default Action Q', color='royalblue', marker='o', markersize=4)
    ax1.plot(logical_indices, max_q, label='Intervention Candidate Q', color='seagreen', linestyle='--', marker='x', markersize=4)
    
    intervention_bools = interventions[:data_len]
    intervention_locs = [i for i, x in enumerate(intervention_bools) if x]
    
    if intervention_locs:
        interv_y = [max_q[i] for i in intervention_locs]
        ax1.scatter(intervention_locs, interv_y, color='red', marker='*', s=150, zorder=5, label='Intervention Triggered')
        for loc in intervention_locs:
             ax1.axvline(x=loc, color='red', alpha=0.3, linestyle='--')
             ax2.axvline(x=loc, color='red', alpha=0.3, linestyle='--')

    ax1.set_ylabel("Q-Value", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(logical_indices, metric_val, label='Advantage', color='darkorange', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax2.fill_between(logical_indices, metric_val, threshold, where=(np.array(metric_val) > threshold), color='red', alpha=0.3)
    
    ax2.set_ylabel("Advantage", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time Step (Simulation Steps)", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax2.set_xticks(logical_indices)
    ax2.set_xticklabels([str(t) for t in real_time_steps], rotation=45)
    plt.xlim(-0.5, data_len - 0.5)

    save_path = os.path.join(save_dir, f"intervention_analysis_{task_id}_{episode_idx}_{outcome}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved analysis to {save_path}")

def get_action_via_volatility(
    policy,
    observation,
    default_sequence,
    q_history,
    args,
    t,
    device="cuda"
):
    stats = {}

    default_action_chunk = default_sequence[:, :args.vqh_chunk_size, :]
    default_q_value_tensor = policy.evaluate_q_values(observation, default_action_chunk.unsqueeze(1))
    current_q = default_q_value_tensor.item()
    
    q_history.append(current_q)
    sma_q = np.mean(list(q_history))
    q_change = abs(current_q - sma_q)
    
    stats['current_q'] = current_q
    stats['q_change'] = q_change
    
    is_critical_action = (len(q_history) >= args.q_history_window_size) and \
                         (q_change > args.q_change_threshold)
    
    start_msgs = ['\n']
    start_msgs.append("=" * 70)
    if t is not None:
        start_msgs.append(f"- Time Step Now - : {t}")
    logging.info("\n".join(start_msgs))


    logging.info(f"[Strategy: Volatility] Step {t}: Q={current_q:.4f}, SMA={sma_q:.4f}, |Delta|={q_change:.4f}, Threshold={args.q_change_threshold}")
    
    should_intervene = (is_critical_action or args.full_intervention_mode) and args.num_candidate_actions > 1
    stats['intervened'] = should_intervene

    final_action_sequence = None

    if should_intervene:
        logging.info(f"Step {t}: Intervention Triggered (Volatility).")

        num_candidates = args.num_candidate_actions
        candidate_sequences, q_values = policy.sample_and_evaluate_actions(observation, num_candidates)

        stats['intervention_candidates'] = candidate_sequences
        stats['intervention_q_values'] = q_values

        best_candidate_idx = choose_best_action(
            q_values=q_values,
        )

        # best_candidate_idx = choose_best_action_hybrid(
        #     q_values=q_values,
        #     candidate_sequences=candidate_sequences,
        #     top_k=args.num_candidate_actions,
        #     cluster_threshold=0.2,
        #     q_gap_threshold=0.15
        # )
        
        best_action_sequence = candidate_sequences[0, best_candidate_idx]
        final_action_sequence = best_action_sequence[:args.g_action_steps]
        
        base_q_values = q_values[0]
        best_base_idx_in_group = torch.argmax(base_q_values).item()
        best_base_action_for_log = candidate_sequences[0, best_base_idx_in_group][:args.g_action_steps]
        
        log_evaluation_results(
            q_values=q_values,
            best_idx=best_candidate_idx,
            best_sequence=final_action_sequence,
            base_q_values=base_q_values,
            best_base_action=best_base_action_for_log,
            time_step=t,
            base_action_sequences=candidate_sequences.squeeze(0),
        )        
    else:
        best_action_sequence = default_sequence[0]
        final_action_sequence = best_action_sequence[:args.n_action_steps]
        
        log_default_action(
            sequence=final_action_sequence,
            time_step=t
        )
    return final_action_sequence, stats

def get_action_via_advantage(
    policy,
    observation,
    default_sequence,
    args,
    t,
    advantage_threshold=0.1,
    num_probe_actions=8, 
    device="cuda"
):
    stats = {}
    
    default_action_chunk = default_sequence[:, :args.vqh_chunk_size, :]
    default_q_value_tensor = policy.evaluate_q_values(observation, default_action_chunk.unsqueeze(1))
    current_q = default_q_value_tensor.item()
    
    stats['current_q'] = current_q

    candidate_sequences, q_values = policy.sample_and_evaluate_actions(observation, num_probe_actions)

    current_q = default_q_value_tensor.item()
    max_q = torch.max(q_values).item()
    min_q = torch.min(q_values).item()
    avg_q = torch.mean(q_values).item()
    advantage = max_q - min_q    
    # advantage = max_q - current_q    

    stats['current_q'] = current_q
    stats['max_q'] = max_q
    stats['min_q'] = min_q
    stats['avg_q'] = avg_q
    stats['advantage'] = advantage

    max_candidate_q = torch.max(q_values).item()

    advantage = max_candidate_q - current_q
    stats['advantage'] = advantage
    stats['q_change'] = advantage
    
    should_intervene = (advantage > advantage_threshold) or args.full_intervention_mode
    
    start_msgs = ['\n']
    start_msgs.append("=" * 70)
    if t is not None:
        start_msgs.append(f"- Time Step Now - : {t}")
    logging.info("\n".join(start_msgs))

    logging.info(f"[Strategy: Advantage] Step {t}: Q_def={current_q:.4f}, Q_max={max_candidate_q:.4f}, Adv={advantage:.4f}, Threshold={advantage_threshold}")
    
    stats['intervened'] = should_intervene
    
    final_action_sequence = None

    if should_intervene:
        logging.info(f"Step {t}: Intervention Triggered (Advantage > {advantage_threshold}).")

        stats['intervention_candidates'] = candidate_sequences
        stats['intervention_q_values'] = q_values

        # best_candidate_idx = choose_best_action(
        #     q_values=q_values,
        # )
        
        # best_candidate_idx = choose_median_action(
        #     q_values=q_values,
        # ) 

        best_candidate_idx = choose_worst_action(
            q_values=q_values,
        )        

        # best_candidate_idx = choose_best_action_hybrid(
        #     q_values=q_values,
        #     candidate_sequences=candidate_sequences,
        #     top_k=num_probe_actions,
        #     cluster_threshold=0.2,
        #     q_gap_threshold=0.15
        # )
        
        best_action_sequence = candidate_sequences[0, best_candidate_idx]
        final_action_sequence = best_action_sequence[:args.g_action_steps]
        
        base_q_values = q_values[0]
        best_base_idx_in_group = torch.argmax(base_q_values).item()
        best_base_action_for_log = candidate_sequences[0, best_base_idx_in_group][:args.g_action_steps]

        log_evaluation_results(
            q_values=q_values,
            best_idx=best_candidate_idx,
            best_sequence=final_action_sequence,
            base_q_values=base_q_values,
            best_base_action=best_base_action_for_log,
            time_step=t,
            base_action_sequences=candidate_sequences.squeeze(0),
        )        
    else:
        logging.info("Advantage too small. Keeping default action.")
        best_action_sequence = default_sequence[0]
        final_action_sequence = best_action_sequence[:args.n_action_steps]
        
        log_default_action(
            sequence=final_action_sequence,
            time_step=t
        )        
    return final_action_sequence, stats

def save_q_value_curve(q_values, metric_values, interventions, threshold, task_id, episode_idx, task_name, save_dir, outcome, metric_name="Q-Delta"):
    plt.switch_backend('Agg') 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    steps = np.arange(len(q_values))
    
    ax1.plot(steps, q_values, label='Current Action Q-Value', color='royalblue', linewidth=1.5)
    
    intervention_indices = [i for i, x in enumerate(interventions) if x]
    if intervention_indices:
        intervention_q = [q_values[i] for i in intervention_indices]
        ax1.scatter(intervention_indices, intervention_q, color='red', marker='*', s=100, label='Intervention Triggered', zorder=5)
    
    ax1.set_title(f"Task {task_id} Ep {episode_idx}: {task_name}\nOutcome: {outcome.upper()}", fontsize=12)
    ax1.set_ylabel("Q-Value")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    plot_metrics = metric_values[:len(steps)]
    line_color = 'orange' if "Delta" in metric_name else 'seagreen'
    
    ax2.plot(steps, plot_metrics, label=metric_name, color=line_color, linewidth=1.5)
    
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')
    ax2.fill_between(steps, plot_metrics, threshold, where=(np.array(plot_metrics) > threshold), 
                     color='red', alpha=0.3, interpolate=True)

    ax2.set_ylabel(metric_name) # 动态设置 Y 轴标签
    ax2.set_xlabel("Step")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # --- 保存 ---
    safe_task_name = task_name.replace(" ", "_").replace("/", "_")
    # 文件名也加上 metric_name 以防混淆
    metric_slug = "adv" if "Advantage" in metric_name else "vol"
    file_name = f"q_analysis_{metric_slug}_task_{task_id}_ep_{episode_idx}_{outcome}.png"
    save_path = os.path.join(save_dir, file_name)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # logging.info(f"Saved Q-curve plot to {save_path}")

def setup_task_predicates(task_id, episode_idx, log_root="./experiments", enable_logging=False):

    PositionUnchanged.start_new_episode(episode_idx)
    PitchRollStable.start_new_episode(episode_idx)

    if enable_logging:
        current_save_dir = os.path.join(log_root, f"task_{task_id}_logs")
    else:
        current_save_dir = None

    if task_id in [2, 3]:
        IsPreciselyOn.start_new_episode(episode_idx, save_dir=current_save_dir)

    elif task_id == 4:
        IsOrientationAligned.start_new_episode(
            episode_idx, 
            constraints={
                "max_roll_deviation": 10.0, 
                "tolerances": (5.0, 15.0, 5.0)
            },
            save_dir=current_save_dir
        )

    elif task_id == 5:
        IsOrientationAligned.start_new_episode(
            episode_idx, 
            constraints={
                "tolerances": (5.0, 15.0, 10.0)
            },
            save_dir=current_save_dir
        )

def setup_global_logging():
    """
    Sets up global logging to console only.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logging.info("Global logging initialized (console only).")

def setup_episode_logging(task_id: int, episode_idx: int, base_dir: str):
    """
    Sets up per-episode file logging. Creates a file for each episode in base_dir/episode_logs/.
    Removes any previous file handlers and adds a new one for this episode.
    """
    episode_log_dir = os.path.join(base_dir, "episode_logs")
    os.makedirs(episode_log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"task_{task_id}_episode_{episode_idx}_{timestamp}.log"
    log_filepath = os.path.join(episode_log_dir, log_filename)

    logger = logging.getLogger()
    # Remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Record DEBUG level to episode file
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logging.info(f"Episode logging initialized for task {task_id}, episode {episode_idx}. Output saved to: {log_filepath}")

def log_evaluation_results(
    q_values: torch.Tensor,
    best_idx: int,
    best_sequence: torch.Tensor,
    base_q_values: torch.Tensor = None,
    best_base_action: torch.Tensor = None,
    time_step: int = None,
    base_action_sequences: torch.Tensor = None,
):
    """Logs the detailed results of a VQH critical action evaluation step."""
    q_values_np = q_values.cpu().numpy().flatten()

    summary_msgs = ["\n"]
    # summary_msgs.append("=" * 70)
    # if time_step is not None:
    #     summary_msgs.append(f"- Time Step Now - : {time_step}")
    summary_msgs.append("[VQH Q-Value Evaluation]")
    if base_q_values is not None and base_q_values.numel() > 0:
        base_q_str = ", ".join([f"{q:.4f}" for q in base_q_values.cpu().numpy().flatten()])
        summary_msgs.append(f"  - Base Policy Q-Values:  [{base_q_str}]")
    summary_msgs.append("  - Summary:")
    summary_msgs.append(f"    - Winning Candidate Idx: {best_idx}")
    summary_msgs.append(f"    - Winning Q-Value:       {q_values_np[best_idx]:.4f}")
    summary_msgs.append(f"    - Average Q-Value:       {np.mean(q_values_np):.4f}")
    logging.debug("\n".join(summary_msgs))

    details_msgs = ["-" * 70]
    details_msgs.append("[All Candidate Actions]")
    if base_action_sequences is not None and base_action_sequences.numel() > 0:
        details_msgs.append("  - Base Policy Actions:")
        for i, sequence in enumerate(base_action_sequences):
            q_value = q_values_np[i]
            details_msgs.append(f"    - Candidate {i} (Q={q_value:.4f}):")
            sequence_np = sequence.cpu().numpy()
            for step_idx, step_action in enumerate(sequence_np):
                step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
                details_msgs.append(f"      - Step {step_idx+1}: [{step_str}]")
    else:
        details_msgs.append("  - No candidate sequences were provided for detailed logging.")
    logging.debug("\n".join(details_msgs))

    decision_msgs = ["-" * 70]
    decision_msgs.append("[Decision]")
    decision_msgs.append(f"  - ✅ VQH selected action from Candidate {best_idx}.")
    decision_msgs.append("-" * 70)
    decision_msgs.append("[Overall Winning Action Sequence]")
    best_sequence_np = best_sequence.cpu().numpy()
    for i, step_action in enumerate(best_sequence_np):
        step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
        decision_msgs.append(f"  - Step {i+1}: [{step_str}]")

    decision_msgs.append("=" * 70)
    logging.debug("\n".join(decision_msgs))

def log_default_action(sequence: torch.Tensor, time_step: int):
    """Logs the default action sequence when it's used as a fallback."""
    log_msgs = ["\n"]
    # log_msgs.append("=" * 70)
    # log_msgs.append(f"- Time Step Now - : {time_step}")
    log_msgs.append("[Decision]")
    log_msgs.append("  - ⚙️ Default action sequence was chosen (non-critical step).")
    log_msgs.append("-" * 70)
    log_msgs.append("[Default Action Executed]")
    
    sequence_np = sequence.cpu().numpy()
    for i, step_action in enumerate(sequence_np):
        step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
        log_msgs.append(f"  - Step {i+1}: [{step_str}]")
    
    log_msgs.append("=" * 70)
    logging.debug("\n".join(log_msgs))

def init_dataset_if_needed(args):
    if not args.save_dataset:
        return None, None

    logging.info(f"Preparing to save dataset to: {args.dataset_repo_id}")
    dataset_path = HF_LEROBOT_HOME / args.dataset_repo_id
    
    if dataset_path.exists():
        logging.info("Dataset found locally. Loading existing dataset to append data.")
        replay_dataset = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            download_videos=True,
        )
    else:
        logging.info("No local dataset found. Creating a new one.")
        dataset_features = {
            "observation.images.image": {"shape": (256, 256, 3), "dtype": "image"},
            "observation.images.wrist_image": {"shape": (256, 256, 3), "dtype": "image"},
            "observation.state": {"shape": (8,), "dtype": "float32"},
            "action": {"shape": (7,), "dtype": "float32"},
        }
        replay_dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            features=dataset_features,
            fps=10,
            use_videos=True,
        )
    
    temp_img_dir = replay_dataset.root / "temp_images"
    return replay_dataset, temp_img_dir

def save_episode_to_dataset(replay_dataset, episode_buffer, task_description, temp_img_dir):
    if replay_dataset is None or not episode_buffer:
        return

    logging.info(f"Saving episode to dataset: {replay_dataset.repo_id}")
    
    episode_data_dict = collections.defaultdict(list)
    for step in episode_buffer:
        for key, value in step.items():
            episode_data_dict[key].append(value)
    
    episode_length = len(episode_buffer)
    episode_data_dict["frame_index"] = list(range(episode_length))
    fps = replay_dataset.fps
    episode_data_dict["timestamp"] = [i / fps for i in range(episode_length)]
    start_index = replay_dataset.meta.total_frames
    episode_data_dict["index"] = list(range(start_index, start_index + episode_length))
    
    task_index = replay_dataset.meta.get_task_index(task_description)
    if task_index is None:
        replay_dataset.meta.add_task(task_description)
        task_index = replay_dataset.meta.get_task_index(task_description)
    episode_data_dict["task_index"] = [task_index] * episode_length
    
    episode_index_to_save = replay_dataset.meta.total_episodes
    episode_data_dict["episode_index"] = episode_index_to_save
    episode_data_dict["size"] = episode_length
    episode_data_dict["task"] = [task_description] * episode_length
    
    replay_dataset.save_episode(episode_data=episode_data_dict)

    for key in ["observation.images.image", "observation.images.wrist_image"]:
        saved_episode_img_dir = temp_img_dir / key / f"episode_{episode_index_to_save:06d}"
        if saved_episode_img_dir.is_dir():
            shutil.rmtree(saved_episode_img_dir)

def choose_best_action_hybrid(
    q_values: torch.Tensor,
    candidate_sequences: torch.Tensor,
    top_k: int = 8,
    cluster_threshold: float = 0.2,
    q_gap_threshold: float = 0.05
) -> int:
    logging.info("--- Starting Hybrid Action Selection (Q-Value First, Consensus Tie-breaker) ---")

    num_candidates = q_values.shape[1]
    if num_candidates < 2:
        logging.info("Not enough candidates, returning argmax.")
        return torch.argmax(q_values, dim=1).item()

    actual_k = min(top_k, num_candidates)
    sorted_q_values, sorted_indices = torch.sort(q_values, dim=1, descending=True)
    top_k_indices = sorted_indices[0, :actual_k]
    top_k_q_values = sorted_q_values[0, :actual_k]
    first_step_actions = candidate_sequences[0, :, 0, :]
    top_k_gripper_actions = first_step_actions[top_k_indices, -1].cpu().numpy()

    clusters = []
    for i in range(actual_k):
        current_gripper = top_k_gripper_actions[i]
        item_info = {
            'original_idx': top_k_indices[i].item(),
            'q_value': top_k_q_values[i].item(),
            'gripper_val': current_gripper
        }
        found_cluster = False
        for cluster in clusters:
            representative_gripper = cluster[0]['gripper_val']
            if abs(current_gripper - representative_gripper) <= cluster_threshold:
                cluster.append(item_info)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([item_info])
            
    if not clusters:
        logging.warning("Clustering failed, returning default argmax.")
        return torch.argmax(q_values, dim=1).item()

    cluster_summaries = []
    for i, cluster in enumerate(clusters):
        if not cluster: continue
        best_candidate_in_cluster = max(cluster, key=lambda x: x['q_value'])
        cluster_summaries.append({
            'cluster_id': i,
            'size': len(cluster),
            'max_q_value': best_candidate_in_cluster['q_value'],
            'best_candidate_info': best_candidate_in_cluster
        })
        
    cluster_summaries.sort(key=lambda x: x['max_q_value'], reverse=True)
    logging.info("\nCluster Analysis (sorted by Q-value):")
    for summary in cluster_summaries:
        logging.info(f"  - Cluster {summary['cluster_id']} | Size: {summary['size']} | Max Q: {summary['max_q_value']:.4f} (Candidate {summary['best_candidate_info']['original_idx']})")
    champion_cluster = cluster_summaries[0]
    default_winner_info = champion_cluster['best_candidate_info']
    
    final_winner_info = default_winner_info
    
    if len(cluster_summaries) > 1:
        challenger_cluster = cluster_summaries[1]
        
        q_gap = champion_cluster['max_q_value'] - challenger_cluster['max_q_value']
        
        logging.info(f"\nDecision Logic:")
        logging.info(f"  - Champion Cluster's Max Q: {champion_cluster['max_q_value']:.4f}")
        logging.info(f"  - Challenger Cluster's Max Q: {challenger_cluster['max_q_value']:.4f}")
        logging.info(f"  - Q-Value Gap: {q_gap:.4f} (Threshold: {q_gap_threshold})")

        if q_gap <= q_gap_threshold:
            logging.info(f"  - Condition MET: Q-gap is within threshold.")
            logging.info(f"  - Champion Size: {champion_cluster['size']}, Challenger Size: {challenger_cluster['size']}")
            if challenger_cluster['size'] > champion_cluster['size']:
                logging.info(f"  - Condition MET: Challenger cluster is larger. Switching to challenger.")
                final_winner_info = challenger_cluster['best_candidate_info']
            else:
                logging.info(f"  - Condition FAILED: Challenger cluster is NOT larger. Sticking with champion.")
        else:
            logging.info(f"  - Condition FAILED: Q-gap is too large. Sticking with champion.")

    logging.info("\n--- Final Decision ---")
    logging.info(f"  - Winning Candidate Index: {final_winner_info['original_idx']}")
    logging.info(f"  - Winning Q-Value: {final_winner_info['q_value']:.4f}")
    return final_winner_info['original_idx']

def choose_best_action(
    q_values: torch.Tensor,
) -> int:
    best_idx = torch.argmax(q_values).item()
    max_q = q_values[0, best_idx].item()
    logging.info(f"Greedy Selection: Chosen Candidate {best_idx} | Q-Value: {max_q:.4f}")
    return best_idx

def choose_worst_action(
    q_values: torch.Tensor,
) -> int:
    min_idx = torch.argmin(q_values).item()
    min_q = q_values[0, min_idx].item()
    logging.info(f"Min Selection: Chosen Candidate {min_idx} | Q-Value: {min_q:.4f}")
    return min_idx

def choose_median_action(
    q_values: torch.Tensor,
) -> int:
    flat_q = q_values.flatten()
    
    sorted_indices = torch.argsort(flat_q)
    mid_point = len(sorted_indices) // 2
    median_idx = sorted_indices[mid_point].item()
    median_q = flat_q[median_idx].item()
    
    logging.info(f"Median Selection: Chosen Candidate {median_idx} | Q-Value: {median_q:.4f}")
    return median_idx

def choose_median_action(
    q_values: torch.Tensor,
) -> int:
    flat_q = q_values.flatten()
    
    sorted_indices = torch.argsort(flat_q)
    mid_point = len(sorted_indices) // 2
    median_idx = sorted_indices[mid_point].item()
    median_q = flat_q[median_idx].item()
    
    logging.info(f"Median Selection: Chosen Candidate {median_idx} | Q-Value: {median_q:.4f}")
    return median_idx