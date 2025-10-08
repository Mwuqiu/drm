import logging
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from reward_dataset import RewardLeRobotDataset 

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inspect_and_generate_annotations(repo_id: str, output_path: str):
    """
    检查 LeRobot 数据集，找出包含正奖励的 episode，
    并为这些 episode 生成一个只包含最后三帧的标注 JSON 文件。

    Args:
        repo_id (str): 需要检查的数据集的仓库 ID。
        output_path (str): 输出的 JSON 文件路径。
    """
    logging.info(f"开始加载数据集: {repo_id}")
    try:
        dataset = RewardLeRobotDataset(repo_id=repo_id)
        logging.info(f"数据集加载成功，共包含 {dataset.num_episodes} 个 episodes。")
    except Exception as e:
        logging.error(f"无法加载数据集 '{repo_id}'。请确保仓库 ID 正确且数据集存在。错误: {e}")
        return

    if dataset.num_episodes == 0:
        logging.warning("数据集中没有任何 episodes。")
        return

    final_annotations = {}
    episodes_with_rewards_indices = []

    logging.info("开始检查每个 episode 的奖励并计算帧长度...")
    
    for i in tqdm(range(dataset.num_episodes), desc="检查 Episodes"):
        ep_start_abs = dataset.episode_data_index["from"][i].item()
        ep_end_abs = dataset.episode_data_index["to"][i].item()
        
        found_positive_reward = False
        for frame_idx in range(ep_start_abs, ep_end_abs):
            reward_value = dataset[frame_idx]['reward'].item()
            if reward_value > 0:
                found_positive_reward = True
                break
        
        if found_positive_reward:
            episodes_with_rewards_indices.append(i)
            
            episode_length = ep_end_abs - ep_start_abs
            
            if episode_length < 3:
                tqdm.write(f"警告: Episode {i} 有奖励但总帧数 ({episode_length}) 少于3帧，无法生成标注。已跳过。")
                continue
                
            start_frame = episode_length - 3
            end_frame = episode_length - 1
            
            final_annotations[str(i)] = [[start_frame, end_frame]]

    output_file = Path(output_path)
    
    print("\n" + "=" * 70)
    print("### 检查与生成完成 ###")
    print(f"针对数据集: {repo_id}")
    print("=" * 70 + "\n")
    
    print(f"共发现 {len(episodes_with_rewards_indices)} 个含有奖励的 Episode:")
    print(episodes_with_rewards_indices if episodes_with_rewards_indices else "  无")
    
    if final_annotations:
        logging.info(f"正在将 {len(final_annotations)} 条标注数据保存到 {output_file.resolve()}...")
        with output_file.open('w') as f:
            json.dump(final_annotations, f, indent=4)
        logging.info("文件保存成功！")
    else:
        logging.warning("没有发现任何含有奖励的 episode，因此不创建 JSON 文件。")
        
    print("\n" + "=" * 70)

def main():
    """主函数，用于解析命令行参数并调用核心功能。"""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="检查一个 reward 数据集，并为有奖励的 episode 生成一个只包含最后三帧的 JSON 标注文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--repo_id', 
        type=str, 
        required=True, 
        help='需要检查的数据集的 Hugging Face 仓库 ID。\n'
             '示例: "libero_elegant_1001/task_3_reward"'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='指定输出的 JSON 文件路径。\n'
             '示例: "generated_annotations.json"'
    )

    args = parser.parse_args()

    inspect_and_generate_annotations(args.repo_id, args.output)

if __name__ == "__main__":
    main()