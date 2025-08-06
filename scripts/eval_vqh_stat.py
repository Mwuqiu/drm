# import torch
# import argparse
# import logging
# import numpy as np
# from pathlib import Path
# from omegaconf import OmegaConf
# from tqdm import tqdm
# from types import SimpleNamespace

# from lerobot.common.utils.utils import init_logging
# from lerobot.configs.default import DatasetConfig
# # Import the policy class to access its methods
# from vqh_training_policy import VQHPolicy, make_dataset

# # The actual action dimension from the dataset.
# ACTION_DIM = 7 

# @torch.no_grad()
# def evaluate_on_dataset(args):
#     """
#     Main function to load the model and evaluate its Q-function on a dataset.
#     """
#     init_logging()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device: {device}")

#     policy_path = args.checkpoint_dir
#     logging.info(f"Loading policy from: {policy_path}")
#     # dataset_stats are loaded automatically by from_pretrained
#     policy = VQHPolicy.from_pretrained(policy_path)
#     policy.to(device)
#     policy.eval()

#     # Get the "live" policy config which includes @property methods
#     policy_cfg_live = policy.config 
    
#     # Create a complete, default dataset config object
#     dataset_cfg = OmegaConf.structured(DatasetConfig)

#     # Set dataset source from command-line arguments
#     if args.repo_id:
#         dataset_cfg.repo_id = args.repo_id
#     elif args.dataset_path:
#         dataset_cfg.path = args.dataset_path
    
#     # Use a simple Python object as a container to avoid OmegaConf's type-stripping
#     eval_cfg_container = SimpleNamespace()
#     eval_cfg_container.policy = policy_cfg_live
#     eval_cfg_container.dataset = dataset_cfg

#     # Safely get the source info for logging
#     if args.repo_id:
#         source_info = eval_cfg_container.dataset.repo_id
#         logging.info(f"Loading dataset from Hugging Face Hub repo_id: {source_info}")
#     else:
#         source_info = eval_cfg_container.dataset.path
#         logging.info(f"Loading dataset from local path: {source_info}")
        
#     dataset = make_dataset(eval_cfg_container)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, 
#         num_workers=args.num_workers, 
#         batch_size=args.batch_size,
#     )
    
#     metrics = {
#         "q_ground_truth": [],
#         "q_max_noise": [],
#         "rank_of_gt": [],
#         "top_1_accuracy": [],
#     }
    
#     num_processed = 0
#     progress_bar = tqdm(total=args.num_samples, desc="Evaluating on dataset")

#     for batch in dataloader:
#         if num_processed >= args.num_samples:
#             break
        
#         # We need the ground-truth action to insert it into the candidates.
#         # It's better to prepare it here once.
#         ground_truth_action = policy.prepare_action(batch)
#         current_batch_size = ground_truth_action.shape[0]

#         # Generate random "noise" actions with the original 7D dimension
#         noise_actions = torch.randn(
#             current_batch_size,
#             args.num_candidates,
#             policy_cfg_live.vqh_chunk_size,
#             ACTION_DIM,
#             device=device,
#         )
    
#         # Replace one of the random actions with the ground-truth action
#         gt_insert_idx = np.random.randint(0, args.num_candidates)
#         noise_actions[:, gt_insert_idx] = ground_truth_action
        
#         # ######################################################################
#         # # THE REFACTORED CALL: All complexity is now hidden in the policy    #
#         # ######################################################################
#         q_values = policy.evaluate_q_values(batch, noise_actions)
        
#         # Calculate metrics for this batch
#         q_values_cpu = q_values.detach().cpu()
        
#         q_gt = q_values_cpu[torch.arange(current_batch_size), gt_insert_idx]
#         metrics["q_ground_truth"].extend(q_gt.tolist())
        
#         q_max = q_values_cpu.max(dim=1).values
#         metrics["q_max_noise"].extend(q_max.tolist())
        
#         sorted_indices = torch.argsort(q_values_cpu, dim=1, descending=True)
#         ranks = (sorted_indices == gt_insert_idx).nonzero(as_tuple=True)[1] + 1
#         metrics["rank_of_gt"].extend(ranks.tolist())

#         is_top_1 = (ranks == 1).float()
#         metrics["top_1_accuracy"].extend(is_top_1.tolist())

#         num_processed += current_batch_size
#         progress_bar.update(current_batch_size)

#     progress_bar.close()

#     # Report final results
#     logging.info("--------- Evaluation Results ---------")
#     logging.info(f"Evaluated on {num_processed} samples.")
    
#     avg_rank = np.mean(metrics["rank_of_gt"])
#     logging.info(f"Average Rank of Ground-Truth Action: {avg_rank:.2f} (out of {args.num_candidates})")
    
#     top_1_acc = np.mean(metrics["top_1_accuracy"]) * 100
#     logging.info(f"Top-1 Accuracy: {top_1_acc:.2f}% (GT action had the highest Q-value)")

#     avg_q_gt = np.mean(metrics["q_ground_truth"])
#     logging.info(f"Average Q-value of Ground-Truth Action: {avg_q_gt:.4f}")
    
#     avg_q_max = np.mean(metrics["q_max_noise"])
#     logging.info(f"Average Max Q-value across all candidates: {avg_q_max:.4f}")
#     logging.info("--------------------------------------")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate a trained VQH model on a dataset.")
#     parser.add_argument(
#         "--checkpoint_dir",
#         type=str,
#         required=True,
#         help="Path to the checkpoint directory."
#     )
    
#     dataset_group = parser.add_mutually_exclusive_group(required=True)
#     dataset_group.add_argument(
#         "--dataset_path",
#         type=str,
#         default=None,
#         help="Local path to the LeRobot dataset to evaluate on."
#     )
#     dataset_group.add_argument(
#         "--repo_id",
#         type=str,
#         default=None,
#         help="Hugging Face Hub repo_id of the dataset (e.g., 'lerobot/aloha_sim_transfer_cube_human')."
#     )
    
#     parser.add_argument("--num_samples", type=int, default=16, help="Number of dataset samples to evaluate.")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
#     parser.add_argument("--num_candidates", type=int, default=8, help="Number of candidate actions (N-1 noise + 1 GT) to evaluate per sample.")
    
#     args = parser.parse_args()
#     evaluate_on_dataset(args)


import torch
import argparse
import logging
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib
# Use a non-GUI backend for plotting on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from lerobot.common.utils.utils import init_logging
from lerobot.configs.default import DatasetConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from vqh_training_policy import VQHPolicy, make_dataset

@torch.no_grad()
def evaluate_q_value_correlation(args):
    init_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    repo_id = args.repo_id    
    dataset = LeRobotDataset(repo_id)
    dataset_stats = dataset.stats

    policy_path = args.policy_path
    policy = VQHPolicy.from_pretrained(policy_path,dataset_stats=dataset_stats)
    policy.to(device)
    policy.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size,
    )

    all_predicted_q_values = []
    all_normalized_mc_returns = []
    
    num_processed = 0
    progress_bar = tqdm(total=args.num_samples, desc="Evaluating Q-value correlation")

    for batch in dataloader:
        if num_processed >= args.num_samples:
            break
        
        ground_truth_action = policy.prepare_action(batch)
        candidate_actions = ground_truth_action.unsqueeze(1)
        
        predicted_q_values = policy.evaluate_q_values(batch, candidate_actions)
        predicted_q_values = predicted_q_values.squeeze(1)
        
        mc_returns_raw = batch["mc_returns"].to(device)
        normalized_mc_returns = mc_returns_raw

        if "mc_returns" in policy.normalize_targets.stats:
            stats = policy.normalize_targets.stats["mc_returns"]
            mc_mean = torch.tensor(stats["mean"], device=device, dtype=mc_returns_raw.dtype)
            mc_std = torch.tensor(stats["std"], device=device, dtype=mc_returns_raw.dtype)
            normalized_mc_returns = (mc_returns_raw - mc_mean) / (mc_std + 1e-8)
        
        all_predicted_q_values.append(predicted_q_values.cpu())
        all_normalized_mc_returns.append(normalized_mc_returns.cpu())

        num_processed += ground_truth_action.shape[0]
        progress_bar.update(ground_truth_action.shape[0])

    progress_bar.close()

    predicted_q_values_np = torch.cat(all_predicted_q_values).numpy()
    mc_returns_np = torch.cat(all_normalized_mc_returns).numpy()

    mae = np.mean(np.abs(predicted_q_values_np - mc_returns_np))
    correlation, p_value = pearsonr(predicted_q_values_np, mc_returns_np)

    logging.info("--------- Q-Value vs. MC-Return Evaluation Results ---------")
    logging.info(f"Evaluated on {num_processed} samples.")
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"Pearson Correlation Coefficient: {correlation:.4f} (p-value: {p_value:.4f})")
    logging.info("----------------------------------------------------------")

    plt.figure(figsize=(8, 8))
    plt.scatter(predicted_q_values_np, mc_returns_np, alpha=0.5)
    plt.title('Predicted Q-Value vs. Normalized MC-Return')
    plt.xlabel('Predicted Q-Value')
    plt.ylabel('Normalized Ground-Truth MC-Return')

    min_val = min(predicted_q_values_np.min(), mc_returns_np.min())
    max_val = max(predicted_q_values_np.max(), mc_returns_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Perfect Correlation)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    plot_filename = "q_vs_mc_return.png"
    plt.savefig(plot_filename)
    logging.info(f"Scatter plot saved to {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained VQH model by comparing Q-values to MC returns.")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
    )
    
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Local path to the LeRobot dataset to evaluate on."
    )
    dataset_group.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face Hub repo_id of the dataset (e.g., 'lerobot/aloha_sim_transfer_cube_human')."
    )
    
    parser.add_argument("--num_samples", type=int, default=512, help="Number of dataset samples to evaluate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    args = parser.parse_args()
    evaluate_q_value_correlation(args)
