import logging
import os
import sys
from datetime import datetime
import numpy as np
import torch

def setup_logging(log_filename_prefix: str = "evaluation_run"):
    """
    Sets up logging to both a file and the console.

    The log file will be created in a 'logs' directory and will include a timestamp.
    File logs will capture all messages at DEBUG level and above, while console
    logs will show messages at INFO level and above.

    Args:
        log_filename_prefix (str): A prefix for the log file name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_with_timestamp = f"{log_filename_prefix}_{timestamp}.log"

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename_with_timestamp)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler (logs everything at DEBUG level and above)
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console Handler (logs INFO level and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Output will be saved to: {log_filepath}")


def log_evaluation_results(
    q_values: torch.Tensor,
    best_idx: int,
    best_sequence: torch.Tensor,
    base_q_values: torch.Tensor = None,
    best_base_action: torch.Tensor = None,
    time_step: int = None,
    base_action_sequences: torch.Tensor = None,
):
    """
    Logs the detailed results of a VQH critical action evaluation step.

    - Uses INFO level for a summary of the winning action and its Q-value.
    - Uses DEBUG level for a detailed list of all candidate actions and their Q-values.

    Args:
        q_values (torch.Tensor): The Q-values of all candidate action sequences.
        best_idx (int): The index of the chosen best action sequence.
        best_sequence (torch.Tensor): The action sequence chosen as the best.
        base_q_values (torch.Tensor, optional): Q-values from the base policy group.
        best_base_action (torch.Tensor, optional): The best action from the base policy for logging.
        time_step (int, optional): The current time step in the episode.
        base_action_sequences (torch.Tensor, optional): All sampled candidate sequences.
    """
    q_values_np = q_values.cpu().numpy().flatten()

    # --- Part 1: Summary Info (INFO Level) ---
    summary_msgs = ["\n"]
    summary_msgs.append("=" * 70)
    if time_step is not None:
        summary_msgs.append(f"- Time Step Now - : {time_step}")
    summary_msgs.append("[VQH Q-Value Evaluation]")
    if base_q_values is not None and base_q_values.numel() > 0:
        base_q_str = ", ".join([f"{q:.4f}" for q in base_q_values.cpu().numpy().flatten()])
        summary_msgs.append(f"  - Base Policy Q-Values:  [{base_q_str}]")
    summary_msgs.append("  - Summary:")
    summary_msgs.append(f"    - Winning Candidate Idx: {best_idx}")
    summary_msgs.append(f"    - Winning Q-Value:       {q_values_np[best_idx]:.4f}")
    summary_msgs.append(f"    - Average Q-Value:       {np.mean(q_values_np):.4f}")
    logging.info("\n".join(summary_msgs))

    # --- Part 2: Detailed Candidate Info (DEBUG Level) ---
    details_msgs = ["-" * 70]
    details_msgs.append("[All Candidate Actions]")
    if base_action_sequences is not None and base_action_sequences.numel() > 0:
        details_msgs.append("  - Base Policy Actions:")
        for i, sequence in enumerate(base_action_sequences):
            q_value = q_values_np[i]
            details_msgs.append(f"    - Candidate {i} (Q={q_value:.4f}):")
            # sequence_np = sequence.cpu().numpy()[:5]  # Log first 5 steps
            sequence_np = sequence.cpu().numpy()
            for step_idx, step_action in enumerate(sequence_np):
                step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
                details_msgs.append(f"      - Step {step_idx+1}: [{step_str}]")
    else:
        details_msgs.append("  - No candidate sequences were provided for detailed logging.")
    logging.debug("\n".join(details_msgs))

    # --- Part 3: Final Decision Info (INFO Level) ---
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
    logging.info("\n".join(decision_msgs))


def log_default_action(sequence: torch.Tensor, time_step: int):
    """
    Logs the default action sequence when it's used as a fallback (non-critical step).

    Args:
        sequence (torch.Tensor): The default action sequence being executed.
        time_step (int): The current time step in the episode.
    """
    log_msgs = ["\n"]
    log_msgs.append("=" * 70)
    log_msgs.append(f"- Time Step Now - : {time_step}")
    log_msgs.append("[Decision]")
    log_msgs.append("  - ⚙️ Default action sequence was chosen (non-critical step).")
    log_msgs.append("-" * 70)
    log_msgs.append("[Default Action Executed]")
    
    sequence_np = sequence.cpu().numpy()
    for i, step_action in enumerate(sequence_np):
        step_str = ", ".join([f"{x: .4f}" for x in np.atleast_1d(step_action)])
        log_msgs.append(f"  - Step {i+1}: [{step_str}]")
    
    log_msgs.append("=" * 70)
    logging.info("\n".join(log_msgs))