import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message="The video decoding and encoding capabilities of torchvision are deprecated.*")

import wandb
import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812

from torch import Tensor, nn
from transformers import AutoProcessor

from accelerate import Accelerator, DistributedDataParallelKwargs
from termcolor import colored
from tqdm.auto import tqdm

from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import get_step_checkpoint_dir, load_training_state, save_checkpoint
from lerobot.common.utils.utils import init_logging
from lerobot.common.constants import OBS_STATE
from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.policies.normalize import Normalize,Unnormalize

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig as LeroBotTrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.value_query_head import ValueQueryHead
from src.lerobot_patch import make_dataset, make_policy

def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img

def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

@PreTrainedConfig.register_subclass("vqh")
@dataclass
class VQHPolicyConfig(PreTrainedConfig):
    type: str = "vqh"
    
    """ Optimization config """
    critic_lr: float = 1e-4
    vlm_lr: float = 2e-5

    """ Distributional RL config """
    num_atoms: int = 201
    v_min: float = 0.0
    v_max: float = 1.0

    """ VQH Policy config """
    discount: float = 0.98
    vqh_chunk_size: int = 10
    next_obs_offset: int = 10

    pretrained_smolvla_path: str = "lerobot/outputs/train/2025-12-04/15-18-52_smolvla/checkpoints/100000/pretrained_model"

    """ SmolVLM config """
    add_image_special_tokens: bool = False
    prefix_length: int = -1

    max_state_dim: int = 32
    max_action_dim: int = 32
    
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    pad_language_to: str = "longest"

    # Tokenizer
    tokenizer_max_length: int = 48

    def get_optimizer_preset(self):
        pass

    def get_scheduler_preset(self):
        pass

    def reward_delta_indices(self):
        return None

    def validate_features(self):
        pass

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.vqh_chunk_size))

@dataclass
class TrainVQHConfig(LeroBotTrainPipelineConfig):
    """The main configuration for the VQH training pipeline."""

    steps: int = 100_000
    save_freq: int = 20_000
    batch_size: int = 32

    policy: VQHPolicyConfig = field(default_factory=VQHPolicyConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: AdamWConfig = field(default_factory=AdamWConfig)

    def __post_init__(self):
        if hasattr(self.policy, "scheduler_decay_steps"):
            self.policy.scheduler_decay_steps = self.steps

class VQHPolicy(PreTrainedPolicy):
    config_class = VQHPolicyConfig
    name = "vqh"

    def __init__(self, config: VQHPolicyConfig, dataset_stats=None):
        super().__init__(config, dataset_stats)

        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.language_tokenizer = processor.tokenizer
        self.vqh = ValueQueryHead(config, self.language_tokenizer, config.pretrained_smolvla_path)
        self.vqh.to(torch.float32)

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

    def prepare_optimizers(self):
        critic_model = self.vqh.critic
        critic_kwargs = {
            "learning_rate": self.config.critic_lr,
            "warmup_steps": 200, 
        }
        critic_optimizer, critic_scheduler = critic_model.make_optimizer(critic_model.parameters(), **critic_kwargs)

        vlm_optimizer_kwargs = {
            "learning_rate": self.config.vlm_lr,
            "warmup_steps": 100,
            "weight_decay": 0.01,
        }
        vlm_optimizer, vlm_scheduler = critic_model.make_optimizer(self.vqh.vqh_backbone.parameters(), **vlm_optimizer_kwargs)

        optimizers = {
            "critic": critic_optimizer,
            "vlm": vlm_optimizer,
        }
        lr_schedulers = {
            "critic": critic_scheduler,
            "vlm": vlm_scheduler,
        }
        return optimizers, lr_schedulers
    
    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]: 
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN found in batch['{key}']"
                assert not torch.isinf(value).any(), f"Inf found in batch['{key}']"

        # normalize "state" "image" "wrist_image"
        batch = self.normalize_inputs(batch)
        # normalize "action"
        batch = self.normalize_targets(batch)

        # normalize "observation.state.vqh" using stats from "observation.state" in the input normalizer
        if "observation.state.vqh" in batch and "observation.state" in self.normalize_inputs.stats:
            device = batch["observation.state.vqh"].device
            state_mean = torch.tensor(self.normalize_inputs.stats["observation.state"]["mean"], device=device, dtype=batch["observation.state.vqh"].dtype)
            state_std = torch.tensor(self.normalize_inputs.stats["observation.state"]["std"], device=device, dtype=batch["observation.state.vqh"].dtype)
            batch["observation.state.vqh"] = (batch["observation.state.vqh"] - state_mean) / (state_std + 1e-8)

        # normalize "observation.images.image.vqh" using stats from "observation.images.image" in the input normalizer
        if "observation.images.image.vqh" in batch and "observation.images.image" in self.normalize_inputs.stats:
            device = batch["observation.images.image.vqh"].device
            image_mean = self.normalize_inputs.stats["observation.images.image"]["mean"].to(device=device, dtype=batch["observation.images.image.vqh"].dtype)
            image_std = self.normalize_inputs.stats["observation.images.image"]["std"].to(device=device, dtype=batch["observation.images.image.vqh"].dtype)
            batch["observation.images.image.vqh"] = (batch["observation.images.image.vqh"] - image_mean) / (image_std + 1e-8)

        # normalize "observation.images.wrist_image.vqh" using stats from "observation.images.wrist_image" in the input normalizer
        if "observation.images.wrist_image.vqh" in batch and "observation.images.wrist_image" in self.normalize_inputs.stats:
            device = batch["observation.images.wrist_image.vqh"].device
            wrist_mean = self.normalize_inputs.stats["observation.images.wrist_image"]["mean"].to(device=device, dtype=batch["observation.images.wrist_image.vqh"].dtype)
            wrist_std = self.normalize_inputs.stats["observation.images.wrist_image"]["std"].to(device=device, dtype=batch["observation.images.wrist_image.vqh"].dtype)
            batch["observation.images.wrist_image.vqh"] = (batch["observation.images.wrist_image.vqh"] - wrist_mean) / (wrist_std + 1e-8)

        if self.normalize_targets.stats and "reward" in self.normalize_targets.stats:
            device = batch["reward"].device
            reward_min = torch.tensor(self.normalize_targets.stats["reward"]["min"], device=device, dtype=batch["reward"].dtype)
            reward_max = torch.tensor(self.normalize_targets.stats["reward"]["max"], device=device, dtype=batch["reward"].dtype)
            denominator = reward_max - reward_min + 1e-8
            batch["reward"] = (batch["reward"] - reward_min) / denominator

        if self.normalize_targets.stats and "mc_returns" in self.normalize_targets.stats:
            device = batch["mc_returns"].device
            mc_min = torch.tensor(self.normalize_targets.stats["mc_returns"]["min"], device=device, dtype=batch["mc_returns"].dtype)
            mc_max = torch.tensor(self.normalize_targets.stats["mc_returns"]["max"], device=device, dtype=batch["mc_returns"].dtype)        
            denominator = mc_max - mc_min + 1e-8        
            batch["mc_returns"] = (batch["mc_returns"] - mc_min) / denominator
            batch["mc_returns"] = torch.clamp(batch["mc_returns"], 0.0, 1.0)

        # # normalize "reward"
        # if self.normalize_targets.stats and "reward" in self.normalize_targets.stats:
        #     device = batch["reward"].device
        #     reward_mean = torch.tensor(self.normalize_targets.stats["reward"]["mean"], device=device, dtype=batch["reward"].dtype)
        #     reward_std = torch.tensor(self.normalize_targets.stats["reward"]["std"], device=device, dtype=batch["reward"].dtype)
        #     batch["reward"] = (batch["reward"] - reward_mean) / (reward_std + 1e-8)

        # # normalize "mc_returns"
        # if self.normalize_targets.stats and "mc_returns" in self.normalize_targets.stats:
        #     device = batch["mc_returns"].device
        #     mc_mean = torch.tensor(self.normalize_targets.stats["mc_returns"]["mean"], device=device, dtype=batch["mc_returns"].dtype)
        #     mc_std = torch.tensor(self.normalize_targets.stats["mc_returns"]["std"], device=device, dtype=batch["mc_returns"].dtype)
        #     batch["mc_returns"] = (batch["mc_returns"] - mc_mean) / (mc_std + 1e-8)

        images, img_masks = self.prepare_images(batch)
        vqh_images, vqh_img_masks = self.prepare_images(batch, map(lambda x: f"{x}.vqh", self.config.image_features))

        state = self.prepare_state(batch, key="observation.state")
        vqh_state = self.prepare_state(batch, key="observation.state.vqh")

        lang_tokens, lang_masks = self.prepare_language(batch)

        action = self.prepare_action(batch)
        next_action = self.prepare_action(batch, key="next_action")

        reward = batch["reward"]
        mc_returns = batch["mc_returns"]
        masks = batch["tdm"].float()

        model_dtype = next(self.parameters()).dtype
        if state is not None:
            state = state.to(model_dtype)
        if vqh_state is not None:
            vqh_state = vqh_state.to(model_dtype)

        images = [img.to(model_dtype) for img in images]
        vqh_images = [img.to(model_dtype) for img in vqh_images]
        action = action.to(model_dtype)
        next_action = next_action.to(model_dtype)


        losses_dict, log_dict = self.vqh.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            vqh_images,
            vqh_img_masks,
            action,
            next_action,
            state,
            vqh_state,
            reward,
            mc_returns,
            masks,
        )

        total_loss = sum(losses_dict.values())
        for key, value in losses_dict.items():
            log_dict[f"{key}_loss"] = value.item()
        log_dict["total_loss"] = total_loss.item()
        return total_loss, log_dict

    def prepare_images(self, batch, image_features=None):
        """Apply preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        image_features = image_features or self.config.image_features
        present_img_keys = [key for key in image_features if key in batch]
        missing_img_keys = [key for key in image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})")

        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch, key: str):
        if key not in batch:
            return None
        state = batch[key][:, -1, :] if batch[key].ndim > 2 else batch[key]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch, key="action"):
        if key not in batch:
            raise ValueError(f"'{key}' key is missing from the batch.")
        actions = batch[key]

        if actions.dim() == 3:
            actions = actions[:, : self.config.vqh_chunk_size, :]
        elif actions.dim() == 2:
            actions = actions.unsqueeze(1)
        else:
            raise ValueError(f"Action tensor has an unsupported dimension: {actions.dim()}")
        return actions

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
        return lang_tokens, lang_masks

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        pass

    def select_action(self) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate_q_values(self, batch: dict, candidate_actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Q-values for a batch of states and candidate actions.
        Designed to be called by _compute_predicted_q_values with RAW data.
        """
        self.eval()
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        norm_batch = self.normalize_inputs(batch)

        temp_batch = {"action": candidate_actions}
        temp_batch_norm = self.normalize_targets(temp_batch)
        candidate_actions_norm = temp_batch_norm["action"]

        batch_size, num_candidates = candidate_actions_norm.shape[:2]
        candidate_actions_norm = candidate_actions_norm.reshape(batch_size, num_candidates, -1)
        
        images, img_masks = self.prepare_images(norm_batch)
        lang_tokens, lang_masks = self.prepare_language(norm_batch)
        state = self.prepare_state(norm_batch, key="observation.state")
        
        if state is not None:
            state = state.to(model_dtype)
        images = [img.to(model_dtype) for img in images]

        q_values = self.vqh.select_q_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            candidate_actions=candidate_actions_norm
        )        
        return q_values
    
    @torch.no_grad()
    def sample_and_evaluate_actions(self, batch: dict, num_candidates: int, default_sequence: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples candidate actions (using internal SmolVLA Policy) and evaluates them (using Distributional Critic).
        """
        self.eval()
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        
        # 1. 准备基础数据和常量
        # 获取实际的 batch size (通常 eval 时为 1)
        batch_size = next(iter(batch.values())).shape[0]
        raw_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        all_raw_actions_list = []

        # 2. 处理默认序列 (如果有)
        if default_sequence is not None:
            # default_sequence shape: [1, seq_len, dim] or [batch, seq_len, dim]
            default_sequence = default_sequence.to(device=device, dtype=model_dtype)
            all_raw_actions_list.append(default_sequence)

        # 3. 采样剩余需要的候选动作
        # 注意：这里计算的是总共需要多少个 candidate，减去已经有的 default
        current_count = len(all_raw_actions_list)
        num_to_sample = num_candidates - current_count
        
        if num_to_sample > 0:
            # 复制 batch 数据以进行并行采样
            # 假设 batch_size=1, 这里将其复制 num_to_sample 份
            batched_raw_batch = {
                key: value.repeat([num_to_sample] + [1] * (value.dim() - 1))
                if isinstance(value, torch.Tensor) else [value] * num_to_sample
                for key, value in raw_batch.items()
            }
            
            # 使用 Policy 采样动作
            l1_features, prefix_pad_masks, past_key_values = self.vqh.smolvla_policy.extract_l1_features(batched_raw_batch)
            sampled_actions_raw = self.vqh.smolvla_policy.sample_actions_from_l1_features(prefix_pad_masks, past_key_values)
            
            sampled_actions_raw = sampled_actions_raw.to(dtype=model_dtype)
            all_raw_actions_list.append(sampled_actions_raw)

        # 合并所有候选动作 (Raw Space)
        # Shape: [num_candidates, seq_len, action_dim] (假设 batch=1)
        all_actions_raw = torch.cat(all_raw_actions_list, dim=0)

        # 4. 动作处理：归一化 -> 切片 (Chunking) -> Flatten
        actions_dict = {"action": all_actions_raw}
        normalized_dict = self.normalize_targets(actions_dict)
        all_actions_norm = normalized_dict["action"]

        # [CRITICAL] 恢复切片逻辑：确保动作长度不超过 Critic 训练时的 chunk size
        if hasattr(self.config, 'vqh_chunk_size') and self.config.vqh_chunk_size is not None:
            all_actions_norm = all_actions_norm[:, :self.config.vqh_chunk_size, :]
        
        # Flatten 动作以适应 Critic 输入接口
        # Target Shape: [batch_size, num_candidates, flattened_dim]
        # flattened_dim = seq_len * action_dim
        num_total_candidates = all_actions_norm.shape[0]
        # 这里假设 batch_size=1，如果 batch>1，需要仔细处理 num_total 的 reshape 逻辑
        all_actions_norm_flat = all_actions_norm.reshape(batch_size, num_total_candidates // batch_size, -1)

        # 5. 观测数据处理：归一化 -> 提取特征
        norm_batch = self.normalize_inputs(raw_batch)
        
        images, img_masks = self.prepare_images(norm_batch)
        lang_tokens, lang_masks = self.prepare_language(norm_batch)
        state = self.prepare_state(norm_batch, key="observation.state")
        
        # 确保 dtype 正确
        if state is not None: state = state.to(model_dtype)
        if images is not None: images = [img.to(model_dtype) for img in images]

        # 6. Critic 评分
        q_values = self.vqh.select_q_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            candidate_actions=all_actions_norm_flat
        )
        
        # 返回：
        # 1. 原始动作序列 (用于执行，未切片，方便后续 debug 或执行长序列)
        # 2. Q 值 (用于排序)
        # 调整 all_actions_raw 维度为 [batch_size, num_candidates, seq, dim]
        return all_actions_raw.unsqueeze(0), q_values

def update_policy(train_metrics: MetricsTracker, policy: VQHPolicy, batch: dict, optimizers: dict, lr_schedulers: dict, accelerator: Accelerator, cfg: TrainVQHConfig) -> dict:

    policy.train()

    with accelerator.accumulate(policy):

        total_loss, output_dict = policy.forward(batch)

        accelerator.backward(total_loss)

        if accelerator.is_main_process and accelerator.sync_gradients:
            unwrapped_policy = accelerator.unwrap_model(policy)

        if accelerator.sync_gradients:
            unwrapped_policy = accelerator.unwrap_model(policy)

            critic_head_params = unwrapped_policy.vqh.critic.parameters()
            critic_head_grads = [p.grad.detach().flatten() for p in critic_head_params if p.grad is not None]
            if critic_head_grads:
                output_dict["critic_head_grad_norm"] = torch.cat(critic_head_grads).norm(2).item()
            
            vlm_params = unwrapped_policy.vqh.vqh_backbone.parameters()
            vlm_grads = [p.grad.detach().flatten() for p in vlm_params if p.grad is not None]
            if vlm_grads:
                vlm_grad_norm = torch.cat(vlm_grads).norm(2).item()
                output_dict["vlm_grad_norm"] = vlm_grad_norm
            
            grad_clip_value = cfg.optimizer.grad_clip_norm if cfg.optimizer else 1.0
            grad_norm = accelerator.clip_grad_norm_(policy.vqh.parameters(), grad_clip_value)
            output_dict["grad_norm"] = grad_norm.item()

        for optimizer in optimizers.values():
            optimizer.step()
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        if lr_schedulers:
            for scheduler in lr_schedulers.values():
                scheduler.step()


    output_dict["lr_critic"] = optimizers["critic"].param_groups[0]["lr"]
    output_dict["lr_vlm"] = optimizers["vlm"].param_groups[0]["lr"]

    return output_dict

@parser.wrap()
def train(cfg: TrainVQHConfig):
    cfg.__post_init__()
    init_logging()

    cfg.validate()

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    if accelerator.is_main_process:
        wandb.init(
            mode="online",
            project="lerobot-vqh-training",
            config=vars(cfg),
            name=f"train-{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )

    logging.info(f"Accelerator initialized on process {accelerator.process_index} with {accelerator.num_processes} processes.")

    if cfg.seed is not None:
        set_seed(cfg.seed)

    dataset = make_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        policy_cls=VQHPolicy,
    )

    if accelerator.is_main_process:
        print(colored("=" * 60, "yellow"))
        print(colored("Inspecting Initialized Policy Modules:", "yellow"))
        unwrapped_policy = accelerator.unwrap_model(policy)
        for name, module in unwrapped_policy.named_children():
            print(colored(f"  - {name}: {module.__class__.__name__}", "yellow"))
        print(colored("=" * 60, "yellow"))

    optimizers, lr_schedulers = policy.prepare_optimizers()

    step = 0
    if cfg.resume:
        step, optimizers, lr_schedulers = load_training_state(cfg.checkpoint_path, optimizers, lr_schedulers)

    policy, dataloader, optimizers, lr_schedulers = accelerator.prepare(policy, dataloader, optimizers, lr_schedulers)

    dl_iter = cycle(dataloader)

    num_episodes = dataset.num_episodes
    if num_episodes == 0:
        logging.warning("Dataset contains 0 episodes. Metrics related to episodes will not be tracked correctly.")
        num_episodes = 1

    train_metrics_def = {}
    train_tracker = MetricsTracker(cfg.batch_size, len(dataset), num_episodes, train_metrics_def, initial_step=step)
    progress_bar = tqdm(range(step, cfg.steps), disable=not accelerator.is_local_main_process, initial=step, total=cfg.steps, mininterval=0.5,dynamic_ncols=True,desc="Training")

    for step in range(step, cfg.steps):

        batch = next(dl_iter)
        output_dict = update_policy(train_tracker, policy, batch, optimizers, lr_schedulers, accelerator, cfg)
        train_tracker.step()

        for key, value in output_dict.items():
            if key not in train_tracker.metrics:
                if accelerator.is_main_process:
                    logging.info(colored(f"Dynamically registering new metric: {key}", "green"))
                train_tracker.metrics[key] = AverageMeter(key, ":.3f")
            setattr(train_tracker, key, value)

        progress_bar.update(1)

        metrics_dict = train_tracker.to_dict()

        loss = metrics_dict.get('total_loss', 0)
        grad_norm = metrics_dict.get('grad_norm', 0)
        postfix_str = f"loss={loss:.4f}, grad={grad_norm:.4f}"
        progress_bar.set_postfix_str(postfix_str)

        if accelerator.is_main_process:
            wandb.log(train_tracker.to_dict(), step=step)

        if step > 0 and step % cfg.save_freq == 0 and accelerator.is_main_process:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            logging.info(colored(f"--- Saving checkpoint at step {step} to {checkpoint_dir} ---", "cyan"))
            save_checkpoint(get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step), step, cfg, accelerator.unwrap_model(policy), optimizers, lr_schedulers)

    if accelerator.is_main_process:
        final_step = cfg.steps
        checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, final_step, final_step)
        logging.info(colored(f"--- Saving final checkpoint at step {final_step} to {checkpoint_dir} ---", "green"))
        save_checkpoint(checkpoint_dir, final_step, cfg, accelerator.unwrap_model(policy), optimizers, lr_schedulers)

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.end_training()

if __name__ == "__main__":
    train()