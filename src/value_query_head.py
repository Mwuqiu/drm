import math
from copy import deepcopy
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from beartype import beartype as typechecker
from jaxtyping import Bool, Int64, Float, jaxtyped

from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForImageTextToText, PretrainedConfig, PreTrainedModel

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy



class DistributionalBatch(TypedDict):
    encoded_observations: Float[Tensor, "batch encoded_dim"]
    actions: Float[Tensor, "batch action_dim"]
    mc_returns: Float[Tensor, " batch"] 
    masks: Float[Tensor, " batch"]

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activations: Union[Callable[[torch.Tensor], torch.Tensor], str] = "silu",
        activate_final: bool = False,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()

        assert not (use_layer_norm and use_group_norm)

        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        if isinstance(activations, str):
            if activations == "silu" or activations == "swish":
                self.activations = nn.SiLU()
            else:
                self.activations = getattr(nn, activations)()
        else:
            self.activations = activations

        layers = []

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

            input_dim = hidden_dim

            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))

                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                elif use_group_norm:
                    num_groups = min(hidden_dim, 32)
                    layers.append(nn.GroupNorm(num_groups, hidden_dim))
                layers.append(self.activations)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class DistributionalCritics(nn.Module):
    def __init__(
        self,
        obs_encoded_dim: int,
        networks: list[nn.Module],
        num_backbones: int = 1, 
        num_atoms: int = 201,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ):
        super().__init__()
        assert len(networks) == num_backbones
        self.obs_encoded_dim = obs_encoded_dim
        self.networks = nn.ModuleList(networks)
        self.num_backbones = num_backbones
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.backbone_output_dims = networks[0].hidden_dims[-1]

        self.output_layer = nn.Linear(self.backbone_output_dims, num_atoms)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
        self.register_buffer(
            "support", 
            torch.linspace(v_min, v_max, num_atoms)
        )
    
    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        encoded_observations: Float[torch.Tensor, "batch {self.obs_encoded_dim}"],
        actions: Float[torch.Tensor, "batch *num_actions action_dim"],
    ) -> Float[torch.Tensor, "{self.num_backbones} batch *num_actions {self.num_atoms}"]:
        if actions.ndim == 3:
            encoded_observations = encoded_observations.unsqueeze(1).expand(-1, actions.shape[1], -1)
            
        inputs = torch.cat([encoded_observations, actions], dim=-1)

        backbone_outputs = []
        for network in self.networks:
            backbone_outputs.append(network(inputs))
        
        backbone_outputs = torch.stack(backbone_outputs, dim=0)
        logits = self.output_layer(backbone_outputs)
        
        return logits

    def get_probs(self, encoded_observations, actions):
        logits = self.forward(encoded_observations, actions)
        return torch.softmax(logits, dim=-1)

    def get_expected_values(self, encoded_observations, actions):
        probs = self.get_probs(encoded_observations, actions)
        expected_values = torch.sum(probs * self.support, dim=-1)
        return expected_values

class DistributionalConfig(PretrainedConfig):
    model_type = "distributional_critic"
    def __init__(
        self,
        obs_encoded_dim=2048,
        action_dim=70,
        lr=3e-4,
        warmup_steps=2000,
        num_atoms=201,
        v_min=0.0,
        v_max=1.0,
        **kwargs,
    ):
        self.action_dim = action_dim
        self.obs_encoded_dim = obs_encoded_dim
        self.critic_ensemble_size = 1

        self.critic_network_kwargs = dict(
            input_dim=self.obs_encoded_dim + self.action_dim,
            hidden_dims=[256, 256, 256],
            activate_final=True,
            use_layer_norm=False,
        )

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.optimizer_kwargs = dict(
            learning_rate=lr,
            warmup_steps=warmup_steps,
        )
        super().__init__(**kwargs)

class DistributionalCriticModel(PreTrainedModel):
    config_class = DistributionalConfig

    def __init__(self, config: DistributionalConfig):
        super().__init__(config=config)
        self.config = config

        self.critics = DistributionalCritics(
            obs_encoded_dim=self.config.obs_encoded_dim,
            networks=[
                MLP(**self.config.critic_network_kwargs) 
                for _ in range(self.config.critic_ensemble_size)
            ],
            num_backbones=self.config.critic_ensemble_size,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
        )

        self.loss_fn = nn.CrossEntropyLoss()
    
    def discretize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        returns = torch.clamp(returns, self.config.v_min, self.config.v_max)
        bin_width = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)
        indices = ((returns - self.config.v_min) / bin_width).round().long()
        return indices
    
    def compute_loss(self, batch: DistributionalBatch):
        pred_logits = self.critics.forward(batch["encoded_observations"], batch["actions"])

        target_returns = batch["mc_returns"]
        target_labels = self.discretize_returns(target_returns)

        loss = 0.0
        for i in range(self.config.critic_ensemble_size):
            # input: (batch, num_classes), target: (batch)
            loss += self.loss_fn(pred_logits[i], target_labels)
            
        return loss, {
            "loss": loss.item(),
            "mean_mc_return": target_returns.mean().item(),
            "mean_pred_value": self.critics.get_expected_values(
                batch["encoded_observations"], batch["actions"]
            ).mean().item()
        }
    
    @staticmethod
    def make_optimizer(
        params: torch.nn.Module,
        learning_rate: float = 3e-4,
        warmup_steps: int = 0,
        cosine_decay_steps: Optional[int] = None,
        weight_decay: Optional[float] = None,
        return_lr_schedule: bool = True,
    ) -> Union[Optimizer, Tuple[Optimizer, LambdaLR]]:
        optimizer: Optimizer
        if weight_decay is not None:
            optimizer = AdamW(params=params, lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = Adam(params=params, lr=learning_rate)

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            if cosine_decay_steps is not None:
                decay_step = step - warmup_steps
                if decay_step < 0: return 0.0
                if decay_step >= cosine_decay_steps: return 0.0
                progress = decay_step / cosine_decay_steps
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
        if return_lr_schedule:
            return optimizer, scheduler
        else:
            return optimizer

    def forward(self, batch: DistributionalBatch):
        loss, info = self.compute_loss(batch)
        return {"loss": loss}, info

    @jaxtyped(typechecker=typechecker)
    def get_q_values(
        self,
        encoded_observations: Float[torch.Tensor, "batch {self.config.obs_encoded_dim}"],
        actions: Float[torch.Tensor, "batch *num_actions action_dim"],
    ) -> Float[torch.Tensor, "batch *num_actions"]: # 返回值也加星号
        expected_values = self.critics.get_expected_values(encoded_observations, actions) 
        return expected_values.mean(dim=0)

class ValueQueryHead(nn.Module):
    def __init__(self, config, tokenizer, pretrained_smolvla_path: str):
        super().__init__()
        self.config = config

        # 1. 加载 SmolVLA Policy (作为特征提取器)
        self.smolvla_policy = SmolVLAPolicy.from_pretrained(pretrained_smolvla_path)
        self.feature_extractor = self.smolvla_policy.model 
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. 加载 VQH Backbone (用于处理特征)
        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        self.vqh_backbone = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )
        self.vqh_backbone.gradient_checkpointing_enable()

        # 3. 投影层 (L1 -> L2)
        l1_hidden_size = self.feature_extractor.vlm_with_expert.vlm.config.text_config.hidden_size
        l2_hidden_size = self.vqh_backbone.config.text_config.hidden_size
        self.embedding_proj = nn.Linear(l1_hidden_size, l2_hidden_size)

        self.tokenizer = tokenizer 
        # Token 处理逻辑保持不变...
        self.fake_image_token = tokenizer.fake_image_token_id
        self.global_image_token = tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

        # 4. 初始化 Distributional Critic
        vlm_hidden_size = self.vqh_backbone.config.text_config.hidden_size
        action_dim_flat = config.vqh_chunk_size * config.action_feature.shape[0]

        # 使用新的 DistributionalConfig
        dist_config = DistributionalConfig(
            obs_encoded_dim=vlm_hidden_size,
            action_dim=action_dim_flat,
            lr=config.critic_lr, # 复用原本的 LR 配置
            warmup_steps=config.critic_wps if hasattr(config, "critic_wps") else 2000,
            num_atoms=201,       # 对应论文
            v_min=0.0,           # 对应 Min-Max Normalization
            v_max=1.0,
        )

        self.critic = DistributionalCriticModel(config=dist_config)

    @torch.no_grad()
    def _get_features_from_l1_extractor(
        self,
        images: list[Float[Tensor, "batch 3 512 512"]],
        img_masks: list[Bool[Tensor, " batch"]],
        lang_tokens: Int64[Tensor, "batch seq_len"],
        lang_masks: Bool[Tensor, "batch seq_len"],
        state,
    ) -> tuple[Float[Tensor, "batch seq_len hidden_dim"], Bool[Tensor, "batch seq_len"]]:
        
        prefix_out, prefix_pad_masks, _ = self.feature_extractor.extract_prefix_features(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        return prefix_out, prefix_pad_masks

    def _get_vqh_embeddings(self, images, img_masks, lang_tokens, lang_masks, state):
        """
        辅助函数：从 Raw Inputs 提取最终用于 Critic 的 Embedding
        """
        # 1. L1 提取 (Frozen)
        with torch.no_grad():
            intermediate_embs, pad_masks = self._get_features_from_l1_extractor(
                images, img_masks, lang_tokens, lang_masks, state
            )
        
        # 2. 投影
        projected_embs = self.embedding_proj(intermediate_embs)

        # 3. VQH Backbone Forward (Trainable)
        batch_size, seq_len = projected_embs.shape[:2]
        dummy_input_ids = torch.zeros(
            (batch_size, seq_len), 
            dtype=torch.long, 
            device=projected_embs.device
        )
        
        outputs = self.vqh_backbone.model.forward(
            input_ids=dummy_input_ids,
            inputs_embeds=projected_embs.to(self.vqh_backbone.dtype),
            attention_mask=pad_masks,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # 4. 提取最后一个 Token 的 Embedding
        last_token_indices = pad_masks.sum(dim=1) - 1
        final_embs = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_token_indices]
        
        return final_embs.to(dtype=torch.float32)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        images: list[Float[Tensor, "batch 3 512 512"]],
        img_masks: list[Bool[Tensor, " batch"]],
        lang_tokens: Int64[Tensor, "batch seq_len"],
        lang_masks: Bool[Tensor, "batch seq_len"],
        vqh_images: list[Float[Tensor, "batch 3 512 512"]],
        vqh_img_masks: list[Bool[Tensor, " batch"]],
        actions,
        next_actions,
        state,
        vqh_state,
        rewards: Float[Tensor, " batch"],
        mc_returns: Float[Tensor, " batch"],
        masks: Float[Tensor, " batch"],
    )  -> tuple[dict[str, Tensor], dict]:
        cur_encoded_obs = self._get_vqh_embeddings(images, img_masks, lang_tokens, lang_masks, state)
        dist_batch: DistributionalBatch = dict(
            encoded_observations=cur_encoded_obs,
            actions=actions.view(actions.shape[0], -1), # Flatten actions
            mc_returns=mc_returns,
            masks=masks,
        )
        losses_dict, critic_log_dict = self.critic(dist_batch)
        log_dict = {}
        for key, value in critic_log_dict.items():
            if hasattr(value, "item"):
                log_dict[key] = value.mean().item() if value.numel() > 1 else value.item()
            else:
                log_dict[key] = value 
        return losses_dict, log_dict
    
    @jaxtyped(typechecker=typechecker)
    def select_q_actions(
        self,
        images: list[Float[Tensor, "batch 3 512 512"]],
        img_masks: list[Bool[Tensor, " batch"]],
        lang_tokens: Int64[Tensor, "batch seq_len"],
        lang_masks: Bool[Tensor, "batch seq_len"],
        state,
        candidate_actions: Float[Tensor, "batch num_candidates action_dim_flat"]
    ) -> Float[Tensor, "batch num_candidates"]:
        
        cur_encoded_obs = self._get_vqh_embeddings(images, img_masks, lang_tokens, lang_masks, state)

        num_candidates = candidate_actions.shape[1]
        batch_size = cur_encoded_obs.shape[0]
        
        expanded_obs = cur_encoded_obs.unsqueeze(1).expand(-1, num_candidates, -1)
        expanded_obs = expanded_obs.reshape(batch_size * num_candidates, -1)
        
        flat_actions = candidate_actions.reshape(batch_size * num_candidates, -1)
        flat_scores = self.critic.get_q_values(expanded_obs, flat_actions)
        
        scores = flat_scores.reshape(batch_size, num_candidates)

        return scores