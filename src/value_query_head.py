import math
from copy import deepcopy
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union
from typing import Annotated, TypeAlias, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from beartype import beartype as typechecker
from jaxtyping import Bool, Int64, Float, jaxtyped
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    TanhTransform,
)
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
)
from transformers import (
    AutoConfig,
    GemmaModel,
    PretrainedConfig,
    PreTrainedModel,
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)
from transformers.models.auto import CONFIG_MAPPING
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full((b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

def print_gpu_memory(step_name=""):
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"[{step_name}] GPU Memory: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")


class CalQlBatch(TypedDict):
    encoded_observations: Float[Tensor, "batch encoded_dim"]
    encoded_next_observations: Float[Tensor, "batch encoded_dim"]
    actions: Float[Tensor, "batch action_dim"]
    rewards: Float[Tensor, " batch"]
    mc_returns: Float[Tensor, " batch"]
    masks: Float[Tensor, " batch"]


class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Tuple[int, ...] = (),
        constraint_type: str = "eq",  # One of ("eq", "leq", "geq")
        parameterization: Optional[str] = None,  # One of ("softplus", "exp"), or None for equality constraints
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization

        if constraint_type != "eq":
            assert init_value > 0, "Inequality constraints must have non-negative initial multiplier values"

            if parameterization == "softplus":
                init_value = torch.log(torch.exp(torch.tensor(init_value)) - 1).item()
            elif parameterization == "exp":
                init_value = torch.log(torch.tensor(init_value)).item()
            else:
                raise ValueError(f"Invalid multiplier parameterization {parameterization}")
        else:
            assert parameterization is None, "Equality constraints must have no parameterization"

        self.multiplier = nn.Parameter(torch.full(constraint_shape, init_value))

    def forward(self, lhs: Optional[torch.Tensor] = None, rhs: Optional[torch.Tensor] = None) -> torch.Tensor:
        multiplier = self.multiplier

        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = torch.nn.functional.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = torch.exp(multiplier)
            else:
                raise ValueError(f"Invalid multiplier parameterization {self.parameterization}")

        if lhs is None:
            return multiplier

        if rhs is None:
            rhs = torch.zeros_like(lhs)

        diff = lhs - rhs

        assert diff.shape == multiplier.shape, f"Shape mismatch: {diff.shape} vs {multiplier.shape}"

        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff


GeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="geq", parameterization="softplus")

LeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="leq", parameterization="softplus")


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


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        self.loc = loc
        self.scale_diag = scale_diag
        base_distribution = Independent(Normal(loc, scale_diag), 1)

        transforms = []
        transforms.append(TanhTransform())
        if not (low is None or high is None):
            transforms.append(AffineTransform(loc=(high + low) / 2, scale=(high - low) / 2))
        transform = ComposeTransform(transforms)

        super().__init__(base_distribution, transform)

    def mode(self) -> torch.Tensor:
        mode = self.loc
        for transform in self.transforms:
            mode = transform(mode)
        return mode

    def stddev(self) -> torch.Tensor:
        return self.transform(self.loc + self.scale_diag) - self.transform(self.loc)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        value = torch.clamp(value, -1 + eps, 1 - eps)
        return super().log_prob(value)


class Policy(nn.Module):
    def __init__(
        self,
        obs_encoded_dim: int,
        network: nn.Module,
        action_dim: int,
        std_parameterization: str = "exp",  # "exp", "softplus", "fixed", or "uniform"
        std_min: Optional[float] = 1e-5,
        std_max: Optional[float] = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.obs_encoded_dim = obs_encoded_dim
        self.network = network
        self.action_dim = action_dim
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std

        self.mean_layer = nn.Linear(network.hidden_dims[-1], action_dim)

        if fixed_std is None:
            if std_parameterization in ["exp", "softplus"]:
                self.std_layer = nn.Linear(network.hidden_dims[-1], action_dim)
            elif std_parameterization == "uniform":
                self.log_stds = nn.Parameter(torch.zeros(action_dim))
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert std_parameterization == "fixed"

        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)

        if fixed_std is None and std_parameterization in ["exp", "softplus"]:
            nn.init.xavier_uniform_(self.std_layer.weight)
            nn.init.zeros_(self.std_layer.bias)

    def forward(self, encoded_observations: torch.Tensor, temperature: float = 1.0) -> Union[TransformedDistribution, Normal]:
        outputs = self.network(encoded_observations)

        means = self.mean_layer(outputs)

        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = self.std_layer(outputs)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = self.std_layer(outputs)
                stds = nn.functional.softplus(stds)
            elif self.std_parameterization == "uniform":
                stds = torch.exp(self.log_stds).expand_as(means)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            stds = self.fixed_std.to(means.device).expand_as(means)

        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature))

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = Normal(loc=means, scale=stds)

        return distribution


class Critics(nn.Module):
    def __init__(
        self,
        obs_encoded_dim: int,
        networks: list[nn.Module],
        num_backbones: int = 2,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        assert len(networks) == num_backbones
        self.obs_encoded_dim = obs_encoded_dim
        self.networks = nn.ModuleList(networks)
        self.num_backbones = num_backbones
        self.init_final = init_final

        self.backbone_output_dims = networks[0].hidden_dims[-1]

        if init_final is not None:
            self.output_layer = nn.Linear(self.backbone_output_dims, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(self.backbone_output_dims, 1)
            nn.init.xavier_uniform_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        encoded_observations: Float[torch.Tensor, "batch {self.obs_encoded_dim}"],
        actions: Float[torch.Tensor, "batch *num_actions action_dim"],
    ) -> Float[torch.Tensor, "{self.num_backbones} batch *num_actions"]:
        if actions.ndim == 3:
            # forward the q function with multiple actions on each state
            encoded_observations = encoded_observations.unsqueeze(1).expand(-1, actions.shape[1], -1)
        # HACK: check dimensions here
        inputs = torch.cat([encoded_observations, actions], dim=-1)

        backbone_outputs = []
        for network in self.networks:
            backbone_outputs.append(network(inputs))
        backbone_outputs: Float[
            torch.Tensor,
            "{self.num_backbones} batch *num_actions {self.backbone_output_dims}",
        ] = torch.stack(backbone_outputs, dim=0)

        value = self.output_layer(backbone_outputs)
        # HACK: check output shape here
        # if actions.ndim == 3:
        #     value = value.squeeze(-1).permute(0, 2, 1)
        # else:
        value = value.squeeze(-1)
        return value  # (num_backbones, batch, *num_actions)


class CalQlConfig(PretrainedConfig):
    moedel_type = "calql"

    def __init__(
        self,
        obs_encoded_dim=2048,
        action_dim=70,
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        actor_wps=2000,
        critic_wps=2000,
        alpha_lr=1e-3,
        cql_alpha=5.0,
        cql_autotune_alpha=False,
        cql_alpha_lagrange_init=0.1,
        cql_target_action_gap=0.5,
        beta_bc_loss: float = 1.0,
        eta_q_loss: float = 0.1,
        **kwargs,
    ):        
        # self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_min = 0
        self.cql_clip_diff_max = np.inf
        self.cql_alpha = cql_alpha
        self.cql_autotune_alpha = cql_autotune_alpha
        self.action_dim = action_dim
        self.target_entropy = -action_dim
        self.obs_encoded_dim = obs_encoded_dim
        self.cql_temperature_init_value = 1.0
        self.critic_ensemble_size = 2
        self.cql_n_actions = 4
        self.cql_max_target_backup = True
        self.policy_network_kwargs = dict(
            input_dim=self.obs_encoded_dim,
            hidden_dims=[256, 256],
            activate_final=True,
            use_layer_norm=False,
        )
        self.critic_network_kwargs = dict(
            input_dim=self.obs_encoded_dim + self.action_dim,
            hidden_dims=[256, 256],
            activate_final=True,
            use_layer_norm=False,
        )
        self.policy_kwargs = dict(
            tanh_squash_distribution=True,
            std_parameterization="exp",
        )
        self.critic_subsample_size = None
        self.cql_max_target_backup = True
        self.backup_entropy = False
        self.discount = 0.98
        self.goal_conditioned = True
        self.gc_kwargs = dict(
            negative_proportion=0.0,
        )
        self.use_td_loss = True
        self.cql_action_sample_method = "uniform"
        self.cql_importance_sample = False
        self.cql_temp = 1.0
        self.use_calql = True

        self.beta_bc_loss = beta_bc_loss
        self.eta_q_loss = eta_q_loss

        self.cql_target_action_gap = cql_target_action_gap
        self.cql_alpha_lagrange_init = cql_alpha_lagrange_init

        self.actor_optimizer_kwargs = dict(
            learning_rate=actor_lr,
            warmup_steps=actor_wps,
        )
        self.critic_optimizer_kwargs = dict(
            learning_rate=critic_lr,
            warmup_steps=critic_wps,
        )
        self.temperature_optimizer_kwargs = dict(learning_rate=temp_lr)
        self.cql_alpha_optimizer_kwargs = dict(learning_rate=alpha_lr)

        super().__init__(**kwargs)


class CalQL(PreTrainedModel):
    config_calss = CalQlConfig

    def __init__(self, config: CalQlConfig):
        super(CalQL, self).__init__(config=config)
        self.config = config

        self.temperature = GeqLagrangeMultiplier(
            init_value=self.config.cql_temperature_init_value,
            constraint_shape=(),
        )

        if self.config.cql_autotune_alpha:
            self.cql_alpha_lagrange = LeqLagrangeMultiplier(
                init_value=self.config.cql_alpha_lagrange_init,
                constraint_shape=(),
                parameterization="softplus",
            )

        self.policy = Policy(
            obs_encoded_dim=self.config.obs_encoded_dim,
            network=MLP(**self.config.policy_network_kwargs),
            action_dim=self.config.action_dim,
            **self.config.policy_kwargs,
        )

        self.critics = Critics(
            obs_encoded_dim=self.config.obs_encoded_dim,
            networks=[MLP(**self.config.critic_network_kwargs) for _ in range(self.config.critic_ensemble_size)],
            num_backbones=self.config.critic_ensemble_size,
        )

        self.target_critics = deepcopy(self.critics)

    def forward_policy_and_sample(
        self,
        encoded_obs: Float[torch.Tensor, "batch {self.config.obs_encoded_dim}"],
        repeat: int = None,
    ):
        action_dist = self.policy.forward(encoded_obs)
        if repeat:
            # new_actions = action_dist.rsample(torch.tensor([repeat]))  # repeat, tensor, act_dim
            new_actions = action_dist.rsample((repeat,))
            log_pi = action_dist.log_prob(new_actions)
            new_actions = new_actions.permute(1, 0, 2)  # (batch, repeat, action_dim)
            log_pi = log_pi.permute(1, 0)  # (batch, repeat)

        else:
            new_actions = action_dist.rsample()  # (batch, action_dim)
            log_pi = action_dist.log_prob(new_actions)  # (batch)
        # NOTE: detach gradient here
        new_actions = new_actions.detach()
        log_pi = log_pi.detach()
        return new_actions, log_pi

    def _compute_next_actions(self, batch: CalQlBatch):
        """
        compute the next actions but with repeat cql_n_actions times
        this should only be used when calculating critic loss using
        cql_max_target_backup
        """
        sample_n_actions = self.config.cql_n_actions if self.config.cql_max_target_backup else None

        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["encoded_next_observations"],
            repeat=sample_n_actions,
        )
        return next_actions, next_actions_log_probs

    def temperature_loss_fn(self, batch: CalQlBatch):
        next_actions, next_actions_log_probs = self._compute_next_actions(batch)

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature.forward(
            lhs=entropy,
            rhs=self.config.target_entropy,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    def cql_alpha_loss_fn(self, batch: CalQlBatch):
        cql_q_diff, _ = self._get_cql_q_diff(batch)
        detached_lhs = cql_q_diff.mean().detach()
        cql_alpha_loss = self.cql_alpha_lagrange.forward(
            lhs=detached_lhs,
            rhs=self.config.cql_target_action_gap,
        )
        alpha = self.cql_alpha_lagrange.forward().detach()
        return cql_alpha_loss, {
            "cql_alpha_loss": cql_alpha_loss.item(),
            "cql_alpha_from_auto_tune": alpha.item(),
        }
    
    def policy_loss_fn(self, batch: CalQlBatch):
        batch_size = batch["rewards"].shape[0]
        temperature = self.temperature.forward().detach()  # detach gradient

        action_distributions = self.policy.forward(batch["encoded_observations"])
        actions = action_distributions.rsample()
        log_probs = action_distributions.log_prob(actions)

        predicted_qs = self.critics.forward(
            batch["encoded_observations"],
            actions,
        ).detach()  # NOTE: detach grads
        predicted_q = predicted_qs.min(dim=0)[0]

        assert predicted_q.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)

        # Q Learning Loss + Behavior Cloning Loss
        # bc_loss = -torch.mean(action_distributions.log_prob(torch.clip(batch["actions"], -0.99, 0.99)))
        # q_learning_loss = -torch.mean(predicted_q) + torch.mean(temperature * log_probs)
        # actor_loss = self.config.beta_bc_loss * bc_loss + self.config.eta_q_loss * q_learning_loss
        
        # origin actor loss 
        bc_loss = -torch.mean(action_distributions.log_prob(torch.clip(batch["actions"], -0.99, 0.99)))
        actor_objective = predicted_q
        actor_loss = -torch.mean(actor_objective) + torch.mean(temperature * log_probs)

        # finetune loss
        # actor_objective = predicted_q
        # actor_loss = -torch.mean(actor_objective)

        info = {
            "actor_nll": self.config.beta_bc_loss * bc_loss,
            "acotr_q" : -torch.mean(predicted_q),
            "acotr_log_probs": torch.mean(temperature * log_probs),
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((actions - batch["actions"]) ** 2).sum(dim=-1).mean(),
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
        }

        return actor_loss, info

    def sac_critic_loss_fn(self, batch: CalQlBatch):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]

        # next_actions, next_actions_log_probs = self._compute_next_actions(batch)

        next_actions = batch["next_actions"]
        next_actions_log_probs = torch.zeros_like(batch["rewards"])

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        with torch.no_grad():
            self.target_critics.eval()
            target_next_qs = self.target_critics.forward(
                batch["encoded_next_observations"],
                next_actions,
            )  # (critic_ensemble_size, batch_size, cql_n_actions)
            self.target_critics.train()

        # Subsample if requested
        if self.config.critic_subsample_size is not None:
            subsample_idcs = torch.randint(
                0,
                self.config.critic_ensemble_size,
                (self.config.critic_ensemble_size,),
                device=target_next_qs.device,
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(dim=0)[0]
        assert target_next_min_q.shape == next_actions_log_probs.shape
        # (batch_size,) for sac, (batch_size, cql_n_actions) for cql

        target_next_min_q = self._process_target_next_qs(
            target_next_min_q,
            next_actions_log_probs,
        )

        target_q = batch["rewards"] + self.config.discount * batch["masks"] * target_next_min_q

        assert target_q.shape == (batch_size,)
        predicted_qs = self.critics.forward(batch["encoded_observations"], batch["actions"])
        assert predicted_qs.shape == (self.config.critic_ensemble_size, batch_size)

        target_qs = target_q.unsqueeze(0).expand(self.config.critic_ensemble_size, -1)
        assert predicted_qs.shape == target_qs.shape
        critic_loss = torch.mean((predicted_qs - target_qs) ** 2)

        info = {
            "td_err": critic_loss,
            "online_q": torch.mean(predicted_qs),
            "target_q": torch.mean(target_qs),
        }

        if self.config.goal_conditioned:
            num_negatives = int(self.config.gc_kwargs["negative_proportion"] * batch_size)
            info["negative_qs"] = torch.mean(predicted_qs, dim=-1)[:num_negatives].mean()
            info["positive_qs"] = torch.mean(predicted_qs, dim=-1)[num_negatives:].mean()

        return critic_loss, info

    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """add cql_max_target_backup option"""

        if self.config.cql_max_target_backup:
            max_target_indices = torch.argmax(target_next_qs, dim=-1, keepdim=True)
            target_next_qs = torch.gather(target_next_qs, -1, max_target_indices).squeeze(-1)
            next_actions_log_probs = torch.gather(next_actions_log_probs, -1, max_target_indices).squeeze(-1)

        target_next_qs = self.sac_process_target_next_qs(
            target_next_qs,
            next_actions_log_probs,
        )

        return target_next_qs

    def sac_process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """classes that inherit this class can add to this function
        e.g. CQL will add the cql_max_target_backup option
        """
        if self.config.backup_entropy:
            temperature = self.forward_temperature()
            target_next_qs = target_next_qs - temperature * next_actions_log_probs

        return target_next_qs

    def critic_loss_fn(self, batch: CalQlBatch):
        """add CQL loss on top of SAC loss"""
        if self.config.use_td_loss:
            td_loss, td_loss_info = self.sac_critic_loss_fn(batch)
        else:
            td_loss, td_loss_info = 0.0, {}

        cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(batch)

        """auto tune cql alpha"""
        if self.config.cql_autotune_alpha:
            alpha = self.cql_alpha_lagrange.forward().detach()
            cql_loss = (cql_q_diff - self.config.cql_target_action_gap).mean()
        else:
            alpha = self.config.cql_alpha
            cql_loss = torch.clip(cql_q_diff, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max).mean()

        critic_loss = td_loss + alpha * cql_loss

        info = {
            **td_loss_info,
            "td_err": td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": alpha,
            "cql_diff": cql_q_diff.mean(),
            **cql_intermediate_results,
        }

        return critic_loss, info

    def _get_cql_q_diff(self, batch: CalQlBatch):
        """
        most of the CQL loss logic is here
        It is needed for both critic_loss_fn and cql_alpha_loss_fn
        """
        batch_size = batch["rewards"].shape[0]

        q_pred = self.critics.forward(batch["encoded_observations"], batch["actions"])
        # HACK: shape changed from jax implementation
        assert q_pred.shape == (self.config.critic_ensemble_size, batch_size)

        """sample random actions"""
        action_dim = batch["actions"].shape[-1]
        if self.config.cql_action_sample_method == "uniform":
            cql_random_actions = (
                torch.rand(
                    (batch_size, self.config.cql_n_actions, action_dim),
                    device=batch["actions"].device,
                )
                * 2.0
                - 1.0
            )
        elif self.config.cql_action_sample_method == "normal":
            cql_random_actions = torch.randn(
                (batch_size, self.config.cql_n_actions, action_dim),
                device=batch["actions"].device,
            )
        else:
            raise NotImplementedError

        cql_current_actions, cql_current_log_pis = self.forward_policy_and_sample(
            batch["encoded_observations"],
            repeat=self.config.cql_n_actions,
        )
        assert cql_current_log_pis.shape == (batch_size, self.config.cql_n_actions)

        cql_next_actions, cql_next_log_pis = self.forward_policy_and_sample(
            batch["encoded_next_observations"],
            repeat=self.config.cql_n_actions,
        )

        all_sampled_actions = torch.cat(
            [
                cql_random_actions,
                cql_current_actions,
                cql_next_actions,
            ],
            dim=1,
        )

        """q values of randomly sampled actions"""
        cql_q_samples = self.critics.forward(batch["encoded_observations"], all_sampled_actions)

        # print(f"DEBUG >>> Min/Max of cql_q_samples before logsumexp: {cql_q_samples.min().item():.2f}, {cql_q_samples.max().item():.2f}")
    
        # HACK: shape changed from jax implementation
        assert cql_q_samples.shape == (
            self.config.critic_ensemble_size,
            batch_size,
            self.config.cql_n_actions * 3,
        )

        if self.config.critic_subsample_size is not None:
            subsample_idcs = torch.randint(
                0,
                self.config.critic_ensemble_size,
                (self.config.critic_ensemble_size,),
                device=cql_q_samples.device,
            )
            cql_q_samples = cql_q_samples[subsample_idcs]

        """Cal-QL"""
        if self.config.use_calql:
            # HACK: check shape of mc_returns
            mc_lower_bound = batch["mc_returns"].reshape(-1, 1).repeat(1, self.config.cql_n_actions * 2)
            assert mc_lower_bound.shape == (
                batch_size,
                self.config.cql_n_actions * 2,
            )

            cql_q_pi = cql_q_samples[:, :, self.config.cql_n_actions :]
            num_vals = cql_q_pi.numel()
            calql_bound_rate = torch.sum((cql_q_pi < mc_lower_bound).float()) / num_vals
            cql_q_pi = torch.maximum(cql_q_pi, mc_lower_bound)
            cql_q_samples = torch.cat(
                [
                    cql_q_samples[:, :, : self.config.cql_n_actions],
                    cql_q_pi,
                ],
                dim=-1,
            )

        if self.config.cql_importance_sample:
            random_density = torch.log(torch.tensor(0.5**action_dim, device=cql_q_samples.device))
            importance_prob = torch.cat(
                [
                    random_density.expand(batch_size, self.config.cql_n_actions),
                    cql_current_log_pis,
                    cql_next_log_pis,
                ],
                dim=1,
            )
            # HACK: check dim
            cql_q_samples = cql_q_samples - importance_prob.unsqueeze(0)
        else:
            cql_q_samples = torch.cat([cql_q_samples, q_pred.unsqueeze(-1)], dim=-1)

            cql_q_samples -= (
                torch.log(
                    torch.tensor(
                        cql_q_samples.shape[-1],
                        dtype=torch.float,
                        device=cql_q_samples.device,
                    )
                )
                * self.config.cql_temp
            )
            # HACK: shape diff from jax implementation
            assert cql_q_samples.shape == (
                self.config.critic_ensemble_size,
                batch_size,
                3 * self.config.cql_n_actions + 1,
            )

        """log sum exp of the ood actions"""
        cql_ood_values = torch.logsumexp(cql_q_samples / self.config.cql_temp, dim=-1) * self.config.cql_temp
        assert cql_ood_values.shape == (self.config.critic_ensemble_size, batch_size)

        cql_q_diff = cql_ood_values - q_pred
        info = {
            "cql_ood_values": cql_ood_values.mean(),
        }
        if self.config.use_calql:
            info["calql_bound_rate"] = calql_bound_rate

        return cql_q_diff, info

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
            optimizer = AdamW(
                params=params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            optimizer = Adam(params=params, lr=learning_rate)

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps

            if cosine_decay_steps is not None:
                decay_step = step - warmup_steps
                if decay_step < 0:
                    return 0.0
                if decay_step >= cosine_decay_steps:
                    return 0.0
                progress = decay_step / cosine_decay_steps
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

        if return_lr_schedule:
            return optimizer, scheduler
        else:
            return optimizer

    def forward(self, batch: CalQlBatch):
        temperature_loss, temperature_loss_info = self.temperature_loss_fn(batch)
        policy_loss, policy_loss_info = self.policy_loss_fn(batch)
        critic_loss, critic_loss_info = self.critic_loss_fn(batch)

        losses = {
            "temperature": temperature_loss,
            "policy": policy_loss,
            "critic": critic_loss,
        }
        
        info = {
            **temperature_loss_info,
            **policy_loss_info,
            **critic_loss_info,
        }
        
        if self.config.cql_autotune_alpha:
            cql_alpha_loss, cql_alpha_loss_info = self.cql_alpha_loss_fn(batch)
            losses["cql_alpha"] = cql_alpha_loss
            info.update(cql_alpha_loss_info)

        return losses, info

    @jaxtyped(typechecker=typechecker)
    def get_q_values(
        self,
        encoded_observations: Float[torch.Tensor, "batch {self.config.obs_encoded_dim}"],
        noise_actions: Float[torch.Tensor, "batch num_actions action_dim"],
    ) -> Float[torch.Tensor, "batch num_actions"]:
        # (num_backbones, batch, *num_actions)
        q_values = self.target_critics.forward(encoded_observations, noise_actions)
        q_values = q_values.min(dim=0)[0]
        return q_values


class ValueQueryHead(nn.Module):
    def __init__(self, config, tokenizer, pretrained_smolvla_path: str):
        super().__init__()
        self.config = config

        # Load and Frozen SmolVLA Feature Extractor
        self.smolvla_policy = SmolVLAPolicy.from_pretrained(pretrained_smolvla_path)
        self.feature_extractor = self.smolvla_policy.model 
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        self.vqh_backbone = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )
        self.vqh_backbone.gradient_checkpointing_enable()

        l1_hidden_size = self.feature_extractor.vlm_with_expert.vlm.config.text_config.hidden_size # Should be 576
        l2_hidden_size = self.vqh_backbone.config.text_config.hidden_size # Should be 960
        
        self.embedding_proj = nn.Linear(l1_hidden_size, l2_hidden_size)

        self.tokenizer = tokenizer 
        self.fake_image_token = tokenizer.fake_image_token_id
        self.global_image_token = tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

        vlm_hidden_size = self.vqh_backbone.config.text_config.hidden_size
        action_dim_flat = config.vqh_chunk_size * config.action_feature.shape[0]

        cal_ql_config = CalQlConfig(
            obs_encoded_dim=vlm_hidden_size,
            action_dim=action_dim_flat,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            temp_lr=config.temp_lr,
            alpha_lr=config.alpha_lr,
            cql_alpha=config.cql_alpha,
            cql_autotune_alpha=config.cql_autotune_alpha,
            cql_alpha_lagrange_init=config.cql_alpha_lagrange_init,
            cql_target_action_gap=config.cql_target_action_gap,
            beta_bc_loss=config.beta_bc_loss,
            eta_q_loss=config.eta_q_loss,
        )
         
        self.calql = CalQL(config=cal_ql_config)

    @torch.no_grad()
    def _get_features_from_l1_extractor(
        self,
        images: list[Float[Tensor, "batch 3 512 512"]],
        img_masks: list[Bool[Tensor, " batch"]],
        lang_tokens: Int64[Tensor, "batch seq_len"],
        lang_masks: Bool[Tensor, "batch seq_len"],
        state,
    ) -> tuple[Float[Tensor, "batch seq_len hidden_dim"], Bool[Tensor, "batch seq_len"]]:
        embs, prefix_pad_masks, att_masks = self.feature_extractor.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        batch_size = embs.shape[0]
        chunk_size = self.feature_extractor.config.chunk_size
        expert_hidden_size = self.feature_extractor.vlm_with_expert.expert_hidden_size
        
        dummy_suffix = torch.zeros(
            (batch_size, chunk_size, expert_hidden_size),
            dtype=embs.dtype,
            device=embs.device
        )
        suffix_pad_masks = torch.ones((batch_size, chunk_size), dtype=torch.bool, device=embs.device)

        full_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        suffix_att_masks = torch.ones((batch_size, chunk_size), dtype=torch.bool, device=embs.device)
        full_att_masks = torch.cat([att_masks, suffix_att_masks], dim=1)

        position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        att_2d_masks = make_att_2d_masks(full_pad_masks, full_att_masks)
        
        (prefix_out, _), _ = self.feature_extractor.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=[embs, dummy_suffix],
            use_cache=False,
            fill_kv_cache=False,
        )

        return prefix_out, prefix_pad_masks
 
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
        
        with torch.no_grad():
            cur_intermediate_embs, cur_pad_masks = self._get_features_from_l1_extractor(
                images, img_masks, lang_tokens, lang_masks, state
            )
            nex_intermediate_embs, nex_pad_masks = self._get_features_from_l1_extractor(
                vqh_images, vqh_img_masks, lang_tokens, lang_masks, vqh_state
            )
        
        cur_projected_embs = self.embedding_proj(cur_intermediate_embs)
        nex_projected_embs = self.embedding_proj(nex_intermediate_embs)


        batch_size, seq_len = cur_projected_embs.shape[:2]
        dummy_input_ids = torch.zeros(
            (batch_size, seq_len), 
            dtype=torch.long, 
            device=cur_projected_embs.device
        )
        
        cur_outputs = self.vqh_backbone.model.forward(
            input_ids=dummy_input_ids,
            inputs_embeds=cur_projected_embs.to(self.vqh_backbone.dtype),
            attention_mask=cur_pad_masks,
            return_dict=True,
        )
        cur_hidden_states = cur_outputs.last_hidden_state


        batch_size, seq_len = nex_projected_embs.shape[:2]
        dummy_next_input_ids = torch.zeros(
            (batch_size, seq_len), 
            dtype=torch.long, 
            device=nex_projected_embs.device
        )
        
        nex_outputs = self.vqh_backbone.model.forward(
            input_ids=dummy_next_input_ids,
            inputs_embeds=nex_projected_embs.to(self.vqh_backbone.dtype),
            attention_mask=nex_pad_masks,
            return_dict=True,
        )
        nex_hidden_states = nex_outputs.last_hidden_state

        batch_size = cur_hidden_states.shape[0]
        device = cur_hidden_states.device

        cur_last_token_indices = cur_pad_masks.sum(dim=1) - 1
        nex_last_token_indices = nex_pad_masks.sum(dim=1) - 1

        cur_input_embs = cur_hidden_states[torch.arange(batch_size, device=device), cur_last_token_indices]
        nex_input_embs = nex_hidden_states[torch.arange(batch_size, device=device), nex_last_token_indices]

        cal_ql_batch: CalQlBatch = dict(
            encoded_observations=cur_input_embs.to(dtype=torch.float32),
            encoded_next_observations=nex_input_embs.to(dtype=torch.float32),
            actions=actions.view(actions.shape[0], -1),
            next_actions=next_actions.view(next_actions.shape[0], -1),
            rewards=rewards,
            mc_returns=mc_returns,
            masks=masks,
        )

        losses_dict, calql_log_dict = self.calql(cal_ql_batch)
                      
        log_dict = {}
        for key, value in calql_log_dict.items():
            if hasattr(value, "item"):
                log_dict[key] = value.mean().item() if value.numel() > 1 else value.item()
            else:
                log_dict[key] = value 
        return losses_dict, log_dict

    @jaxtyped(typechecker=typechecker)
    def select_q_actions(
        self,
        images: list[Float[Tensor, "Batch 3 512 512"]],
        img_masks: list[Bool[Tensor, " Batch"]],
        lang_tokens: Int64[Tensor, "Batch seq_len"],
        lang_masks: Bool[Tensor, "Batch seq_len"],
        state,
        noise_actions,
    )-> tuple[Int64[Tensor, " Batch"], Float[Tensor, "Batch candidates_num"]]:
        batch_size = noise_actions.shape[0]
        candidates_num = noise_actions.shape[1]
        device = noise_actions.device

        intermediate_embs, pad_masks = self._get_features_from_l1_extractor(
            images, img_masks, lang_tokens, lang_masks, state
        )

        projected_embs = self.embedding_proj(intermediate_embs)

        seq_len = projected_embs.shape[1]
        dummy_input_ids = torch.zeros(
            (batch_size, seq_len), 
            dtype=torch.long, 
            device=device
        )

        outputs = self.vqh_backbone.model.forward(
            input_ids=dummy_input_ids,
            inputs_embeds=projected_embs.to(self.vqh_backbone.dtype),
            attention_mask=pad_masks,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        last_token_indices = pad_masks.sum(dim=1) - 1
        query_embedding = hidden_states[torch.arange(batch_size, device=device), last_token_indices]

        flat_noise_actions = noise_actions.reshape(batch_size, candidates_num, -1)
        q_values = self.calql.get_q_values(
            query_embedding.to(dtype=torch.float32), 
            flat_noise_actions
        )

        action_index = torch.argmax(q_values, dim=1)
        return action_index, q_values

    @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def sample_calql_policy_action(
        self,
        images: list[Float[Tensor, "Batch 3 512 512"]],
        img_masks: list[Bool[Tensor, " Batch"]],
        lang_tokens: Int64[Tensor, "Batch seq_len"],
        lang_masks: Bool[Tensor, "Batch seq_len"],
        state,
        num_samples: int = 1,
    ) -> Float[Tensor, "Batch num_samples chunk_size action_feature_dim"]:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        intermediate_embs, pad_masks = self._get_features_from_l1_extractor(
            images, img_masks, lang_tokens, lang_masks, state
        )
        projected_embs = self.embedding_proj(intermediate_embs)

        seq_len = projected_embs.shape[1]
        dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        outputs = self.vqh_backbone.model.forward(
            input_ids=dummy_input_ids,
            inputs_embeds=projected_embs.to(self.vqh_backbone.dtype),
            attention_mask=pad_masks,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        last_token_indices = pad_masks.sum(dim=1) - 1
        query_embedding = hidden_states[torch.arange(batch_size, device=device), last_token_indices]

        calql_actions, _ = self.calql.forward_policy_and_sample(
            query_embedding.to(dtype=torch.float32),
            repeat=num_samples if num_samples > 1 else None 
        )

        action_chunk_size = self.config.vqh_chunk_size
        action_feature_dim = self.config.action_feature.shape[0]
        
        reshaped_actions = calql_actions.view(
            batch_size, 
            num_samples, 
            action_chunk_size, 
            action_feature_dim
        )
        return reshaped_actions