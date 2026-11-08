from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .adapters import ResidualFeedForwardBlock


class ConditionalRefinementBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        expert_dim: int,
        expansion: int,
        dropout: float,
        modulation_mode: str = "full",
        routing_mode: str = "dynamic",
        prompt_mode: str = "both",
        gate_mode: str = "learned",
        condition_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        if modulation_mode not in {"full", "scale_only", "shift_only", "none"}:
            raise ValueError(f"Unsupported modulation mode: {modulation_mode}")
        if routing_mode not in {
            "dynamic",
            "uniform",
            "static_learned",
            "target_only",
            "context_only",
        }:
            raise ValueError(f"Unsupported routing mode: {routing_mode}")
        if prompt_mode not in {"both", "dynamic_only", "static_only", "none"}:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        if gate_mode not in {"learned", "open", "closed"}:
            raise ValueError(f"Unsupported gate mode: {gate_mode}")
        if expert_dim <= 0:
            raise ValueError("expert_dim must be positive.")
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.modulation_mode = modulation_mode
        self.routing_mode = routing_mode
        self.prompt_mode = prompt_mode
        self.gate_mode = gate_mode
        self.feature_normalization = nn.LayerNorm(hidden_dim)
        self.context_normalization = nn.LayerNorm(hidden_dim)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        if routing_mode == "static_learned":
            self.static_route_logits = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_parameter("static_route_logits", None)
        self.experts = nn.Parameter(torch.empty(num_experts, expert_dim))
        self.expert_projection = (
            nn.Identity()
            if expert_dim == hidden_dim
            else nn.Linear(expert_dim, hidden_dim, bias=False)
        )
        self.static_prompt = nn.Parameter(torch.empty(hidden_dim))
        self.context_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.condition_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_scale = nn.Parameter(
            torch.full((hidden_dim,), float(condition_scale_init))
        )
        self.feed_forward = ResidualFeedForwardBlock(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )
        nn.init.normal_(self.experts, std=0.02)
        nn.init.normal_(self.static_prompt, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        route_scores_override: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_features = self.feature_normalization(features)
        normalized_context = self.context_normalization(context)
        joined = torch.cat([normalized_features, normalized_context], dim=-1)

        if route_scores_override is not None:
            route_scores = route_scores_override
        elif self.routing_mode == "dynamic":
            route_scores = torch.softmax(self.router(joined), dim=-1)
        elif self.routing_mode == "uniform":
            route_scores = joined.new_full(
                (joined.shape[0], self.num_experts), 1.0 / self.num_experts
            )
        elif self.routing_mode == "static_learned":
            if self.static_route_logits is None:
                raise RuntimeError("static_route_logits is missing.")
            route_scores = torch.softmax(self.static_route_logits, dim=-1).unsqueeze(0)
            route_scores = route_scores.expand(joined.shape[0], -1)
        elif self.routing_mode == "target_only":
            router_input = torch.cat(
                [normalized_features, torch.zeros_like(normalized_context)], dim=-1
            )
            route_scores = torch.softmax(self.router(router_input), dim=-1)
        else:
            router_input = torch.cat(
                [torch.zeros_like(normalized_features), normalized_context], dim=-1
            )
            route_scores = torch.softmax(self.router(router_input), dim=-1)
        dynamic_prompt = self.expert_projection(
            torch.matmul(route_scores, self.experts)
        )

        modulation_scale, modulation_shift = self.context_modulation(
            normalized_context
        ).chunk(2, dim=-1)
        if self.modulation_mode == "full":
            modulated_features = normalized_features * (
                1.0 + 0.1 * torch.tanh(modulation_scale)
            ) + modulation_shift
        elif self.modulation_mode == "scale_only":
            modulated_features = normalized_features * (
                1.0 + 0.1 * torch.tanh(modulation_scale)
            )
        elif self.modulation_mode == "shift_only":
            modulated_features = normalized_features + modulation_shift
        else:
            modulated_features = normalized_features
        conditioned = modulated_features
        if self.prompt_mode in {"both", "dynamic_only"}:
            conditioned = conditioned + dynamic_prompt
        if self.prompt_mode in {"both", "static_only"}:
            conditioned = conditioned + self.static_prompt.unsqueeze(0)
        conditioned = self.condition_projection(conditioned)
        if self.gate_mode == "learned":
            gate = torch.sigmoid(self.condition_gate(joined))
        elif self.gate_mode == "open":
            gate = torch.ones_like(features)
        else:
            gate = torch.zeros_like(features)
        features = features + self.condition_scale * gate * conditioned
        features = self.feed_forward(features)

        importance = route_scores.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance.square()) - 1.0
        return features, balance_loss, route_scores
