from __future__ import annotations

import torch
from torch import nn


class InputAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.missing_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.missing_embedding, std=0.02)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        missing_mask = input_features.abs().sum(dim=-1, keepdim=True) <= 1e-12
        projected = self.projection(self.normalization(input_features))
        missing_embedding = self.missing_embedding.unsqueeze(0).expand_as(projected)
        projected = torch.where(missing_mask, missing_embedding, projected)
        return self.output_normalization(projected)



class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.normalization = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale = nn.Parameter(torch.full((hidden_dim,), 1e-2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = self.feed_forward(self.normalization(features))
        return features + self.layer_scale * update
