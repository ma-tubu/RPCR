from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.feature_v2a2t import FeatureV2A2T, FeatureV2A2TOutput


class MaskedBiGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if sequence_dim % 2 != 0:
            raise ValueError("sequence_dim must be even for the bidirectional GRU.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.input_normalization = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, sequence_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=sequence_dim,
            hidden_size=sequence_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(sequence_dim, sequence_dim // 2),
            nn.Tanh(),
            nn.Linear(sequence_dim // 2, 1),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(sequence_dim * 3),
            nn.Linear(sequence_dim * 3, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        maximum_length = features.shape[1]
        lengths = lengths.to(dtype=torch.long).clamp(min=1, max=maximum_length)
        projected = self.input_projection(self.input_normalization(features))
        packed = pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.gru(packed)
        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=maximum_length,
        )
        positions = torch.arange(maximum_length, device=features.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)

        attention_scores = self.attention(encoded).squeeze(-1)
        attention_scores = attention_scores.masked_fill(
            ~mask, torch.finfo(attention_scores.dtype).min
        )
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_pool = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)
        mean_pool = torch.sum(encoded * mask.unsqueeze(-1), dim=1) / lengths.unsqueeze(
            -1
        ).to(encoded.dtype)
        final_state = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.output_projection(
            torch.cat([attention_pool, mean_pool, final_state], dim=-1)
        )


class CHSIMSUnalignedV2A2T(nn.Module):
    def __init__(
        self,
        *,
        sequence_dim: int = 128,
        sequence_output_dim: int = 128,
        sequence_layers: int = 1,
        sequence_dropout: float = 0.2,
        output_activation: str = "tanh",
        chain_config: Dict[str, object],
    ) -> None:
        super().__init__()
        if output_activation not in {"identity", "tanh"}:
            raise ValueError(f"Unsupported output activation: {output_activation}")
        self.output_activation = output_activation
        self.visual_sequence_encoder = MaskedBiGRUEncoder(
            709,
            sequence_dim,
            sequence_output_dim,
            sequence_layers,
            sequence_dropout,
        )
        self.audio_sequence_encoder = MaskedBiGRUEncoder(
            33,
            sequence_dim,
            sequence_output_dim,
            sequence_layers,
            sequence_dropout,
        )
        self.text_sequence_encoder = MaskedBiGRUEncoder(
            768,
            sequence_dim,
            sequence_output_dim,
            sequence_layers,
            sequence_dropout,
        )
        self.chain = FeatureV2A2T(
            visual_dim=sequence_output_dim,
            audio_dim=sequence_output_dim,
            text_dim=sequence_output_dim,
            **chain_config,
        )

    def forward(
        self,
        *,
        vision: torch.Tensor,
        vision_lengths: torch.Tensor,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> FeatureV2A2TOutput:
        visual_features = self.visual_sequence_encoder(vision, vision_lengths)
        audio_features = self.audio_sequence_encoder(audio, audio_lengths)
        text_features = self.text_sequence_encoder(text, text_lengths)
        output = self.chain(visual_features, audio_features, text_features)
        if self.output_activation == "identity":
            return output
        return FeatureV2A2TOutput(
            prediction=torch.tanh(output.prediction),
            auxiliary_predictions={
                name: torch.tanh(prediction)
                for name, prediction in output.auxiliary_predictions.items()
            },
            router_loss=output.router_loss,
            route_statistics=output.route_statistics,
            features=output.features,
        )
