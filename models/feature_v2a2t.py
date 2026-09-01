from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeatureV2A2TOutput:
    prediction: torch.Tensor
    auxiliary_predictions: Dict[str, torch.Tensor]
    router_loss: torch.Tensor
    route_statistics: Dict[str, torch.Tensor]
    features: Dict[str, torch.Tensor]


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


class CrossModalMoPEBlock(nn.Module):
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        entropy = -(route_scores * torch.log(route_scores.clamp_min(1e-8))).sum(
            dim=-1
        ).mean()
        return features, balance_loss, entropy, route_scores


class FeatureV2A2T(nn.Module):
    def __init__(
        self,
        visual_dim: int = 1024,
        audio_dim: int = 512,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        visual_depth: int = 3,
        audio_depth: int = 4,
        text_depth: int = 4,
        num_experts: int = 8,
        expert_dim: int | None = None,
        expansion: int = 4,
        dropout: float = 0.2,
        architecture_mode: str = "chain",
        chain_order: str = "v2a2t",
        unimodal_modality: str = "text",
        direct_fusion_dim: int | None = None,
        direct_fusion_depth: int = 2,
        audio_encoder_mode: str = "mope",
        text_encoder_mode: str = "mope",
        audio_modulation_mode: str = "full",
        text_modulation_mode: str = "full",
        audio_context_mode: str = "real",
        text_context_mode: str = "real",
        audio_routing_mode: str = "dynamic",
        text_routing_mode: str = "dynamic",
        audio_route_sharing: str = "per_layer",
        text_route_sharing: str = "per_layer",
        audio_prompt_mode: str = "both",
        text_prompt_mode: str = "both",
        audio_gate_mode: str = "learned",
        text_gate_mode: str = "learned",
        va_fusion_mode: str = "gated_second_order",
        audio_ffn_expansion: int | None = None,
        text_ffn_expansion: int | None = None,
        audio_condition_scale_init: float = 1e-2,
        text_condition_scale_init: float = 1e-2,
        audio_condition_scale_growth: float = 1.0,
        text_condition_scale_growth: float = 1.0,
    ) -> None:
        super().__init__()
        if architecture_mode not in {"chain", "direct_concat", "unimodal"}:
            raise ValueError(f"Unsupported architecture mode: {architecture_mode}")
        valid_chain_orders = {
            "v2a2t",
            "v2t2a",
            "a2v2t",
            "a2t2v",
            "t2v2a",
            "t2a2v",
        }
        if chain_order not in valid_chain_orders:
            raise ValueError(f"Unsupported chain order: {chain_order}")
        if unimodal_modality not in {"audio", "visual", "text"}:
            raise ValueError(f"Unsupported unimodal modality: {unimodal_modality}")
        valid_encoder_modes = {"mope", "ffn", "skip"}
        if audio_encoder_mode not in valid_encoder_modes:
            raise ValueError(f"Unsupported audio encoder mode: {audio_encoder_mode}")
        if text_encoder_mode not in valid_encoder_modes:
            raise ValueError(f"Unsupported text encoder mode: {text_encoder_mode}")
        if audio_condition_scale_init < 0 or text_condition_scale_init < 0:
            raise ValueError("Condition scale initialization must be non-negative.")
        if audio_condition_scale_growth <= 0 or text_condition_scale_growth <= 0:
            raise ValueError("Condition scale growth must be positive.")
        valid_context_modes = {"real", "shuffled", "zero", "mean"}
        if audio_context_mode not in valid_context_modes:
            raise ValueError(f"Unsupported audio context mode: {audio_context_mode}")
        if text_context_mode not in valid_context_modes:
            raise ValueError(f"Unsupported text context mode: {text_context_mode}")
        valid_route_sharing = {"per_layer", "shared_first"}
        if audio_route_sharing not in valid_route_sharing:
            raise ValueError(f"Unsupported audio route sharing: {audio_route_sharing}")
        if text_route_sharing not in valid_route_sharing:
            raise ValueError(f"Unsupported text route sharing: {text_route_sharing}")
        valid_va_fusion_modes = {"gated_second_order", "concat_mlp", "add"}
        if va_fusion_mode not in valid_va_fusion_modes:
            raise ValueError(f"Unsupported VA fusion mode: {va_fusion_mode}")
        if direct_fusion_depth <= 0:
            raise ValueError("direct_fusion_depth must be positive.")
        self.audio_encoder_mode = audio_encoder_mode
        self.text_encoder_mode = text_encoder_mode
        self.architecture_mode = architecture_mode
        self.chain_order = chain_order
        self.unimodal_modality = unimodal_modality
        self.audio_context_mode = audio_context_mode
        self.text_context_mode = text_context_mode
        self.audio_route_sharing = audio_route_sharing
        self.text_route_sharing = text_route_sharing
        self.va_fusion_mode = va_fusion_mode
        resolved_expert_dim = hidden_dim if expert_dim is None else expert_dim
        if resolved_expert_dim <= 0:
            raise ValueError("expert_dim must be positive.")
        if architecture_mode != "unimodal" or unimodal_modality == "visual":
            self.visual_adapter = InputAdapter(visual_dim, hidden_dim, dropout)
        if architecture_mode != "unimodal" or unimodal_modality == "audio":
            self.audio_adapter = InputAdapter(audio_dim, hidden_dim, dropout)
        if architecture_mode != "unimodal" or unimodal_modality == "text":
            self.text_adapter = InputAdapter(text_dim, hidden_dim, dropout)

        if architecture_mode == "unimodal":
            self.unimodal_encoder = nn.ModuleList(
                ResidualFeedForwardBlock(hidden_dim, expansion, dropout)
                for _ in range(visual_depth)
            )
            self.unimodal_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.visual_encoder = nn.ModuleList(
            ResidualFeedForwardBlock(hidden_dim, expansion, dropout)
            for _ in range(visual_depth if architecture_mode == "chain" else 0)
        )
        if architecture_mode != "chain":
            self.audio_encoder = nn.ModuleList()
        elif audio_encoder_mode == "mope":
            self.audio_encoder = nn.ModuleList(
                CrossModalMoPEBlock(
                    hidden_dim,
                    num_experts,
                    resolved_expert_dim,
                    expansion,
                    dropout,
                    modulation_mode=audio_modulation_mode,
                    routing_mode=audio_routing_mode,
                    prompt_mode=audio_prompt_mode,
                    gate_mode=audio_gate_mode,
                    condition_scale_init=(
                        audio_condition_scale_init
                        * audio_condition_scale_growth**layer_index
                    ),
                )
                for layer_index in range(audio_depth)
            )
        elif audio_encoder_mode == "ffn":
            resolved_audio_ffn_expansion = (
                expansion if audio_ffn_expansion is None else audio_ffn_expansion
            )
            self.audio_encoder = nn.ModuleList(
                ResidualFeedForwardBlock(hidden_dim, resolved_audio_ffn_expansion, dropout)
                for _ in range(audio_depth)
            )
        else:
            self.audio_encoder = nn.ModuleList()

        if architecture_mode != "chain":
            self.text_encoder = nn.ModuleList()
        elif text_encoder_mode == "mope":
            self.text_encoder = nn.ModuleList(
                CrossModalMoPEBlock(
                    hidden_dim,
                    num_experts,
                    resolved_expert_dim,
                    expansion,
                    dropout,
                    modulation_mode=text_modulation_mode,
                    routing_mode=text_routing_mode,
                    prompt_mode=text_prompt_mode,
                    gate_mode=text_gate_mode,
                    condition_scale_init=(
                        text_condition_scale_init
                        * text_condition_scale_growth**layer_index
                    ),
                )
                for layer_index in range(text_depth)
            )
        elif text_encoder_mode == "ffn":
            resolved_text_ffn_expansion = (
                expansion if text_ffn_expansion is None else text_ffn_expansion
            )
            self.text_encoder = nn.ModuleList(
                ResidualFeedForwardBlock(hidden_dim, resolved_text_ffn_expansion, dropout)
                for _ in range(text_depth)
            )
        else:
            self.text_encoder = nn.ModuleList()

        pair_dim = hidden_dim * 4
        resolved_direct_dim = (
            hidden_dim * 2 if direct_fusion_dim is None else direct_fusion_dim
        )
        if resolved_direct_dim <= 0:
            raise ValueError("direct_fusion_dim must be positive.")
        if architecture_mode == "chain":
            if va_fusion_mode == "gated_second_order":
                self.va_candidate = nn.Sequential(
                    nn.LayerNorm(pair_dim),
                    nn.Linear(pair_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                self.va_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )
            elif va_fusion_mode == "concat_mlp":
                self.va_concat_fusion = nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
            self.va_normalization = nn.LayerNorm(hidden_dim)
            self.final_head = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif architecture_mode == "direct_concat":
            direct_layers = [
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, resolved_direct_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(direct_fusion_depth - 1):
                direct_layers.extend(
                    [
                        nn.Linear(resolved_direct_dim, resolved_direct_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
            direct_layers.append(nn.Linear(resolved_direct_dim, 1))
            self.direct_head = nn.Sequential(*direct_layers)
        self.auxiliary_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for name in (("visual", "audio", "va", "text") if architecture_mode == "chain" else ())
            }
        )

    @staticmethod
    def parse_chain_order(chain_order: str) -> Tuple[str, str, str]:
        alias = {"v": "visual", "a": "audio", "t": "text"}
        tokens = chain_order.split("2")
        if len(tokens) != 3 or set(tokens) != set(alias):
            raise ValueError(f"Invalid chain order: {chain_order}")
        return tuple(alias[token] for token in tokens)  # type: ignore[return-value]

    @staticmethod
    def pair_features(
        primary: torch.Tensor,
        secondary: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                primary,
                secondary,
                primary * secondary,
                torch.abs(primary - secondary),
            ],
            dim=-1,
        )

    @staticmethod
    def feature_change_statistics(
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            before_float = before.detach().float()
            after_float = after.detach().float()
            relative_change = (
                torch.linalg.vector_norm(after_float - before_float, dim=-1)
                / torch.linalg.vector_norm(before_float, dim=-1).clamp_min(1e-8)
            ).mean()
            cosine_similarity = F.cosine_similarity(
                before_float, after_float, dim=-1, eps=1e-8
            ).mean()
        return relative_change, cosine_similarity

    @staticmethod
    def mean_condition_scale(
        blocks: nn.ModuleList,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        scales = [
            block.condition_scale.detach().abs().mean()
            for block in blocks
            if isinstance(block, CrossModalMoPEBlock)
        ]
        return torch.stack(scales).mean() if scales else reference.new_zeros(())

    @staticmethod
    def perturb_context(context: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply deterministic within-batch context controls without changing parameters."""
        if mode == "real":
            return context
        if mode == "shuffled":
            return torch.roll(context, shifts=1, dims=0) if context.shape[0] > 1 else context
        if mode == "zero":
            return torch.zeros_like(context)
        if mode == "mean":
            return context.mean(dim=0, keepdim=True).expand_as(context)
        raise ValueError(f"Unsupported context mode: {mode}")

    def set_context_modes(self, *, audio: str, text: str) -> None:
        valid_modes = {"real", "shuffled", "zero", "mean"}
        if audio not in valid_modes or text not in valid_modes:
            raise ValueError(f"Unsupported context modes: audio={audio}, text={text}")
        self.audio_context_mode = audio
        self.text_context_mode = text

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> FeatureV2A2TOutput:
        if self.architecture_mode == "unimodal":
            raw_features = {
                "visual": visual_features,
                "audio": audio_features,
                "text": text_features,
            }
            adapter = getattr(self, f"{self.unimodal_modality}_adapter")
            unimodal_hidden = adapter(raw_features[self.unimodal_modality])
            before_encoder = unimodal_hidden
            for block in self.unimodal_encoder:
                unimodal_hidden = block(unimodal_hidden)
            prediction = self.unimodal_head(unimodal_hidden).squeeze(-1)
            relative_change, cosine_similarity = self.feature_change_statistics(
                before_encoder, unimodal_hidden
            )
            zero = prediction.new_zeros(())
            return FeatureV2A2TOutput(
                prediction=prediction,
                auxiliary_predictions={},
                router_loss=zero,
                route_statistics={
                    "audio_entropy": zero,
                    "text_entropy": zero,
                    "va_gate_mean": zero,
                    "audio_relative_change": relative_change,
                    "audio_cosine_similarity": cosine_similarity,
                    "text_relative_change": zero,
                    "text_cosine_similarity": prediction.new_ones(()),
                    "audio_condition_scale_mean": zero,
                    "text_condition_scale_mean": zero,
                },
                features={self.unimodal_modality: unimodal_hidden},
            )

        adapted_features = {
            "visual": self.visual_adapter(visual_features),
            "audio": self.audio_adapter(audio_features),
            "text": self.text_adapter(text_features),
        }
        if self.architecture_mode == "direct_concat":
            visual_hidden = adapted_features["visual"]
            audio_hidden = adapted_features["audio"]
            text_hidden = adapted_features["text"]
            va_hidden = 0.5 * (visual_hidden + audio_hidden)
            prediction = self.direct_head(
                torch.cat([visual_hidden, audio_hidden, text_hidden], dim=-1)
            ).squeeze(-1)
            zero = prediction.new_zeros(())
            return FeatureV2A2TOutput(
                prediction=prediction,
                auxiliary_predictions={},
                router_loss=zero,
                route_statistics={
                    "audio_entropy": zero,
                    "text_entropy": zero,
                    "va_gate_mean": zero,
                    "audio_relative_change": zero,
                    "audio_cosine_similarity": prediction.new_ones(()),
                    "text_relative_change": zero,
                    "text_cosine_similarity": prediction.new_ones(()),
                    "audio_condition_scale_mean": zero,
                    "text_condition_scale_mean": zero,
                },
                features={
                    "visual": visual_hidden,
                    "audio": audio_hidden,
                    "va": va_hidden,
                    "text": text_hidden,
                },
            )

        first_modality, second_modality, third_modality = self.parse_chain_order(
            self.chain_order
        )
        # These local names denote chain stages for checkpoint compatibility:
        # visual_hidden=source, audio_hidden=middle target, text_hidden=final target.
        visual_hidden = adapted_features[first_modality]
        for block in self.visual_encoder:
            visual_hidden = block(visual_hidden)

        audio_hidden = adapted_features[second_modality]
        text_hidden = adapted_features[third_modality]

        audio_before_encoder = audio_hidden
        router_losses = []
        audio_entropies = []
        audio_layer_statistics = {}
        if self.audio_encoder_mode == "mope":
            audio_context = self.perturb_context(
                visual_hidden, self.audio_context_mode
            )
            shared_audio_routes = None
            for layer_index, block in enumerate(self.audio_encoder, start=1):
                before_block = audio_hidden
                audio_hidden, balance_loss, entropy, route_scores = block(
                    audio_hidden,
                    audio_context,
                    route_scores_override=(
                        shared_audio_routes
                        if self.audio_route_sharing == "shared_first"
                        else None
                    ),
                )
                if self.audio_route_sharing == "shared_first" and shared_audio_routes is None:
                    shared_audio_routes = route_scores.detach()
                router_losses.append(balance_loss)
                audio_entropies.append(entropy)
                layer_change, layer_cosine = self.feature_change_statistics(
                    before_block, audio_hidden
                )
                audio_layer_statistics.update(
                    {
                        f"audio_layer_{layer_index}_relative_change": layer_change,
                        f"audio_layer_{layer_index}_cosine_similarity": layer_cosine,
                        f"audio_layer_{layer_index}_condition_scale": (
                            block.condition_scale.detach().abs().mean()
                        ),
                    }
                )
        else:
            for layer_index, block in enumerate(self.audio_encoder, start=1):
                before_block = audio_hidden
                audio_hidden = block(audio_hidden)
                layer_change, layer_cosine = self.feature_change_statistics(
                    before_block, audio_hidden
                )
                audio_layer_statistics.update(
                    {
                        f"audio_layer_{layer_index}_relative_change": layer_change,
                        f"audio_layer_{layer_index}_cosine_similarity": layer_cosine,
                    }
                )
        audio_relative_change, audio_cosine_similarity = (
            self.feature_change_statistics(audio_before_encoder, audio_hidden)
        )

        if self.va_fusion_mode == "gated_second_order":
            va_pair = self.pair_features(audio_hidden, visual_hidden)
            va_candidate = self.va_candidate(va_pair)
            va_gate = self.va_gate(torch.cat([audio_hidden, visual_hidden], dim=-1))
            va_hidden = self.va_normalization(
                va_gate * va_candidate + (1.0 - va_gate) * audio_hidden
            )
            va_gate_mean = va_gate.mean()
        elif self.va_fusion_mode == "concat_mlp":
            va_hidden = self.va_normalization(
                self.va_concat_fusion(torch.cat([audio_hidden, visual_hidden], dim=-1))
            )
            va_gate_mean = audio_hidden.new_zeros(())
        else:
            va_hidden = self.va_normalization(0.5 * (audio_hidden + visual_hidden))
            va_gate_mean = audio_hidden.new_zeros(())

        text_before_encoder = text_hidden
        text_entropies = []
        text_layer_statistics = {}
        if self.text_encoder_mode == "mope":
            text_context = self.perturb_context(va_hidden, self.text_context_mode)
            shared_text_routes = None
            for layer_index, block in enumerate(self.text_encoder, start=1):
                before_block = text_hidden
                text_hidden, balance_loss, entropy, route_scores = block(
                    text_hidden,
                    text_context,
                    route_scores_override=(
                        shared_text_routes
                        if self.text_route_sharing == "shared_first"
                        else None
                    ),
                )
                if self.text_route_sharing == "shared_first" and shared_text_routes is None:
                    shared_text_routes = route_scores.detach()
                router_losses.append(balance_loss)
                text_entropies.append(entropy)
                layer_change, layer_cosine = self.feature_change_statistics(
                    before_block, text_hidden
                )
                text_layer_statistics.update(
                    {
                        f"text_layer_{layer_index}_relative_change": layer_change,
                        f"text_layer_{layer_index}_cosine_similarity": layer_cosine,
                        f"text_layer_{layer_index}_condition_scale": (
                            block.condition_scale.detach().abs().mean()
                        ),
                    }
                )
        else:
            for layer_index, block in enumerate(self.text_encoder, start=1):
                before_block = text_hidden
                text_hidden = block(text_hidden)
                layer_change, layer_cosine = self.feature_change_statistics(
                    before_block, text_hidden
                )
                text_layer_statistics.update(
                    {
                        f"text_layer_{layer_index}_relative_change": layer_change,
                        f"text_layer_{layer_index}_cosine_similarity": layer_cosine,
                    }
                )
        text_relative_change, text_cosine_similarity = (
            self.feature_change_statistics(text_before_encoder, text_hidden)
        )

        final_features = self.pair_features(text_hidden, va_hidden)
        prediction = self.final_head(final_features).squeeze(-1)
        auxiliary_predictions = {
            "visual": self.auxiliary_heads["visual"](visual_hidden).squeeze(-1),
            "audio": self.auxiliary_heads["audio"](audio_hidden).squeeze(-1),
            "va": self.auxiliary_heads["va"](va_hidden).squeeze(-1),
            "text": self.auxiliary_heads["text"](text_hidden).squeeze(-1),
        }
        router_loss = (
            torch.stack(router_losses).mean()
            if router_losses
            else prediction.new_zeros(())
        )
        route_statistics = {
            "audio_entropy": (
                torch.stack(audio_entropies).mean()
                if audio_entropies
                else prediction.new_zeros(())
            ),
            "text_entropy": (
                torch.stack(text_entropies).mean()
                if text_entropies
                else prediction.new_zeros(())
            ),
            "va_gate_mean": va_gate_mean,
            "audio_relative_change": audio_relative_change,
            "audio_cosine_similarity": audio_cosine_similarity,
            "text_relative_change": text_relative_change,
            "text_cosine_similarity": text_cosine_similarity,
            "audio_condition_scale_mean": self.mean_condition_scale(
                self.audio_encoder, prediction
            ),
            "text_condition_scale_mean": self.mean_condition_scale(
                self.text_encoder, prediction
            ),
            **audio_layer_statistics,
            **text_layer_statistics,
        }
        return FeatureV2A2TOutput(
            prediction=prediction,
            auxiliary_predictions=auxiliary_predictions,
            router_loss=router_loss,
            route_statistics=route_statistics,
            features={
                "visual": visual_hidden,
                "audio": audio_hidden,
                "va": va_hidden,
                "text": text_hidden,
            },
        )


def feature_v2a2t_loss(
    output: FeatureV2A2TOutput,
    labels: torch.Tensor,
    auxiliary_weight: float = 0.1,
    correlation_weight: float = 0.05,
    router_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    labels = labels.reshape(-1)
    main_mae = F.l1_loss(output.prediction, labels)
    auxiliary_losses = [
        F.l1_loss(prediction, labels)
        for prediction in output.auxiliary_predictions.values()
    ]
    auxiliary_mae = (
        torch.stack(auxiliary_losses).mean()
        if auxiliary_losses
        else output.prediction.new_zeros(())
    )

    centered_prediction = output.prediction - output.prediction.mean()
    centered_labels = labels - labels.mean()
    correlation = torch.sum(centered_prediction * centered_labels) / (
        torch.sqrt(torch.sum(centered_prediction.square()) + 1e-8)
        * torch.sqrt(torch.sum(centered_labels.square()) + 1e-8)
    )
    correlation_loss = 1.0 - correlation
    total_loss = (
        main_mae
        + auxiliary_weight * auxiliary_mae
        + correlation_weight * correlation_loss
        + router_weight * output.router_loss
    )
    return total_loss, {
        "main_mae": main_mae.detach(),
        "auxiliary_mae": auxiliary_mae.detach(),
        "correlation_loss": correlation_loss.detach(),
        "router_loss": output.router_loss.detach(),
    }
