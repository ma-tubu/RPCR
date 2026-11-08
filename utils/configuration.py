from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RPCR from utterance-level feature caches."
    )
    parser.add_argument("--config", type=Path, help="JSON experiment configuration.")
    parser.add_argument("--dataset-name", choices=["CMUMOSI", "CMUMOSEI", "CHSIMS"])
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--output-activation",
        choices=["auto", "identity", "tanh"],
        default="auto",
        help="auto uses tanh for CHSIMS and identity for CMU-MOSI/MOSEI.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--visual-depth", type=int, default=3)
    parser.add_argument("--audio-depth", type=int, default=4)
    parser.add_argument("--text-depth", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--expert-dim",
        type=int,
        default=64,
        help="Refinement basis dimension; 0 uses hidden_dim.",
    )
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--architecture-mode",
        choices=["chain", "direct_concat", "unimodal"],
        default="chain",
    )
    parser.add_argument(
        "--chain-order",
        choices=["v2a2t", "v2t2a", "a2v2t", "a2t2v", "t2v2a", "t2a2v"],
        default="v2a2t",
        help="Directed order; stage capacities stay fixed and only modality assignment changes.",
    )
    parser.add_argument(
        "--unimodal-modality",
        choices=["audio", "visual", "text"],
        default="text",
    )
    parser.add_argument(
        "--direct-fusion-dim",
        type=int,
        default=0,
        help="Width of the direct V/A/T concat MLP; 0 uses 2*hidden_dim.",
    )
    parser.add_argument("--direct-fusion-depth", type=int, default=2)
    parser.add_argument(
        "--audio-encoder-mode", choices=["mope", "ffn", "skip"], default="mope"
    )
    parser.add_argument(
        "--text-encoder-mode", choices=["mope", "ffn", "skip"], default="mope"
    )
    parser.add_argument(
        "--audio-modulation-mode",
        choices=["full", "scale_only", "shift_only", "none"],
        default="none",
    )
    parser.add_argument(
        "--text-modulation-mode",
        choices=["full", "scale_only", "shift_only", "none"],
        default="full",
    )
    parser.add_argument(
        "--audio-context-mode",
        choices=["real", "shuffled", "zero", "mean"],
        default="real",
    )
    parser.add_argument(
        "--text-context-mode",
        choices=["real", "shuffled", "zero", "mean"],
        default="real",
    )
    parser.add_argument(
        "--audio-routing-mode",
        choices=["dynamic", "uniform", "static_learned", "target_only", "context_only"],
        default="dynamic",
    )
    parser.add_argument(
        "--text-routing-mode",
        choices=["dynamic", "uniform", "static_learned", "target_only", "context_only"],
        default="dynamic",
    )
    parser.add_argument(
        "--audio-route-sharing",
        choices=["per_layer", "shared_first"],
        default="per_layer",
    )
    parser.add_argument(
        "--text-route-sharing",
        choices=["per_layer", "shared_first"],
        default="per_layer",
    )
    parser.add_argument(
        "--audio-prompt-mode",
        choices=["both", "dynamic_only", "static_only", "none"],
        default="both",
    )
    parser.add_argument(
        "--text-prompt-mode",
        choices=["both", "dynamic_only", "static_only", "none"],
        default="both",
    )
    parser.add_argument(
        "--audio-gate-mode", choices=["learned", "open", "closed"], default="learned"
    )
    parser.add_argument(
        "--text-gate-mode", choices=["learned", "open", "closed"], default="learned"
    )
    parser.add_argument(
        "--va-fusion-mode",
        choices=["gated_second_order", "concat_mlp", "add"],
        default="gated_second_order",
    )
    parser.add_argument(
        "--audio-ffn-expansion",
        type=int,
        default=0,
        help="Stage-specific FFN expansion; 0 uses --expansion.",
    )
    parser.add_argument(
        "--text-ffn-expansion",
        type=int,
        default=0,
        help="Stage-specific FFN expansion; 0 uses --expansion.",
    )
    parser.add_argument("--audio-condition-scale-init", type=float, default=0.003)
    parser.add_argument("--text-condition-scale-init", type=float, default=0.01)
    parser.add_argument("--audio-condition-scale-growth", type=float, default=1.0)
    parser.add_argument("--text-condition-scale-growth", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--correlation-weight", type=float, default=0.05)
    parser.add_argument("--router-weight", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--monitor",
        choices=["MAE", "Corr", "F1", "Acc_2", "Acc_3", "Acc_5", "Acc_7", "loss"],
        default="MAE",
        help="Validation metric used for best.pt and early stopping.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run every configured epoch while still saving the validation-selected best.pt.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    if config_args.config is not None:
        with config_args.config.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
        if not isinstance(settings, dict):
            parser.error("Configuration must be a JSON object.")
        actions = {action.dest: action for action in parser._actions}
        unknown = set(settings) - set(actions)
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        for key, value in settings.items():
            action = actions[key]
            if key in {"config", "cache_path", "output_dir", "resume", "help"}:
                parser.error(f"Pass {key} on the command line.")
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f"{key} must be a boolean.")
            elif action.type is not None:
                try:
                    settings[key] = action.type(value)
                except (TypeError, ValueError) as error:
                    parser.error(f"Invalid value for {key}: {error}")
            if action.choices is not None and settings[key] not in action.choices:
                parser.error(f"Invalid value for {key}: {settings[key]!r}")
        parser.set_defaults(**settings)
    return parser.parse_args()


def model_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "visual_dim": 1024,
        "audio_dim": 512,
        "text_dim": 1024,
        "hidden_dim": args.hidden_dim,
        "visual_depth": args.visual_depth,
        "audio_depth": args.audio_depth,
        "text_depth": args.text_depth,
        "num_experts": args.num_experts,
        "expert_dim": (
            args.hidden_dim
            if getattr(args, "expert_dim", 0) == 0
            else args.expert_dim
        ),
        "expansion": args.expansion,
        "dropout": args.dropout,
        "architecture_mode": getattr(args, "architecture_mode", "chain"),
        "chain_order": getattr(args, "chain_order", "v2a2t"),
        "unimodal_modality": getattr(args, "unimodal_modality", "text"),
        "direct_fusion_dim": (
            None
            if getattr(args, "direct_fusion_dim", 0) == 0
            else args.direct_fusion_dim
        ),
        "direct_fusion_depth": getattr(args, "direct_fusion_depth", 2),
        "audio_encoder_mode": args.audio_encoder_mode,
        "text_encoder_mode": args.text_encoder_mode,
        "audio_modulation_mode": args.audio_modulation_mode,
        "text_modulation_mode": args.text_modulation_mode,
        "audio_context_mode": getattr(args, "audio_context_mode", "real"),
        "text_context_mode": getattr(args, "text_context_mode", "real"),
        "audio_routing_mode": getattr(args, "audio_routing_mode", "dynamic"),
        "text_routing_mode": getattr(args, "text_routing_mode", "dynamic"),
        "audio_route_sharing": getattr(args, "audio_route_sharing", "per_layer"),
        "text_route_sharing": getattr(args, "text_route_sharing", "per_layer"),
        "audio_prompt_mode": getattr(args, "audio_prompt_mode", "both"),
        "text_prompt_mode": getattr(args, "text_prompt_mode", "both"),
        "audio_gate_mode": getattr(args, "audio_gate_mode", "learned"),
        "text_gate_mode": getattr(args, "text_gate_mode", "learned"),
        "va_fusion_mode": getattr(args, "va_fusion_mode", "gated_second_order"),
        "audio_ffn_expansion": (
            None
            if getattr(args, "audio_ffn_expansion", 0) == 0
            else args.audio_ffn_expansion
        ),
        "text_ffn_expansion": (
            None
            if getattr(args, "text_ffn_expansion", 0) == 0
            else args.text_ffn_expansion
        ),
        "audio_condition_scale_init": args.audio_condition_scale_init,
        "text_condition_scale_init": args.text_condition_scale_init,
        "audio_condition_scale_growth": args.audio_condition_scale_growth,
        "text_condition_scale_growth": args.text_condition_scale_growth,
    }
