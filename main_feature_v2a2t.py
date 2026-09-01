from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.feature_v2a2t import FeatureV2A2T, feature_v2a2t_loss
from utils.ebmc_feature_dataset import (
    EBMCCachedFeatureDataset,
    iter_split_sizes,
    load_ebmc_feature_cache,
)
from utils.regression_metrics import format_metrics, regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train feature-domain V->A->T on utterance-level MSA features."
    )
    parser.add_argument("--dataset-name", choices=["CMUMOSI", "CMUMOSEI", "CHSIMS"])
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
        default=0,
        help="Latent expert-prompt dimension; 0 uses hidden_dim and preserves the original model.",
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
        default="full",
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
    parser.add_argument("--audio-condition-scale-init", type=float, default=0.01)
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
    parser.add_argument(
        "--eval-test-every-epoch",
        action="store_true",
        help=(
            "Evaluate the test split after every epoch for diagnostics only. "
            "Test metrics never affect optimization, early stopping, or checkpoint selection."
        ),
    )
    parser.add_argument(
        "--evaluate-context-controls",
        action="store_true",
        help=(
            "After validation-only checkpoint selection, evaluate deterministic shuffled, "
            "zero, and batch-mean context controls on that same checkpoint."
        ),
    )
    parser.add_argument(
        "--enable-test-target-checkpoints",
        action="store_true",
        help=(
            "Exploratory only: save test-driven Acc-7 and all-target checkpoints. "
            "These checkpoints must not be used as validation-selected formal results."
        ),
    )
    parser.add_argument(
        "--enable-validation-acc7-checkpoint",
        action="store_true",
        help=(
            "Save and finally evaluate the checkpoint with highest validation Acc-7. "
            "This selection path never uses per-epoch test metrics."
        ),
    )
    parser.add_argument(
        "--enable-validation-acc5-checkpoint",
        action="store_true",
        help=(
            "Save and finally evaluate the checkpoint with highest validation Acc-5. "
            "This is the recommended formal selection companion for CHSIMS."
        ),
    )
    parser.add_argument("--target-acc7", type=float, default=0.477)
    parser.add_argument("--secondary-target-acc7", type=float, default=0.466)
    parser.add_argument("--target-acc2", type=float, default=0.864)
    parser.add_argument("--target-f1", type=float, default=0.864)
    parser.add_argument("--target-mae", type=float, default=0.700)
    parser.add_argument("--target-corr", type=float, default=0.800)
    parser.add_argument(
        "--target-specs",
        default="",
        help=(
            "Optional semicolon-separated exploratory test target groups, e.g. "
            "'strong:Acc_7>=0.541,Acc_2>=0.865,F1>=0.865,MAE<=0.526,Corr>=0.772;"
            "weak:Acc_7>=0.541,Acc_2>=0.863,F1>=0.862,MAE<=0.536,Corr>=0.787'. "
            "If omitted, legacy primary/secondary target arguments are used."
        ),
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def monitor_improved(current: float, best: float, monitor: str) -> bool:
    return current < best if monitor in {"MAE", "loss"} else current > best


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset: EBMCCachedFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_seed if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


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


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, learning_rate_multiplier)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def target_thresholds(
    args: argparse.Namespace, *, acc7: Optional[float] = None
) -> Dict[str, float]:
    return {
        "Acc_7": args.target_acc7 if acc7 is None else acc7,
        "Acc_2": args.target_acc2,
        "F1": args.target_f1,
        "MAE": args.target_mae,
        "Corr": args.target_corr,
    }


def meets_test_targets(metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    return (
        metrics["Acc_7"] >= thresholds["Acc_7"]
        and metrics["Acc_2"] >= thresholds["Acc_2"]
        and metrics["F1"] >= thresholds["F1"]
        and metrics["MAE"] <= thresholds["MAE"]
        and metrics["Corr"] >= thresholds["Corr"]
    )


def parse_target_specs(
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    if not args.target_specs.strip():
        return [
            {
                "name": "target",
                "hit_key": "target_hits",
                "best_key": "best_target_hit",
                "checkpoint_stem": "target_hit",
                "priority_metric": "Acc_7",
                "conditions": [
                    ("Acc_7", ">=", args.target_acc7),
                    ("Acc_2", ">=", args.target_acc2),
                    ("F1", ">=", args.target_f1),
                    ("MAE", "<=", args.target_mae),
                    ("Corr", ">=", args.target_corr),
                ],
            },
            {
                "name": "secondary_target",
                "hit_key": "secondary_target_hits",
                "best_key": "best_secondary_target_hit",
                "checkpoint_stem": "secondary_target_hit",
                "priority_metric": "Acc_7",
                "conditions": [
                    ("Acc_7", ">=", args.secondary_target_acc7),
                    ("Acc_2", ">=", args.target_acc2),
                    ("F1", ">=", args.target_f1),
                    ("MAE", "<=", args.target_mae),
                    ("Corr", ">=", args.target_corr),
                ],
            },
        ]

    groups: list[Dict[str, Any]] = []
    for group_text in args.target_specs.split(";"):
        group_text = group_text.strip()
        if not group_text:
            continue
        if ":" not in group_text:
            raise ValueError(f"Target group must use name:conditions: {group_text!r}")
        raw_name, raw_conditions = group_text.split(":", 1)
        name = raw_name.strip()
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid target group name: {name!r}")
        safe_name = name.replace("-", "_")
        conditions = []
        for condition_text in raw_conditions.split(","):
            condition_text = condition_text.strip()
            operator = None
            for candidate in (">=", "<=", ">", "<"):
                if candidate in condition_text:
                    operator = candidate
                    metric_name, value = condition_text.split(candidate, 1)
                    break
            if operator is None:
                raise ValueError(f"Invalid target condition: {condition_text!r}")
            metric_name = metric_name.strip()
            if not metric_name:
                raise ValueError(f"Missing metric in target condition: {condition_text!r}")
            conditions.append((metric_name, operator, float(value.strip())))
        priority_metric = next(
            (
                metric_name
                for metric_name, operator, _ in conditions
                if operator in {">=", ">"}
            ),
            conditions[0][0],
        )
        groups.append(
            {
                "name": safe_name,
                "hit_key": f"{safe_name}_hits",
                "best_key": f"best_{safe_name}_hit",
                "checkpoint_stem": f"{safe_name}_hit",
                "priority_metric": priority_metric,
                "conditions": conditions,
            }
        )
    if not groups:
        raise ValueError("--target-specs did not contain any target groups.")
    return groups


def target_group_summary(target_groups: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        group["name"]: {
            "priority_metric": group["priority_metric"],
            "conditions": {
                metric: {"operator": operator, "value": value}
                for metric, operator, value in group["conditions"]
            },
        }
        for group in target_groups
    }


def meets_target_group(metrics: Dict[str, float], group: Dict[str, Any]) -> bool:
    for metric_name, operator, threshold in group["conditions"]:
        if metric_name not in metrics:
            return False
        value = float(metrics[metric_name])
        if operator == ">=" and value < threshold:
            return False
        if operator == ">" and value <= threshold:
            return False
        if operator == "<=" and value > threshold:
            return False
        if operator == "<" and value >= threshold:
            return False
    return True


def load_test_search_state(
    path: Path,
    thresholds: Dict[str, float],
    secondary_thresholds: Dict[str, float],
    target_groups: list[Dict[str, Any]],
) -> Dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "protocol": (
                "Exploratory test-driven monitoring. Never report these checkpoints as "
                "validation-selected formal results."
            ),
            "best_test_acc7": None,
            "best_test_acc5": None,
            "target_hits": [],
        }
    state["thresholds"] = thresholds
    state["secondary_thresholds"] = secondary_thresholds
    state["target_groups"] = target_group_summary(target_groups)
    state.setdefault("target_hits", [])
    state.setdefault("secondary_target_hits", [])
    for group in target_groups:
        state.setdefault(group["hit_key"], [])
    return state


def copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary_path)
    os.replace(temporary_path, destination)


def record_test_target_hit(
    *,
    output_dir: Path,
    state: Dict[str, Any],
    hit_key: str,
    best_key: str,
    checkpoint_stem: str,
    priority_metric: str,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_arguments: Dict[str, Any],
) -> Dict[str, Any]:
    epoch_checkpoint_name = f"{checkpoint_stem}_epoch_{epoch:03d}.pt"
    epoch_checkpoint = output_dir / epoch_checkpoint_name
    save_checkpoint(epoch_checkpoint, **checkpoint_arguments)

    hits = state.setdefault(hit_key, [])
    first_hit = not hits
    hit = {
        "epoch": epoch,
        "metrics": metrics,
        "checkpoint": epoch_checkpoint_name,
        "is_first": first_hit,
    }
    hits.append(hit)
    if first_hit:
        copy_checkpoint_atomic(epoch_checkpoint, output_dir / f"{checkpoint_stem}.pt")

    best_hit = state.get(best_key)
    if priority_metric not in metrics:
        raise KeyError(f"Priority metric {priority_metric!r} is missing from metrics.")
    best_improved = best_hit is None or float(metrics[priority_metric]) > float(
        best_hit["metrics"][priority_metric]
    )
    if best_improved:
        metric_suffix = priority_metric.lower()
        best_checkpoint_name = f"{checkpoint_stem}_best_{metric_suffix}.pt"
        copy_checkpoint_atomic(epoch_checkpoint, output_dir / best_checkpoint_name)
        state[best_key] = {
            "epoch": epoch,
            "metrics": metrics,
            "checkpoint": best_checkpoint_name,
            "priority_metric": priority_metric,
            "source_checkpoint": epoch_checkpoint_name,
        }
    return {
        "first_hit": first_hit,
        "best_improved": best_improved,
        "epoch_checkpoint": epoch_checkpoint_name,
    }


def save_checkpoint(
    path: Path,
    *,
    model: FeatureV2A2T,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_validation_value: float,
    monitor: str,
    patience_count: int,
    arguments: Dict[str, Any],
    current_metrics: Dict[str, Any],
) -> None:
    checkpoint = {
        "format": "feature_v2a2t_full_checkpoint_v1",
        "epoch": epoch,
        "monitor": monitor,
        "best_validation_value": best_validation_value,
        "best_validation_mae": float(current_metrics["valid"]["MAE"]),
        "patience_count": patience_count,
        "arguments": arguments,
        "model_config": model_config(argparse.Namespace(**arguments)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "metrics": current_metrics,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def restore_checkpoint(
    checkpoint_path: Path,
    *,
    model: FeatureV2A2T,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def run_epoch(
    model: FeatureV2A2T,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LambdaLR],
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
    auxiliary_weight: float,
    correlation_weight: float,
    router_weight: float,
    max_batches: Optional[int],
    dataset_name: str,
    output_activation: str,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    prediction_batches = []
    label_batches = []
    total_loss = 0.0
    sample_count = 0
    component_totals: Dict[str, float] = {}
    route_totals: Dict[str, float] = {}

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        visual = batch["vision"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        text = batch["text"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).reshape(-1)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(visual, audio, text)
                if output_activation == "tanh":
                    output.prediction = torch.tanh(output.prediction)
                    output.auxiliary_predictions = {
                        name: torch.tanh(prediction)
                        for name, prediction in output.auxiliary_predictions.items()
                    }
                loss, components = feature_v2a2t_loss(
                    output,
                    labels,
                    auxiliary_weight=auxiliary_weight,
                    correlation_weight=correlation_weight,
                    router_weight=router_weight,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        prediction_batches.append(output.prediction.detach().cpu())
        label_batches.append(labels.detach().cpu())
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(value) * batch_size
        for name, value in output.route_statistics.items():
            route_totals[name] = route_totals.get(name, 0.0) + float(value.detach()) * batch_size

    if sample_count == 0:
        raise RuntimeError("No samples were processed in the epoch.")
    predictions = torch.cat(prediction_batches)
    labels = torch.cat(label_batches)
    metrics = regression_metrics(predictions, labels, dataset_name)
    metrics["loss"] = total_loss / sample_count
    metrics.update(
        {
            f"loss_{name}": value / sample_count
            for name, value in component_totals.items()
        }
    )
    metrics.update(
        {
            f"route_{name}": value / sample_count
            for name, value in route_totals.items()
        }
    )
    metrics["samples"] = float(sample_count)
    return metrics


def evaluate_context_controls(
    model: FeatureV2A2T,
    loader: DataLoader,
    *,
    baseline_metrics: Dict[str, float],
    run_epoch_arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate content controls on one already validation-selected checkpoint."""
    if model.architecture_mode != "chain":
        raise ValueError("Context controls only apply to architecture_mode=chain.")
    original_audio = model.audio_context_mode
    original_text = model.text_context_mode
    controls = {
        "real": {"audio": "real", "text": "real"},
        "audio_shuffled": {"audio": "shuffled", "text": "real"},
        "audio_zero": {"audio": "zero", "text": "real"},
        "audio_mean": {"audio": "mean", "text": "real"},
        "text_shuffled": {"audio": "real", "text": "shuffled"},
        "text_zero": {"audio": "real", "text": "zero"},
        "text_mean": {"audio": "real", "text": "mean"},
        "both_shuffled": {"audio": "shuffled", "text": "shuffled"},
        "both_zero": {"audio": "zero", "text": "zero"},
    }
    results: Dict[str, Any] = {
        "protocol": (
            "Post-hoc controls evaluated on the same validation-MAE-selected checkpoint. "
            "Shuffling is a deterministic one-position within-batch roll; no control "
            "metric participates in training or model selection."
        ),
        "controls": {},
    }
    try:
        for name, modes in controls.items():
            if name == "real" and original_audio == "real" and original_text == "real":
                metrics = baseline_metrics
            else:
                model.set_context_modes(audio=modes["audio"], text=modes["text"])
                metrics = run_epoch(model, loader, **run_epoch_arguments)
            results["controls"][name] = {"modes": modes, "test": metrics}
    finally:
        model.set_context_modes(audio=original_audio, text=original_text)
    return results


def main() -> None:
    args = parse_args()
    if args.enable_test_target_checkpoints and not args.eval_test_every_epoch:
        raise ValueError(
            "--enable-test-target-checkpoints requires --eval-test-every-epoch."
        )
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    status_path = output_dir / "status.json"
    test_search_path = output_dir / "test_search_state.json"
    validation_acc7_state_path = output_dir / "validation_acc7_state.json"
    validation_acc5_state_path = output_dir / "validation_acc5_state.json"

    if args.resume is None:
        existing_training_artifacts = [
            path
            for path in (
                metrics_path,
                output_dir / "best.pt",
                output_dir / "last.pt",
                output_dir / "final_result.json",
                output_dir / "best_test_acc7.pt",
                output_dir / "best_test_acc5.pt",
                output_dir / "target_hit.pt",
                output_dir / "target_hit_best_acc7.pt",
                output_dir / "target_hit_best_acc5.pt",
                output_dir / "secondary_target_hit.pt",
                output_dir / "secondary_target_hit_best_acc7.pt",
                output_dir / "secondary_target_hit_best_acc5.pt",
                output_dir / "best_valid_acc7.pt",
                output_dir / "best_valid_acc5.pt",
                test_search_path,
                validation_acc7_state_path,
                validation_acc5_state_path,
            )
            if path.exists()
        ]
        if existing_training_artifacts:
            names = ", ".join(path.name for path in existing_training_artifacts)
            raise FileExistsError(
                f"Output directory already contains training artifacts ({names}): "
                f"{output_dir}. Use a new output directory or --resume last.pt."
            )

    if args.num_workers > 0 and os.name == "nt":
        print(
            "Warning: num_workers>0 duplicates the in-memory cache on Windows; num_workers=0 is recommended.",
            flush=True,
        )

    cache = load_ebmc_feature_cache(args.cache_path)
    cache_dataset_name = str(cache.get("dataset_name", "")).upper()
    if args.dataset_name is None:
        args.dataset_name = cache_dataset_name or "CMUMOSEI"
    elif cache_dataset_name and args.dataset_name != cache_dataset_name:
        raise ValueError(
            f"Dataset mismatch: --dataset-name={args.dataset_name}, "
            f"cache dataset_name={cache_dataset_name}."
        )
    if args.output_activation == "auto":
        args.output_activation = "tanh" if args.dataset_name == "CHSIMS" else "identity"
    split_sizes = dict(iter_split_sizes(cache))
    datasets = {
        split: EBMCCachedFeatureDataset(cache, split)
        for split in ("train", "valid", "test")
    }
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        ),
        "valid": make_loader(
            datasets["valid"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        ),
    }

    configuration = model_config(args)
    model = FeatureV2A2T(**configuration).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_batches = len(loaders["train"])
    if args.max_train_batches is not None:
        train_batches = min(train_batches, args.max_train_batches)
    total_steps = max(1, train_batches * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_validation_value = (
        float("inf") if args.monitor in {"MAE", "loss"} else float("-inf")
    )
    patience_count = 0
    if args.resume is not None:
        checkpoint = restore_checkpoint(
            args.resume.resolve(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        checkpoint_monitor = str(checkpoint.get("monitor", "MAE"))
        if checkpoint_monitor != args.monitor:
            raise ValueError(
                f"Resume monitor mismatch: checkpoint={checkpoint_monitor}, "
                f"arguments={args.monitor}."
            )
        best_validation_value = float(
            checkpoint.get(
                "best_validation_value", checkpoint["best_validation_mae"]
            )
        )
        patience_count = int(checkpoint.get("patience_count", 0))

    argument_dict = vars(args).copy()
    argument_dict["cache_path"] = str(args.cache_path.resolve())
    argument_dict["output_dir"] = str(output_dir)
    argument_dict["resume"] = str(args.resume.resolve()) if args.resume else None
    thresholds = target_thresholds(args)
    secondary_thresholds = target_thresholds(
        args, acc7=args.secondary_target_acc7
    )
    target_groups = parse_target_specs(args)
    test_search_state = load_test_search_state(
        test_search_path, thresholds, secondary_thresholds, target_groups
    )
    validation_acc7_state = (
        json.loads(validation_acc7_state_path.read_text(encoding="utf-8"))
        if validation_acc7_state_path.is_file()
        else {"best": None}
    )
    validation_acc5_state = (
        json.loads(validation_acc5_state_path.read_text(encoding="utf-8"))
        if validation_acc5_state_path.is_file()
        else {"best": None}
    )
    run_metadata = {
        "status": "running",
        "pid": os.getpid(),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "split_sizes": split_sizes,
        "arguments": argument_dict,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json_atomic(status_path, run_metadata)
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2), flush=True)

    best_epoch = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model,
                loaders["train"],
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                auxiliary_weight=args.auxiliary_weight,
                correlation_weight=args.correlation_weight,
                router_weight=args.router_weight,
                max_batches=args.max_train_batches,
                dataset_name=args.dataset_name,
                output_activation=args.output_activation,
            )
            valid_metrics = run_epoch(
                model,
                loaders["valid"],
                device=device,
                optimizer=None,
                scheduler=None,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                auxiliary_weight=args.auxiliary_weight,
                correlation_weight=args.correlation_weight,
                router_weight=args.router_weight,
                max_batches=args.max_eval_batches,
                dataset_name=args.dataset_name,
                output_activation=args.output_activation,
            )
            test_monitor_metrics = None
            if args.eval_test_every_epoch:
                test_monitor_metrics = run_epoch(
                    model,
                    loaders["test"],
                    device=device,
                    optimizer=None,
                    scheduler=None,
                    scaler=scaler,
                    use_amp=use_amp,
                    grad_clip=args.grad_clip,
                    auxiliary_weight=args.auxiliary_weight,
                    correlation_weight=args.correlation_weight,
                    router_weight=args.router_weight,
                    max_batches=args.max_eval_batches,
                    dataset_name=args.dataset_name,
                    output_activation=args.output_activation,
                )
            current_validation_value = float(valid_metrics[args.monitor])
            improved = monitor_improved(
                current_validation_value, best_validation_value, args.monitor
            )
            if improved:
                best_validation_value = current_validation_value
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            epoch_record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": time.time() - epoch_start,
                "train": train_metrics,
                "valid": valid_metrics,
                "improved": improved,
                "monitor": args.monitor,
                "monitor_value": current_validation_value,
                "best_validation_value": best_validation_value,
                "best_validation_mae": valid_metrics["MAE"],
                "patience_count": patience_count,
            }
            if test_monitor_metrics is not None:
                epoch_record["test_monitor"] = test_monitor_metrics
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            print(
                f"Epoch {epoch:03d} | train {format_metrics(train_metrics)}",
                flush=True,
            )
            print(
                f"Epoch {epoch:03d} | valid {format_metrics(valid_metrics)} | "
                f"best_{args.monitor}={best_validation_value:.4f} "
                f"patience={patience_count}/{args.patience}",
                flush=True,
            )
            if test_monitor_metrics is not None:
                print(
                    f"Epoch {epoch:03d} | test-monitor "
                    f"{format_metrics(test_monitor_metrics)} | diagnostic_only=true",
                    flush=True,
                )

            checkpoint_metrics = {"train": train_metrics, "valid": valid_metrics}
            if test_monitor_metrics is not None:
                checkpoint_metrics["test_monitor"] = test_monitor_metrics
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_value=best_validation_value,
                monitor=args.monitor,
                patience_count=patience_count,
                arguments=argument_dict,
                current_metrics=checkpoint_metrics,
            )
            if improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_validation_value=best_validation_value,
                    monitor=args.monitor,
                    patience_count=patience_count,
                    arguments=argument_dict,
                    current_metrics=checkpoint_metrics,
                )

            if args.enable_validation_acc7_checkpoint:
                best_valid_acc7 = validation_acc7_state.get("best")
                current_valid_acc7 = float(valid_metrics["Acc_7"])
                current_valid_mae = float(valid_metrics["MAE"])
                valid_acc7_improved = best_valid_acc7 is None or (
                    current_valid_acc7 > float(best_valid_acc7["value"])
                    or (
                        current_valid_acc7 == float(best_valid_acc7["value"])
                        and current_valid_mae < float(best_valid_acc7["validation_mae"])
                    )
                )
                if valid_acc7_improved:
                    save_checkpoint(
                        output_dir / "best_valid_acc7.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_validation_value=best_validation_value,
                        monitor=args.monitor,
                        patience_count=patience_count,
                        arguments=argument_dict,
                        current_metrics=checkpoint_metrics,
                    )
                    validation_acc7_state["best"] = {
                        "epoch": epoch,
                        "value": current_valid_acc7,
                        "validation_mae": current_valid_mae,
                        "validation_metrics": valid_metrics,
                        "checkpoint": "best_valid_acc7.pt",
                    }
                    write_json_atomic(
                        validation_acc7_state_path, validation_acc7_state
                    )
                    print(
                        f"Epoch {epoch:03d} | best Valid Acc_7="
                        f"{current_valid_acc7:.4f} checkpoint=best_valid_acc7.pt",
                        flush=True,
                    )

            if args.enable_validation_acc5_checkpoint:
                best_valid_acc5 = validation_acc5_state.get("best")
                current_valid_acc5 = float(valid_metrics["Acc_5"])
                current_valid_mae = float(valid_metrics["MAE"])
                valid_acc5_improved = best_valid_acc5 is None or (
                    current_valid_acc5 > float(best_valid_acc5["value"])
                    or (
                        current_valid_acc5 == float(best_valid_acc5["value"])
                        and current_valid_mae < float(best_valid_acc5["validation_mae"])
                    )
                )
                if valid_acc5_improved:
                    save_checkpoint(
                        output_dir / "best_valid_acc5.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_validation_value=best_validation_value,
                        monitor=args.monitor,
                        patience_count=patience_count,
                        arguments=argument_dict,
                        current_metrics=checkpoint_metrics,
                    )
                    validation_acc5_state["best"] = {
                        "epoch": epoch,
                        "value": current_valid_acc5,
                        "validation_mae": current_valid_mae,
                        "validation_metrics": valid_metrics,
                        "checkpoint": "best_valid_acc5.pt",
                    }
                    write_json_atomic(
                        validation_acc5_state_path, validation_acc5_state
                    )
                    print(
                        f"Epoch {epoch:03d} | best Valid Acc_5="
                        f"{current_valid_acc5:.4f} checkpoint=best_valid_acc5.pt",
                        flush=True,
                    )

            if args.enable_test_target_checkpoints and test_monitor_metrics is not None:
                best_test_acc7 = test_search_state.get("best_test_acc7")
                current_acc7 = float(test_monitor_metrics["Acc_7"])
                if best_test_acc7 is None or current_acc7 > float(best_test_acc7["value"]):
                    save_checkpoint(
                        output_dir / "best_test_acc7.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_validation_value=best_validation_value,
                        monitor=args.monitor,
                        patience_count=patience_count,
                        arguments=argument_dict,
                        current_metrics=checkpoint_metrics,
                    )
                    test_search_state["best_test_acc7"] = {
                        "epoch": epoch,
                        "value": current_acc7,
                        "metrics": test_monitor_metrics,
                        "checkpoint": "best_test_acc7.pt",
                    }
                    print(
                        f"Epoch {epoch:03d} | exploratory best Test Acc_7="
                        f"{current_acc7:.4f} checkpoint=best_test_acc7.pt",
                        flush=True,
                    )

                best_test_acc5 = test_search_state.get("best_test_acc5")
                current_acc5 = test_monitor_metrics.get("Acc_5")
                if current_acc5 is not None and (
                    best_test_acc5 is None
                    or float(current_acc5) > float(best_test_acc5["value"])
                ):
                    save_checkpoint(
                        output_dir / "best_test_acc5.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        monitor=args.monitor,
                        best_validation_value=best_validation_value,
                        patience_count=patience_count,
                        arguments=argument_dict,
                        current_metrics=checkpoint_metrics,
                    )
                    test_search_state["best_test_acc5"] = {
                        "epoch": epoch,
                        "value": float(current_acc5),
                        "metrics": test_monitor_metrics,
                        "checkpoint": "best_test_acc5.pt",
                    }
                    print(
                        f"Epoch {epoch:03d} | exploratory best Test Acc_5="
                        f"{float(current_acc5):.4f} checkpoint=best_test_acc5.pt",
                        flush=True,
                    )

                target_checkpoint_arguments = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "scaler": scaler,
                    "epoch": epoch,
                    "monitor": args.monitor,
                    "best_validation_value": best_validation_value,
                    "patience_count": patience_count,
                    "arguments": argument_dict,
                    "current_metrics": checkpoint_metrics,
                }
                for target_group in target_groups:
                    if meets_target_group(test_monitor_metrics, target_group):
                        target_hit = record_test_target_hit(
                            output_dir=output_dir,
                            state=test_search_state,
                            hit_key=target_group["hit_key"],
                            best_key=target_group["best_key"],
                            checkpoint_stem=target_group["checkpoint_stem"],
                            priority_metric=target_group["priority_metric"],
                            epoch=epoch,
                            metrics=test_monitor_metrics,
                            checkpoint_arguments=target_checkpoint_arguments,
                        )
                        print(
                            f"Epoch {epoch:03d} | EXPLORATORY TEST TARGET "
                            f"{target_group['name']} MET | "
                            f"checkpoint={target_hit['epoch_checkpoint']}",
                            flush=True,
                        )
                write_json_atomic(test_search_path, test_search_state)

            run_metadata.update(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_value": best_validation_value,
                    "best_validation_mae": valid_metrics["MAE"],
                    "monitor": args.monitor,
                    "patience_count": patience_count,
                    "latest_metrics": epoch_record,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            write_json_atomic(status_path, run_metadata)
            if not args.disable_early_stopping and patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

        best_checkpoint = restore_checkpoint(output_dir / "best.pt", model=model)
        test_metrics = run_epoch(
            model,
            loaders["test"],
            device=device,
            optimizer=None,
            scheduler=None,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            auxiliary_weight=args.auxiliary_weight,
            correlation_weight=args.correlation_weight,
            router_weight=args.router_weight,
            max_batches=args.max_eval_batches,
            dataset_name=args.dataset_name,
            output_activation=args.output_activation,
        )
        final_result = {
            "selection_protocol": (
                f"Best checkpoint selected only by validation {args.monitor}. Per-epoch test "
                "monitoring, when enabled, is diagnostic only and does not affect selection."
            ),
            "best_epoch": int(best_checkpoint["epoch"]),
            "monitor": args.monitor,
            "best_validation_value": float(
                best_checkpoint.get(
                    "best_validation_value", best_checkpoint["best_validation_mae"]
                )
            ),
            "best_validation_mae": float(best_checkpoint["metrics"]["valid"]["MAE"]),
            "test": test_metrics,
            "arguments": argument_dict,
            "model_config": configuration,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "split_sizes": split_sizes,
        }
        if args.evaluate_context_controls:
            final_result["context_control_test"] = evaluate_context_controls(
                model,
                loaders["test"],
                baseline_metrics=test_metrics,
                run_epoch_arguments={
                    "device": device,
                    "optimizer": None,
                    "scheduler": None,
                    "scaler": scaler,
                    "use_amp": use_amp,
                    "grad_clip": args.grad_clip,
                    "auxiliary_weight": args.auxiliary_weight,
                    "correlation_weight": args.correlation_weight,
                    "router_weight": args.router_weight,
                    "max_batches": args.max_eval_batches,
                    "dataset_name": args.dataset_name,
                    "output_activation": args.output_activation,
                },
            )
        if args.enable_test_target_checkpoints:
            final_result["exploratory_test_search"] = test_search_state
        if args.enable_validation_acc7_checkpoint:
            validation_acc7_checkpoint = restore_checkpoint(
                output_dir / "best_valid_acc7.pt", model=model
            )
            validation_acc7_test_metrics = run_epoch(
                model,
                loaders["test"],
                device=device,
                optimizer=None,
                scheduler=None,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                auxiliary_weight=args.auxiliary_weight,
                correlation_weight=args.correlation_weight,
                router_weight=args.router_weight,
                max_batches=args.max_eval_batches,
                dataset_name=args.dataset_name,
                output_activation=args.output_activation,
            )
            final_result["validation_acc7_selection"] = {
                "protocol": (
                    "Checkpoint selected only by maximum validation Acc-7, with "
                    "validation MAE as a tie-breaker. Test metrics do not affect selection."
                ),
                "epoch": int(validation_acc7_checkpoint["epoch"]),
                "validation": validation_acc7_checkpoint["metrics"]["valid"],
                "test": validation_acc7_test_metrics,
                "checkpoint": "best_valid_acc7.pt",
            }
        if args.enable_validation_acc5_checkpoint:
            validation_acc5_checkpoint = restore_checkpoint(
                output_dir / "best_valid_acc5.pt", model=model
            )
            validation_acc5_test_metrics = run_epoch(
                model,
                loaders["test"],
                device=device,
                optimizer=None,
                scheduler=None,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                auxiliary_weight=args.auxiliary_weight,
                correlation_weight=args.correlation_weight,
                router_weight=args.router_weight,
                max_batches=args.max_eval_batches,
                dataset_name=args.dataset_name,
                output_activation=args.output_activation,
            )
            final_result["validation_acc5_selection"] = {
                "protocol": (
                    "Checkpoint selected only by maximum validation Acc-5, with "
                    "validation MAE as a tie-breaker. Test metrics do not affect selection."
                ),
                "epoch": int(validation_acc5_checkpoint["epoch"]),
                "validation": validation_acc5_checkpoint["metrics"]["valid"],
                "test": validation_acc5_test_metrics,
                "checkpoint": "best_valid_acc5.pt",
            }
        write_json_atomic(output_dir / "final_result.json", final_result)
        print(f"Test | {format_metrics(test_metrics)}", flush=True)
        run_metadata.update(
            {
                "status": "completed",
                "best_epoch": final_result["best_epoch"],
                "monitor": args.monitor,
                "best_validation_value": final_result["best_validation_value"],
                "best_validation_mae": final_result["best_validation_mae"],
                "test_metrics": test_metrics,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        write_json_atomic(status_path, run_metadata)
    except Exception as error:
        run_metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        write_json_atomic(status_path, run_metadata)
        raise


if __name__ == "__main__":
    main()
