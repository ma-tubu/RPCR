from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.chsims_unaligned_v2a2t import CHSIMSUnalignedV2A2T
from models.feature_v2a2t import feature_v2a2t_loss
from utils.chsims_unaligned_dataset import load_chsims_unaligned_datasets
from utils.regression_metrics import chsims_regression_metrics, format_metrics


LOWER_IS_BETTER = {"MAE", "loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train chained V->A->T MoPE on CH-SIMS unaligned_39 features."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--clip-percentile-low", type=float, default=0.1)
    parser.add_argument("--clip-percentile-high", type=float, default=99.9)

    parser.add_argument("--sequence-dim", type=int, default=128)
    parser.add_argument("--sequence-output-dim", type=int, default=128)
    parser.add_argument("--sequence-layers", type=int, default=1)
    parser.add_argument("--sequence-dropout", type=float, default=0.20)
    parser.add_argument(
        "--output-activation", choices=("identity", "tanh"), default="tanh"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--visual-depth", type=int, default=3)
    parser.add_argument("--audio-depth", type=int, default=4)
    parser.add_argument("--text-depth", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--audio-condition-scale-init", type=float, default=0.01)
    parser.add_argument("--text-condition-scale-init", type=float, default=0.01)

    parser.add_argument("--learning-rate", type=float, default=0.00010)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--auxiliary-weight", type=float, default=0.02)
    parser.add_argument("--correlation-weight", type=float, default=0.0)
    parser.add_argument("--router-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=11)
    parser.add_argument(
        "--monitor",
        choices=("MAE", "Corr", "Acc_2", "Acc_3", "Acc_5", "F1", "loss"),
        default="F1",
    )
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
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


def chain_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "hidden_dim": args.hidden_dim,
        "visual_depth": args.visual_depth,
        "audio_depth": args.audio_depth,
        "text_depth": args.text_depth,
        "num_experts": args.num_experts,
        "expert_dim": args.expert_dim,
        "expansion": args.expansion,
        "dropout": args.dropout,
        "audio_encoder_mode": "mope",
        "text_encoder_mode": "mope",
        "audio_modulation_mode": "full",
        "text_modulation_mode": "full",
        "audio_condition_scale_init": args.audio_condition_scale_init,
        "text_condition_scale_init": args.text_condition_scale_init,
    }


def model_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "sequence_dim": args.sequence_dim,
        "sequence_output_dim": args.sequence_output_dim,
        "sequence_layers": args.sequence_layers,
        "sequence_dropout": args.sequence_dropout,
        "output_activation": args.output_activation,
        "chain_config": chain_configuration(args),
    }


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    *,
    model: CHSIMSUnalignedV2A2T,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_validation_value: float,
    patience_count: int,
    arguments: Dict[str, Any],
    current_metrics: Dict[str, Any],
) -> None:
    payload = {
        "format": "chsims_unaligned_v2a2t_full_checkpoint_v1",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_value": best_validation_value,
        "patience_count": patience_count,
        "arguments": arguments,
        "model_config": model_configuration(argparse.Namespace(**arguments)),
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    *,
    model: CHSIMSUnalignedV2A2T,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


def run_epoch(
    model: CHSIMSUnalignedV2A2T,
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
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    predictions = []
    labels = []
    route_totals: Dict[str, float] = {}

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_labels = batch["label"].to(device, non_blocking=True).reshape(-1)
        batch_size = batch_labels.shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast_context:
            output = model(
                vision=batch["vision"].to(device, non_blocking=True),
                vision_lengths=batch["vision_length"].to(device, non_blocking=True),
                audio=batch["audio"].to(device, non_blocking=True),
                audio_lengths=batch["audio_length"].to(device, non_blocking=True),
                text=batch["text"].to(device, non_blocking=True),
                text_lengths=batch["text_length"].to(device, non_blocking=True),
            )
            loss, _ = feature_v2a2t_loss(
                output,
                batch_labels,
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

        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        predictions.append(output.prediction.detach().float().cpu())
        labels.append(batch_labels.detach().float().cpu())
        for name, value in output.route_statistics.items():
            if value.numel() == 1:
                route_totals[name] = route_totals.get(name, 0.0) + float(value) * batch_size

    if total_samples == 0:
        raise RuntimeError("No batches were processed.")
    metrics = chsims_regression_metrics(
        torch.cat(predictions), torch.cat(labels)
    )
    metrics["loss"] = total_loss / total_samples
    for name, value in route_totals.items():
        metrics[f"route_{name}"] = value / total_samples
    return metrics


def is_improved(current: float, best: float, monitor: str) -> bool:
    return current < best if monitor in LOWER_IS_BETTER else current > best


def main() -> None:
    args = parse_args()
    if args.num_workers > 0 and os.name == "nt":
        print("Windows detected: forcing num_workers=0 for the large PKL dataset.")
        args.num_workers = 0
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    status_path = output_dir / "status.json"

    if args.resume is None:
        existing = [
            path
            for path in (
                metrics_path,
                output_dir / "best.pt",
                output_dir / "last.pt",
                output_dir / "final_result.json",
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"Output directory contains training artifacts: "
                f"{', '.join(path.name for path in existing)}"
            )

    print(f"Loading CH-SIMS from {args.data_path.resolve()}...", flush=True)
    datasets = load_chsims_unaligned_datasets(
        args.data_path,
        standardize=not args.no_standardize,
        clip_percentiles=(
            args.clip_percentile_low,
            args.clip_percentile_high,
        ),
    )
    pin_memory = device.type == "cuda"
    loaders = {
        split: make_loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            seed=args.seed + index,
            pin_memory=pin_memory,
        )
        for index, (split, dataset) in enumerate(datasets.items())
    }

    configuration = model_configuration(args)
    model = CHSIMSUnalignedV2A2T(**configuration).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = (
        min(len(loaders["train"]), args.max_train_batches)
        if args.max_train_batches is not None
        else len(loaders["train"])
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(total_steps * args.warmup_ratio),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_epoch = 0
    best_validation_value = (
        float("inf") if args.monitor in LOWER_IS_BETTER else float("-inf")
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
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_value = float(checkpoint["best_validation_value"])
        patience_count = int(checkpoint.get("patience_count", 0))

    argument_dict = vars(args).copy()
    argument_dict["data_path"] = str(args.data_path.resolve())
    argument_dict["output_dir"] = str(output_dir)
    argument_dict["resume"] = str(args.resume.resolve()) if args.resume else None
    dataset_summary = {split: dataset.summary() for split, dataset in datasets.items()}
    run_metadata = {
        "status": "running",
        "pid": os.getpid(),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "dataset": dataset_summary,
        "arguments": argument_dict,
        "model_config": configuration,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json_atomic(status_path, run_metadata)
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2), flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
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
                )

            current_value = float(valid_metrics[args.monitor])
            improved = is_improved(
                current_value, best_validation_value, args.monitor
            )
            if improved:
                best_validation_value = current_value
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            epoch_record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "valid": valid_metrics,
                "test_monitor": test_monitor_metrics,
                "monitor": args.monitor,
                "monitor_value": current_value,
                "improved": improved,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

            print(f"Epoch {epoch:03d} | train {format_metrics(train_metrics)}", flush=True)
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
            checkpoint_arguments = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_value": best_validation_value,
                "patience_count": patience_count,
                "arguments": argument_dict,
                "current_metrics": checkpoint_metrics,
            }
            save_checkpoint(output_dir / "last.pt", **checkpoint_arguments)
            if improved:
                save_checkpoint(output_dir / "best.pt", **checkpoint_arguments)

            run_metadata.update(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_value": best_validation_value,
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
        )
        final_result = {
            "selection_protocol": (
                f"Checkpoint selected only by validation {args.monitor}; "
                "per-epoch Test monitoring is diagnostic only."
            ),
            "best_epoch": int(best_checkpoint["epoch"]),
            "monitor": args.monitor,
            "best_validation_value": float(best_checkpoint["best_validation_value"]),
            "validation": best_checkpoint["metrics"]["valid"],
            "test": test_metrics,
            "arguments": argument_dict,
            "model_config": configuration,
            "dataset": dataset_summary,
            "checkpoint": "best.pt",
        }
        write_json_atomic(output_dir / "final_result.json", final_result)
        print(f"Test | {format_metrics(test_metrics)}", flush=True)
        run_metadata.update(
            {
                "status": "completed",
                "best_epoch": int(best_checkpoint["epoch"]),
                "best_validation_value": float(
                    best_checkpoint["best_validation_value"]
                ),
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
