from __future__ import annotations

import json
import math
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models import RPCR
from utils.checkpoint import restore_checkpoint, save_checkpoint, write_json_atomic
from utils.configuration import model_config, parse_args
from utils.engine import (
    make_loader,
    resolve_device,
    run_epoch,
    seed_everything,
)
from utils.feature_dataset import CachedFeatureDataset, iter_split_sizes, load_feature_cache
from utils.regression_metrics import format_metrics


def monitor_improved(current: float, best: float, monitor: str) -> bool:
    return current < best if monitor in {"MAE", "loss"} else current > best


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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    if args.resume is None:
        existing_training_artifacts = [
            path
            for path in (
                metrics_path,
                output_dir / "best.pt",
                output_dir / "last.pt",
                output_dir / "final_result.json",
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

    cache = load_feature_cache(args.cache_path)
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
        split: CachedFeatureDataset(cache, split)
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
    model = RPCR(**configuration).to(device)
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
    argument_dict["config"] = str(args.config.resolve()) if args.config else None
    argument_dict["cache_path"] = str(args.cache_path.resolve())
    argument_dict["output_dir"] = str(output_dir)
    argument_dict["resume"] = str(args.resume.resolve()) if args.resume else None

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
        current_validation_value = float(valid_metrics[args.monitor])
        improved = monitor_improved(
            current_validation_value, best_validation_value, args.monitor
        )
        if improved:
            best_validation_value = current_validation_value
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

        checkpoint_metrics = {"train": train_metrics, "valid": valid_metrics}
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
            f"Checkpoint selected by validation {args.monitor}."
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
    write_json_atomic(output_dir / "final_result.json", final_result)
    print(f"Test | {format_metrics(test_metrics)}", flush=True)


if __name__ == "__main__":
    main()
