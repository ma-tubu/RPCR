from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models import RPCR, rpcr_loss
from .feature_dataset import CachedFeatureDataset
from .regression_metrics import regression_metrics


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
    dataset: CachedFeatureDataset,
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


def run_epoch(
    model: RPCR,
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
                loss, components = rpcr_loss(
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
    metrics["samples"] = float(sample_count)
    return metrics
