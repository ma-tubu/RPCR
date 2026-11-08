from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from models import RPCR
from .configuration import model_config


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)



def save_checkpoint(
    path: Path,
    *,
    model: RPCR,
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
        "format": "rpcr_full_checkpoint_v1",
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
    model: RPCR,
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
