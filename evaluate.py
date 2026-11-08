from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import RPCR
from utils.checkpoint import write_json_atomic
from utils.engine import make_loader, resolve_device, run_epoch
from utils.feature_dataset import CachedFeatureDataset, load_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an RPCR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_args = checkpoint["arguments"]
    configuration = checkpoint["model_config"]
    cache = load_feature_cache(args.cache_path)
    dataset_name = str(training_args["dataset_name"])
    cache_name = str(cache.get("dataset_name", ""))
    if cache_name and cache_name.upper() != dataset_name.upper():
        raise ValueError(f"Dataset mismatch: checkpoint={dataset_name}, cache={cache_name}.")

    model = RPCR(**configuration).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = make_loader(
        CachedFeatureDataset(cache, args.split),
        batch_size=args.batch_size or int(training_args["batch_size"]),
        shuffle=False,
        num_workers=0,
        seed=int(training_args["seed"]),
        pin_memory=device.type == "cuda",
    )
    use_amp = (args.precision or training_args["precision"]) == "fp16" and device.type == "cuda"
    output_activation = training_args.get("output_activation", "auto")
    if output_activation == "auto":
        output_activation = "tanh" if dataset_name.upper() == "CHSIMS" else "identity"
    metrics = run_epoch(
        model,
        loader,
        device=device,
        optimizer=None,
        scheduler=None,
        scaler=torch.cuda.amp.GradScaler(enabled=use_amp),
        use_amp=use_amp,
        grad_clip=float(training_args["grad_clip"]),
        auxiliary_weight=float(training_args["auxiliary_weight"]),
        correlation_weight=float(training_args["correlation_weight"]),
        router_weight=float(training_args["router_weight"]),
        max_batches=None,
        dataset_name=dataset_name,
        output_activation=output_activation,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
