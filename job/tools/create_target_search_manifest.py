#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "stage",
    "variant",
    "seed",
    "epochs",
    "batch_size",
    "hidden_dim",
    "visual_depth",
    "audio_depth",
    "text_depth",
    "num_experts",
    "expert_dim",
    "expansion",
    "dropout",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "grad_clip",
    "auxiliary_weight",
    "correlation_weight",
    "router_weight",
    "patience",
    "monitor",
    "output_activation",
    "audio_encoder_mode",
    "text_encoder_mode",
    "audio_modulation_mode",
    "text_modulation_mode",
    "audio_condition_scale_init",
    "text_condition_scale_init",
    "audio_condition_scale_growth",
    "text_condition_scale_growth",
)


BASELINES: dict[str, dict[str, Any]] = {
    "CMUMOSEI": {
        "epochs": 60,
        "batch_size": 256,
        "hidden_dim": 512,
        "visual_depth": 3,
        "audio_depth": 4,
        "text_depth": 4,
        "num_experts": 8,
        "expert_dim": 64,
        "expansion": 4,
        "dropout": 0.20,
        "learning_rate": 0.00030,
        "weight_decay": 0.00010,
        "warmup_ratio": 0.08,
        "grad_clip": 1.0,
        "auxiliary_weight": 0.10,
        "correlation_weight": 0.05,
        "router_weight": 0.01,
        "patience": 12,
        "monitor": "MAE",
        "output_activation": "identity",
        "audio_encoder_mode": "mope",
        "text_encoder_mode": "mope",
        "audio_modulation_mode": "none",
        "text_modulation_mode": "full",
        "audio_condition_scale_init": 0.01,
        "text_condition_scale_init": 0.01,
        "audio_condition_scale_growth": 1.0,
        "text_condition_scale_growth": 1.0,
    },
    "CHSIMS": {
        "epochs": 70,
        "batch_size": 24,
        "hidden_dim": 64,
        "visual_depth": 2,
        "audio_depth": 3,
        "text_depth": 3,
        "num_experts": 2,
        "expert_dim": 32,
        "expansion": 2,
        "dropout": 0.35,
        "learning_rate": 0.00020,
        "weight_decay": 0.002,
        "warmup_ratio": 0.10,
        "grad_clip": 0.5,
        "auxiliary_weight": 0.03,
        "correlation_weight": 0.05,
        "router_weight": 0.0,
        "patience": 10,
        "monitor": "Acc_5",
        "output_activation": "tanh",
        "audio_encoder_mode": "mope",
        "text_encoder_mode": "mope",
        "audio_modulation_mode": "full",
        "text_modulation_mode": "full",
        "audio_condition_scale_init": 0.03,
        "text_condition_scale_init": 0.03,
        "audio_condition_scale_growth": 1.0,
        "text_condition_scale_growth": 1.0,
    },
}


DIRECTIONS: dict[str, dict[str, list[tuple[str, dict[str, Any]]]]] = {
    "CMUMOSEI": {
        "stage0_reference": [
            ("mosei_audio_mod_none", {}),
            ("mosei_audio_mod_full", {"audio_modulation_mode": "full"}),
            ("mosei_monitor_acc7", {"monitor": "Acc_7"}),
        ],
        "stage1_audio_text_scale": [
            ("audio003_text010", {"audio_condition_scale_init": 0.003}),
            ("audio010_text0075", {"text_condition_scale_init": 0.0075}),
            ("audio030_text010", {"audio_condition_scale_init": 0.030}),
            ("audio010_text030", {"text_condition_scale_init": 0.030}),
            (
                "audio030_text030",
                {"audio_condition_scale_init": 0.030, "text_condition_scale_init": 0.030},
            ),
        ],
        "stage2_depth_pairs": [
            ("depth_audio2_text16", {"audio_depth": 2, "text_depth": 16}),
            ("depth_audio6_text2", {"audio_depth": 6, "text_depth": 2}),
            ("depth_audio6_text6", {"audio_depth": 6, "text_depth": 6}),
            ("depth_audio12_text4", {"audio_depth": 12, "text_depth": 4}),
        ],
        "stage3_expert_budget": [
            ("expert1_dim128", {"num_experts": 1, "expert_dim": 128, "router_weight": 0.0}),
            ("expert2_dim128", {"num_experts": 2, "expert_dim": 128}),
            ("expert16_dim32", {"num_experts": 16, "expert_dim": 32}),
            ("expert_dim128", {"expert_dim": 128}),
        ],
        "stage4_optimization": [
            ("lr250_wd0001", {"learning_rate": 0.00025}),
            ("lr350_wd0001", {"learning_rate": 0.00035}),
            ("aux075_corr050", {"auxiliary_weight": 0.075}),
            ("aux100_corr000", {"correlation_weight": 0.0}),
            ("router000", {"router_weight": 0.0}),
        ],
    },
    "CHSIMS": {
        "stage0_reference": [
            ("small64_monitor_acc5", {}),
            ("small64_monitor_mae", {"monitor": "MAE"}),
            ("small64_monitor_acc7_diagnostic", {"monitor": "Acc_7"}),
        ],
        "stage1_capacity": [
            ("hidden48", {"hidden_dim": 48}),
            ("hidden80", {"hidden_dim": 80}),
            ("hidden96", {"hidden_dim": 96}),
            ("expert1_dim32", {"num_experts": 1, "router_weight": 0.0}),
            ("expert2_dim64", {"expert_dim": 64}),
            ("expert4_dim32", {"num_experts": 4, "expert_dim": 32}),
        ],
        "stage2_depth_pairs": [
            ("depth_v1_a3_t3", {"visual_depth": 1}),
            ("depth_v2_a2_t2", {"audio_depth": 2, "text_depth": 2}),
            ("depth_v2_a4_t2", {"audio_depth": 4, "text_depth": 2}),
            ("depth_v2_a12_t2", {"audio_depth": 12, "text_depth": 2}),
            ("depth_v2_a12_t4", {"audio_depth": 12, "text_depth": 4}),
        ],
        "stage3_regularization": [
            ("dropout030", {"dropout": 0.30}),
            ("dropout040", {"dropout": 0.40}),
            ("dropout045", {"dropout": 0.45}),
            ("wd001", {"weight_decay": 0.001}),
            ("wd003", {"weight_decay": 0.003}),
        ],
        "stage4_loss_scale": [
            ("aux020_corr050", {"auxiliary_weight": 0.02}),
            ("aux050_corr050", {"auxiliary_weight": 0.05}),
            ("aux030_corr030", {"correlation_weight": 0.03}),
            ("aux030_corr080", {"correlation_weight": 0.08}),
            ("scale050", {"audio_condition_scale_init": 0.05, "text_condition_scale_init": 0.05}),
        ],
    },
}


DEFAULT_SEEDS = {
    "CMUMOSEI": "42,3407,2024",
    "CHSIMS": "42,3407,2024,1,7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create target-search manifest.")
    parser.add_argument("--dataset", choices=["CMUMOSEI", "CHSIMS"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--variants", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds_text = args.seeds or DEFAULT_SEEDS[args.dataset]
    seeds = [int(value.strip()) for value in seeds_text.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty unique list.")

    selected = {value.strip() for value in args.variants.split(",") if value.strip()}
    known = {
        variant
        for variants in DIRECTIONS[args.dataset].values()
        for variant, _ in variants
    }
    unknown = selected.difference(known)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    rows = []
    counts = {}
    for stage, variants in DIRECTIONS[args.dataset].items():
        stage_count = 0
        for variant, overrides in variants:
            if selected and variant not in selected:
                continue
            configuration = {**BASELINES[args.dataset], **overrides}
            for seed in seeds:
                rows.append(
                    {"stage": stage, "variant": variant, "seed": seed, **configuration}
                )
                stage_count += 1
        if stage_count:
            counts[stage] = stage_count

    if not rows:
        raise ValueError("No trials selected.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "total": len(rows),
                "variants": len(rows) // len(seeds),
                "seeds": seeds,
                "stages": counts,
                "manifest": str(args.output),
                "baseline": BASELINES[args.dataset],
            }
        )
    )


if __name__ == "__main__":
    main()
