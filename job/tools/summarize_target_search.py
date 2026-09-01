#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_ablation import (
    collect_epoch_trajectories,
    collect_trials,
    load_json,
    mean_std,
    show,
)


SUMMARY_METRICS = (
    "best_validation_mae",
    "test_mae",
    "test_corr",
    "test_acc2",
    "test_f1",
    "test_acc5",
    "test_acc7",
    "valid_acc7_selected_test_mae",
    "valid_acc7_selected_test_corr",
    "valid_acc7_selected_test_acc2",
    "valid_acc7_selected_test_f1",
    "valid_acc7_selected_test_acc5",
    "valid_acc7_selected_test_acc7",
    "valid_acc5_selected_test_mae",
    "valid_acc5_selected_test_corr",
    "valid_acc5_selected_test_acc2",
    "valid_acc5_selected_test_f1",
    "valid_acc5_selected_test_acc5",
    "valid_acc5_selected_test_acc7",
)


CONFIG_FIELDS = (
    "epochs",
    "batch_size",
    "hidden_dim",
    "visual_depth",
    "audio_depth",
    "text_depth",
    "num_experts",
    "expert_dim",
    "dropout",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "auxiliary_weight",
    "correlation_weight",
    "router_weight",
    "monitor",
    "audio_modulation_mode",
    "text_modulation_mode",
    "audio_condition_scale_init",
    "text_condition_scale_init",
)


TARGET_HIT_FIELDS = (
    "target_name",
    "stage",
    "variant",
    "seed",
    "epoch",
    "Acc_7",
    "Acc_5",
    "Acc_3",
    "Acc_2",
    "F1",
    "MAE",
    "Corr",
    "epoch_checkpoint",
    "first_hit_checkpoint",
    "best_hit_checkpoint",
    "output_dir",
)


def write_dynamic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_groups_from_state(state: dict[str, Any]) -> list[dict[str, str]]:
    groups = []
    for name in (state.get("target_groups") or {}):
        groups.append(
            {
                "name": name,
                "hit_key": f"{name}_hits",
                "best_key": f"best_{name}_hit",
                "checkpoint_stem": f"{name}_hit",
            }
        )
    if not groups:
        groups = [
            {
                "name": "target",
                "hit_key": "target_hits",
                "best_key": "best_target_hit",
                "checkpoint_stem": "target_hit",
            },
            {
                "name": "secondary_target",
                "hit_key": "secondary_target_hits",
                "best_key": "best_secondary_target_hit",
                "checkpoint_stem": "secondary_target_hit",
            },
        ]
    return groups


def enrich_trials(trials: list[dict[str, Any]]) -> list[str]:
    target_names: set[str] = set()
    for trial in trials:
        state = load_json(Path(trial["output_dir"]) / "test_search_state.json")
        best = state.get("best_test_acc7") or {}
        best_metrics = best.get("metrics") or {}
        best_acc5 = state.get("best_test_acc5") or {}
        best_acc5_metrics = best_acc5.get("metrics") or {}
        trial.update(
            {
                "exploratory_best_acc5_epoch": best_acc5.get("epoch", ""),
                "exploratory_best_acc5": best_acc5_metrics.get("Acc_5", ""),
                "exploratory_best_acc7_epoch": best.get("epoch", ""),
                "exploratory_best_acc7": best_metrics.get("Acc_7", ""),
                "exploratory_best_acc2": best_metrics.get("Acc_2", ""),
                "exploratory_best_f1": best_metrics.get("F1", ""),
                "exploratory_best_mae": best_metrics.get("MAE", ""),
                "exploratory_best_corr": best_metrics.get("Corr", ""),
                "best_test_acc7_checkpoint": (
                    str(Path(trial["output_dir"]) / "best_test_acc7.pt") if best else ""
                ),
                "best_test_acc5_checkpoint": (
                    str(Path(trial["output_dir"]) / "best_test_acc5.pt")
                    if best_acc5
                    else ""
                ),
            }
        )
        for group in target_groups_from_state(state):
            target_names.add(group["name"])
            hits = state.get(group["hit_key"]) or []
            best_hit = state.get(group["best_key"]) or {}
            best_metrics = best_hit.get("metrics") or {}
            prefix = group["name"]
            trial.update(
                {
                    f"{prefix}_hit_count": len(hits),
                    f"{prefix}_first_epoch": hits[0]["epoch"] if hits else "",
                    f"{prefix}_best_epoch": best_hit.get("epoch", ""),
                    f"{prefix}_priority_metric": best_hit.get(
                        "priority_metric",
                        (state.get("target_groups") or {})
                        .get(group["name"], {})
                        .get("priority_metric", ""),
                    ),
                    f"{prefix}_best_acc5": best_metrics.get("Acc_5", ""),
                    f"{prefix}_best_acc7": best_metrics.get("Acc_7", ""),
                    f"{prefix}_best_acc2": best_metrics.get("Acc_2", ""),
                    f"{prefix}_best_f1": best_metrics.get("F1", ""),
                    f"{prefix}_best_mae": best_metrics.get("MAE", ""),
                    f"{prefix}_best_corr": best_metrics.get("Corr", ""),
                    f"{prefix}_checkpoint": (
                        str(Path(trial["output_dir"]) / best_hit["checkpoint"])
                        if best_hit.get("checkpoint")
                        else ""
                    ),
                }
            )
    return sorted(target_names)


def aggregate(trials: list[dict[str, Any]], target_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[(trial["stage"], trial["variant"])].append(trial)

    summaries = []
    for (stage, variant), rows in grouped.items():
        completed = [row for row in rows if row["state"] == "completed"]
        summary: dict[str, Any] = {
            "stage": stage,
            "variant": variant,
            "completed": len(completed),
            "expected": len(rows),
            **{field: rows[0].get(field, "") for field in CONFIG_FIELDS},
        }
        for target_name in target_names:
            hit_rows = [
                row for row in rows if int(row.get(f"{target_name}_hit_count") or 0) > 0
            ]
            summary[f"{target_name}_hit_seeds"] = len(hit_rows)
            summary[f"{target_name}_hit_epochs_total"] = sum(
                int(row.get(f"{target_name}_hit_count") or 0) for row in rows
            )
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in completed if row.get(metric, "") != ""]
            mean, std = mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        for metric in (
            "exploratory_best_acc5",
            "exploratory_best_acc7",
            "exploratory_best_acc2",
            "exploratory_best_f1",
            "exploratory_best_mae",
            "exploratory_best_corr",
        ):
            values = [float(row[metric]) for row in rows if row.get(metric, "") != ""]
            mean, std = mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            -sum(int(row.get(f"{name}_hit_seeds") or 0) for name in target_names),
            -(float(row.get("exploratory_best_acc5_mean") or -1.0)),
            -(float(row.get("exploratory_best_acc7_mean") or -1.0)),
            float(row.get("exploratory_best_mae_mean") or float("inf")),
            row["stage"],
            row["variant"],
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def collect_target_hit_epochs(
    trials: list[dict[str, Any]], target_names: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        output_dir = Path(trial["output_dir"])
        state = load_json(output_dir / "test_search_state.json")
        groups = {group["name"]: group for group in target_groups_from_state(state)}
        for target_name in target_names:
            group = groups.get(target_name)
            if not group:
                continue
            for hit_index, hit in enumerate(state.get(group["hit_key"]) or []):
                metrics = hit.get("metrics") or {}
                rows.append(
                    {
                        "target_name": target_name,
                        "stage": trial["stage"],
                        "variant": trial["variant"],
                        "seed": trial["seed"],
                        "epoch": hit.get("epoch", ""),
                        "Acc_7": metrics.get("Acc_7", ""),
                        "Acc_5": metrics.get("Acc_5", ""),
                        "Acc_3": metrics.get("Acc_3", ""),
                        "Acc_2": metrics.get("Acc_2", ""),
                        "F1": metrics.get("F1", ""),
                        "MAE": metrics.get("MAE", ""),
                        "Corr": metrics.get("Corr", ""),
                        "epoch_checkpoint": (
                            str(output_dir / hit["checkpoint"]) if hit.get("checkpoint") else ""
                        ),
                        "first_hit_checkpoint": (
                            str(output_dir / f"{group['checkpoint_stem']}.pt")
                            if hit.get("is_first") or hit_index == 0
                            else ""
                        ),
                        "best_hit_checkpoint": (
                            str(output_dir / best_checkpoint)
                            if (best_checkpoint := (state.get(group["best_key"]) or {}).get("checkpoint"))
                            else ""
                        ),
                        "output_dir": str(output_dir),
                    }
                )
    return rows


def write_target_hit_epochs(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_HIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V2A2T target search.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    trials = collect_trials(args.run_dir)
    target_names = enrich_trials(trials)
    summaries = aggregate(trials, target_names)
    trajectories = collect_epoch_trajectories(trials)
    target_hit_epochs = collect_target_hit_epochs(trials, target_names)
    target_hit_trials = [
        row
        for row in trials
        if any(int(row.get(f"{target_name}_hit_count") or 0) > 0 for target_name in target_names)
    ]

    write_dynamic_csv(args.run_dir / "target_search_trials.csv", trials)
    write_dynamic_csv(args.run_dir / "target_search_summary.csv", summaries)
    write_dynamic_csv(args.run_dir / "target_search_epoch_trajectories.csv", trajectories)
    write_dynamic_csv(args.run_dir / "target_search_target_hit_trials.csv", target_hit_trials)
    write_target_hit_epochs(
        args.run_dir / "target_search_target_hit_epochs.csv", target_hit_epochs
    )
    (args.run_dir / "target_search_summary.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "Exploratory target hits are selected from per-epoch test monitoring. "
                    "Use validation-selected checkpoints for formal model selection."
                ),
                "target_names": target_names,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    hit_columns = " ".join(f"{name[:10]:>10}" for name in target_names)
    print(
        f"{'Rank':>4} {'Stage':<25} {'Variant':<28} {'Done':>7} {hit_columns} "
        f"{'A5-Val':>8} {'A5-Peak*':>9} {'A7-Peak*':>9} {'Acc2*':>8} {'F1*':>8} {'MAE*':>8} {'Corr*':>8}"
    )
    for row in summaries:
        hit_values = " ".join(
            f"{row.get(f'{name}_hit_seeds', 0):>3}/{row['expected']:<6}"
            for name in target_names
        )
        print(
            f"{row['rank']:>4} {row['stage']:<25} {row['variant']:<28} "
            f"{row['completed']:>3}/{row['expected']:<3} {hit_values} "
            f"{show(row.get('valid_acc5_selected_test_acc5_mean', '')):>8} "
            f"{show(row.get('exploratory_best_acc5_mean', '')):>9} "
            f"{show(row.get('exploratory_best_acc7_mean', '')):>9} "
            f"{show(row.get('exploratory_best_acc2_mean', '')):>8} "
            f"{show(row.get('exploratory_best_f1_mean', '')):>8} "
            f"{show(row.get('exploratory_best_mae_mean', '')):>8} "
            f"{show(row.get('exploratory_best_corr_mean', '')):>8}"
        )
    print("\nA5-Val: Test result from validation-Acc5 checkpoint.")
    print("A5/A7-Peak*: per-epoch test-driven exploratory peaks; not formal selection.")
    print(f"Target-hit trials: {len(target_hit_trials)}/{len(trials)}")
    print(f"Target-hit epochs: {len(target_hit_epochs)}")
    print(f"Trial table: {args.run_dir / 'target_search_trials.csv'}")
    print(f"Variant summary: {args.run_dir / 'target_search_summary.csv'}")
    print(f"Target-hit trials: {args.run_dir / 'target_search_target_hit_trials.csv'}")
    print(f"Target-hit epochs: {args.run_dir / 'target_search_target_hit_epochs.csv'}")
    print(f"Epoch trajectories: {args.run_dir / 'target_search_epoch_trajectories.csv'}")


if __name__ == "__main__":
    main()
