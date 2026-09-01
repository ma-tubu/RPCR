#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = {
    "best_validation_mae": ("best_validation_mae",),
    "test_mae": ("test", "MAE"),
    "test_corr": ("test", "Corr"),
    "test_acc2": ("test", "Acc_2"),
    "test_f1": ("test", "F1"),
    "test_acc5": ("test", "Acc_5"),
    "test_acc7": ("test", "Acc_7"),
    "valid_acc7_selected_epoch": ("validation_acc7_selection", "epoch"),
    "valid_acc7_selected_valid_acc7": (
        "validation_acc7_selection",
        "validation",
        "Acc_7",
    ),
    "valid_acc7_selected_test_mae": (
        "validation_acc7_selection",
        "test",
        "MAE",
    ),
    "valid_acc7_selected_test_corr": (
        "validation_acc7_selection",
        "test",
        "Corr",
    ),
    "valid_acc7_selected_test_acc2": (
        "validation_acc7_selection",
        "test",
        "Acc_2",
    ),
    "valid_acc7_selected_test_f1": (
        "validation_acc7_selection",
        "test",
        "F1",
    ),
    "valid_acc7_selected_test_acc5": (
        "validation_acc7_selection",
        "test",
        "Acc_5",
    ),
    "valid_acc7_selected_test_acc7": (
        "validation_acc7_selection",
        "test",
        "Acc_7",
    ),
    "valid_acc5_selected_epoch": ("validation_acc5_selection", "epoch"),
    "valid_acc5_selected_valid_acc5": (
        "validation_acc5_selection",
        "validation",
        "Acc_5",
    ),
    "valid_acc5_selected_test_mae": (
        "validation_acc5_selection",
        "test",
        "MAE",
    ),
    "valid_acc5_selected_test_corr": (
        "validation_acc5_selection",
        "test",
        "Corr",
    ),
    "valid_acc5_selected_test_acc2": (
        "validation_acc5_selection",
        "test",
        "Acc_2",
    ),
    "valid_acc5_selected_test_f1": (
        "validation_acc5_selection",
        "test",
        "F1",
    ),
    "valid_acc5_selected_test_acc5": (
        "validation_acc5_selection",
        "test",
        "Acc_5",
    ),
    "valid_acc5_selected_test_acc7": (
        "validation_acc5_selection",
        "test",
        "Acc_7",
    ),
    "audio_relative_change": ("test", "route_audio_relative_change"),
    "audio_cosine_similarity": ("test", "route_audio_cosine_similarity"),
    "text_relative_change": ("test", "route_text_relative_change"),
    "text_cosine_similarity": ("test", "route_text_cosine_similarity"),
    "audio_condition_scale_mean": ("test", "route_audio_condition_scale_mean"),
    "text_condition_scale_mean": ("test", "route_text_condition_scale_mean"),
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return ""
        value = value[key]
    return value


def manifest_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = []
    manifest_dir = run_dir / "control" / "manifests"
    for path in sorted(manifest_dir.glob("*.tsv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


def collect_trials(run_dir: Path) -> list[dict[str, Any]]:
    trials = []
    for manifest in manifest_rows(run_dir):
        trial_dir = (
            run_dir
            / manifest["stage"]
            / manifest["variant"]
            / f"seed_{manifest['seed']}"
        )
        status = load_json(trial_dir / "status.json")
        result = load_json(trial_dir / "final_result.json")
        state = "completed" if result else status.get("status", "pending")
        trial: dict[str, Any] = {
            **manifest,
            "state": state,
            "epoch": status.get("epoch", ""),
            "output_dir": str(trial_dir),
        }
        for name, path in METRICS.items():
            trial[name] = nested_value(result, path)
        trials.append(trial)
    return trials


def mean_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[(trial["stage"], trial["variant"])].append(trial)

    baseline_by_seed = {
        str(trial["seed"]): trial
        for trial in trials
        if trial["variant"] == "baseline" and trial["state"] == "completed"
    }
    summaries = []
    for (stage, variant), rows in grouped.items():
        completed = [row for row in rows if row["state"] == "completed"]
        summary: dict[str, Any] = {
            "stage": stage,
            "variant": variant,
            "completed": len(completed),
            "expected": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in completed if row[metric] != ""]
            mean, std = mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        for metric in ("test_mae", "test_corr", "test_acc2", "test_f1", "test_acc7"):
            paired_deltas = []
            for row in completed:
                baseline = baseline_by_seed.get(str(row["seed"]))
                if baseline and row[metric] != "" and baseline[metric] != "":
                    paired_deltas.append(float(row[metric]) - float(baseline[metric]))
            delta_mean, delta_std = mean_std(paired_deltas)
            summary[f"{metric}_delta_vs_baseline_mean"] = delta_mean
            summary[f"{metric}_delta_vs_baseline_std"] = delta_std
        summaries.append(summary)
    return summaries


def collect_epoch_trajectories(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajectories = []
    for trial in trials:
        metrics_path = Path(trial["output_dir"]) / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                train = record.get("train", {})
                valid = record.get("valid", {})
                test_monitor = record.get("test_monitor", {})
                valid_mae = valid.get("MAE", "")
                test_mae = test_monitor.get("MAE", "")
                gap = ""
                if valid_mae != "" and test_mae != "":
                    gap = float(test_mae) - float(valid_mae)
                trajectories.append(
                    {
                        "stage": trial["stage"],
                        "variant": trial["variant"],
                        "seed": trial["seed"],
                        "epoch": record.get("epoch", ""),
                        "train_mae": train.get("MAE", ""),
                        "valid_mae": valid_mae,
                        "test_monitor_mae": test_mae,
                        "test_minus_valid_mae": gap,
                        "valid_corr": valid.get("Corr", ""),
                        "test_monitor_corr": test_monitor.get("Corr", ""),
                        "valid_acc2": valid.get("Acc_2", ""),
                        "test_monitor_acc2": test_monitor.get("Acc_2", ""),
                        "valid_f1": valid.get("F1", ""),
                        "test_monitor_f1": test_monitor.get("F1", ""),
                        "valid_acc5": valid.get("Acc_5", ""),
                        "test_monitor_acc5": test_monitor.get("Acc_5", ""),
                        "valid_acc7": valid.get("Acc_7", ""),
                        "test_monitor_acc7": test_monitor.get("Acc_7", ""),
                        "valid_audio_relative_change": valid.get(
                            "route_audio_relative_change", ""
                        ),
                        "test_audio_relative_change": test_monitor.get(
                            "route_audio_relative_change", ""
                        ),
                        "test_audio_cosine_similarity": test_monitor.get(
                            "route_audio_cosine_similarity", ""
                        ),
                        "valid_text_relative_change": valid.get(
                            "route_text_relative_change", ""
                        ),
                        "test_text_relative_change": test_monitor.get(
                            "route_text_relative_change", ""
                        ),
                        "test_text_cosine_similarity": test_monitor.get(
                            "route_text_cosine_similarity", ""
                        ),
                        "test_audio_condition_scale_mean": test_monitor.get(
                            "route_audio_condition_scale_mean", ""
                        ),
                        "test_text_condition_scale_mean": test_monitor.get(
                            "route_text_condition_scale_mean", ""
                        ),
                        **{
                            f"test_audio_layer_{layer_index}_{metric}": test_monitor.get(
                                f"route_audio_layer_{layer_index}_{metric}", ""
                            )
                            for layer_index in range(1, 5)
                            for metric in (
                                "relative_change",
                                "cosine_similarity",
                                "condition_scale",
                            )
                        },
                        **{
                            f"test_text_layer_{layer_index}_{metric}": test_monitor.get(
                                f"route_text_layer_{layer_index}_{metric}", ""
                            )
                            for layer_index in range(1, 5)
                            for metric in (
                                "relative_change",
                                "cosine_similarity",
                                "condition_scale",
                            )
                        },
                    }
                )
    return trajectories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def show(value: Any) -> str:
    return "-" if value == "" or value is None else f"{float(value):.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize staged V2A2T ablations.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    trials = collect_trials(args.run_dir)
    summaries = aggregate(trials)
    trajectories = collect_epoch_trajectories(trials)
    write_csv(args.run_dir / "ablation_trials.csv", trials)
    write_csv(args.run_dir / "ablation_summary.csv", summaries)
    write_csv(args.run_dir / "ablation_epoch_trajectories.csv", trajectories)
    (args.run_dir / "ablation_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"{'Stage':<25} {'Variant':<30} {'Done':>7} {'Val MAE':>9} "
        f"{'Test MAE':>9} {'dMAE':>9} {'Corr':>9} {'dCorr':>9} "
        f"{'A-change':>9} {'A-cos':>9} {'T-change':>9} {'T-cos':>9}"
    )
    for row in summaries:
        print(
            f"{row['stage']:<25} {row['variant']:<30} "
            f"{row['completed']:>3}/{row['expected']:<3} "
            f"{show(row['best_validation_mae_mean']):>9} "
            f"{show(row['test_mae_mean']):>9} "
            f"{show(row['test_mae_delta_vs_baseline_mean']):>9} "
            f"{show(row['test_corr_mean']):>9} "
            f"{show(row['test_corr_delta_vs_baseline_mean']):>9} "
            f"{show(row['audio_relative_change_mean']):>9} "
            f"{show(row['audio_cosine_similarity_mean']):>9} "
            f"{show(row['text_relative_change_mean']):>9} "
            f"{show(row['text_cosine_similarity_mean']):>9}"
        )
    completed = sum(1 for trial in trials if trial["state"] == "completed")
    print(f"\nCompleted trials: {completed}/{len(trials)}")
    print(f"Trial table: {args.run_dir / 'ablation_trials.csv'}")
    print(f"Variant summary: {args.run_dir / 'ablation_summary.csv'}")
    print(f"Epoch trajectories: {args.run_dir / 'ablation_epoch_trajectories.csv'}")


if __name__ == "__main__":
    main()
