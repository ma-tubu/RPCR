#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRIC_NAMES = (
    "Acc_2",
    "Acc_2_has0",
    "Acc_3",
    "Acc_5",
    "Acc_7",
    "F1",
    "MAE",
    "Corr",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def expected_trials(run_dir: Path) -> list[tuple[str, int | None]]:
    manifest = run_dir / "manifest.tsv"
    if not manifest.is_file():
        return [(path.name, None) for path in sorted(run_dir.glob("seed_*")) if path.is_dir()]
    rows: list[tuple[str, int | None]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append((row["trial_id"], int(row["seed"])))
    return rows


def collect(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for trial_id, seed in expected_trials(run_dir):
        trial_dir = run_dir / trial_id
        status = load_json(trial_dir / "status.json")
        result = load_json(trial_dir / "final_result.json")
        latest = status.get("latest_metrics", {})
        latest_valid = latest.get("valid", {})
        latest_test_monitor = latest.get("test_monitor", {})
        test = result.get("test", {})
        state = "pending"
        if status:
            state = str(status.get("status", "unknown"))
        if result:
            state = "completed"
        row: dict[str, Any] = {
            "trial_id": trial_id,
            "seed": seed if seed is not None else status.get("arguments", {}).get("seed"),
            "state": state,
            "epoch": status.get("epoch", ""),
            "best_epoch": result.get("best_epoch", status.get("best_epoch", "")),
            "best_validation_mae": result.get(
                "best_validation_mae", status.get("best_validation_mae", "")
            ),
            "latest_validation_mae": latest_valid.get("MAE", ""),
            "latest_test_monitor_mae": latest_test_monitor.get("MAE", ""),
            "latest_test_monitor_corr": latest_test_monitor.get("Corr", ""),
            "latest_test_monitor_acc2": latest_test_monitor.get("Acc_2", ""),
        }
        if row["latest_validation_mae"] != "" and row["latest_test_monitor_mae"] != "":
            row["latest_monitor_mae_gap"] = (
                float(row["latest_test_monitor_mae"])
                - float(row["latest_validation_mae"])
            )
        else:
            row["latest_monitor_mae_gap"] = ""
        for metric in METRIC_NAMES:
            row[f"test_{metric}"] = test.get(metric, "")
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["state"] == "completed"]
    summary: dict[str, Any] = {"completed": len(completed), "total": len(rows), "metrics": {}}
    for metric in METRIC_NAMES:
        values = [
            float(row[f"test_{metric}"])
            for row in completed
            if row[f"test_{metric}"] != ""
        ]
        if not values:
            continue
        summary["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V2A2T Slurm array results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(args.run_dir)
    summary = aggregate(rows)
    fieldnames = [
        "trial_id",
        "seed",
        "state",
        "epoch",
        "best_epoch",
        "best_validation_mae",
        "latest_validation_mae",
        "latest_test_monitor_mae",
        "latest_monitor_mae_gap",
        "latest_test_monitor_corr",
        "latest_test_monitor_acc2",
        *(f"test_{metric}" for metric in METRIC_NAMES),
    ]
    with (args.run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"{'Trial':<16} {'Seed':>7} {'State':<10} {'Epoch':>7} "
        f"{'Val MAE':>9} {'TestMon':>9} {'Gap':>9} {'Best val':>9} {'Final MAE':>10}"
    )
    for row in rows:
        def show(value: Any, digits: int = 5) -> str:
            return "-" if value == "" or value is None else f"{float(value):.{digits}f}"

        print(
            f"{row['trial_id']:<16} {str(row['seed']):>7} {row['state']:<10} "
            f"{str(row['epoch'] or '-'):>7} {show(row['latest_validation_mae']):>9} "
            f"{show(row['latest_test_monitor_mae']):>9} "
            f"{show(row['latest_monitor_mae_gap']):>9} "
            f"{show(row['best_validation_mae']):>9} {show(row['test_MAE']):>10}"
        )

    print(f"\nCompleted: {summary['completed']}/{summary['total']}")
    for metric, values in summary["metrics"].items():
        print(f"{metric:<12} {values['mean']:.6f} +/- {values['std']:.6f}")
    print(f"Summary CSV: {args.run_dir / 'summary.csv'}")
    print(f"Summary JSON: {args.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
