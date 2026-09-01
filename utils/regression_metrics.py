from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def as_flat_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).astype(np.float64)


def multiclass_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    return float(np.mean(np.round(predictions) == np.round(labels)))


def mosei_regression_metrics(predictions, labels) -> Dict[str, float]:
    predictions = as_flat_numpy(predictions)
    labels = as_flat_numpy(labels)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Predictions and labels must match: {predictions.shape} vs {labels.shape}."
        )
    if predictions.size == 0:
        raise ValueError("Cannot compute metrics on empty predictions.")

    clipped_7_predictions = np.clip(predictions, -3.0, 3.0)
    clipped_7_labels = np.clip(labels, -3.0, 3.0)
    clipped_5_predictions = np.clip(predictions, -2.0, 2.0)
    clipped_5_labels = np.clip(labels, -2.0, 2.0)

    absolute_errors = np.abs(predictions - labels)
    squared_errors = np.square(predictions - labels)
    mae = float(np.mean(absolute_errors))
    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    if np.std(predictions) < 1e-12 or np.std(labels) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(predictions, labels)[0, 1])

    non_zero = labels != 0
    if np.any(non_zero):
        acc_2 = float(
            accuracy_score(labels[non_zero] > 0, predictions[non_zero] > 0)
        )
        f1 = float(
            f1_score(
                labels[non_zero] > 0,
                predictions[non_zero] > 0,
                average="weighted",
                zero_division=0,
            )
        )
    else:
        acc_2 = 0.0
        f1 = 0.0

    acc_2_has0 = float(accuracy_score(labels >= 0, predictions >= 0))
    f1_has0 = float(
        f1_score(
            labels >= 0,
            predictions >= 0,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "Acc_2": acc_2,
        "Acc_2_has0": acc_2_has0,
        "Acc_5": multiclass_accuracy(clipped_5_predictions, clipped_5_labels),
        "Acc_7": multiclass_accuracy(clipped_7_predictions, clipped_7_labels),
        "F1": f1,
        "F1_has0": f1_has0,
        "MAE": mae,
        "Corr": correlation,
        "MSE": mse,
        "RMSE": rmse,
    }


def chsims_regression_metrics(predictions, labels) -> Dict[str, float]:
    predictions = np.clip(as_flat_numpy(predictions), -1.0, 1.0)
    labels = np.clip(as_flat_numpy(labels), -1.0, 1.0)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Predictions and labels must match: {predictions.shape} vs {labels.shape}."
        )
    if predictions.size == 0:
        raise ValueError("Cannot compute metrics on empty predictions.")

    pred_2 = (predictions > 0.0).astype(np.int64)
    label_2 = (labels > 0.0).astype(np.int64)
    pred_3 = np.digitize(predictions, [-0.1, 0.1], right=True)
    label_3 = np.digitize(labels, [-0.1, 0.1], right=True)
    pred_5 = np.digitize(predictions, [-0.7, -0.1, 0.1, 0.7], right=True)
    label_5 = np.digitize(labels, [-0.7, -0.1, 0.1, 0.7], right=True)
    pred_7 = np.rint(predictions * 3.0).astype(np.int64)
    label_7 = np.rint(labels * 3.0).astype(np.int64)

    absolute_errors = np.abs(predictions - labels)
    squared_errors = np.square(predictions - labels)
    mae = float(np.mean(absolute_errors))
    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    if np.std(predictions) < 1e-12 or np.std(labels) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(predictions, labels)[0, 1])
    return {
        "Acc_2": float(accuracy_score(label_2, pred_2)),
        "Acc_3": float(accuracy_score(label_3, pred_3)),
        "Acc_5": float(accuracy_score(label_5, pred_5)),
        "Acc_7": float(accuracy_score(label_7, pred_7)),
        "F1": float(
            f1_score(label_2, pred_2, average="weighted", zero_division=0)
        ),
        "MAE": mae,
        "Corr": correlation,
        "MSE": mse,
        "RMSE": rmse,
    }


def regression_metrics(predictions, labels, dataset_name: str) -> Dict[str, float]:
    normalized = dataset_name.upper().replace("-", "").replace("_", "")
    if normalized in {"CHSIMS", "SIMS"}:
        return chsims_regression_metrics(predictions, labels)
    if normalized in {"CMUMOSI", "CMUMOSEI", "MOSI", "MOSEI"}:
        return mosei_regression_metrics(predictions, labels)
    raise ValueError(f"Unsupported regression dataset: {dataset_name!r}")


def format_metrics(metrics: Dict[str, float]) -> str:
    if "Acc_3" in metrics:
        return (
            f"loss={metrics['loss']:.4f} "
            f"Acc_2={metrics['Acc_2']:.4f} "
            f"Acc_3={metrics['Acc_3']:.4f} "
            f"Acc_5={metrics['Acc_5']:.4f} "
            f"Acc_7={metrics['Acc_7']:.4f} "
            f"F1={metrics['F1']:.4f} "
            f"MAE={metrics['MAE']:.4f} "
            f"Corr={metrics['Corr']:.4f} "
            f"RMSE={metrics['RMSE']:.4f}"
        )
    return (
        f"loss={metrics['loss']:.4f} "
        f"Acc_2={metrics['Acc_2']:.4f} "
        f"Acc_2_has0={metrics['Acc_2_has0']:.4f} "
        f"Acc_5={metrics['Acc_5']:.4f} "
        f"Acc_7={metrics['Acc_7']:.4f} "
        f"F1={metrics['F1']:.4f} "
        f"F1_has0={metrics['F1_has0']:.4f} "
        f"MAE={metrics['MAE']:.4f} "
        f"Corr={metrics['Corr']:.4f} "
        f"RMSE={metrics['RMSE']:.4f}"
    )
