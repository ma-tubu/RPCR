from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the structure of a trusted MMSA-style pickle."
    )
    parser.add_argument("pickle_path", type=Path)
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=2.0,
        help="Refuse to load files larger than this threshold.",
    )
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def summarize_array(values: np.ndarray, field_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": "ndarray",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nbytes": int(values.nbytes),
    }
    if values.size and np.issubdtype(values.dtype, np.number) and (
        "label" in field_name or "length" in field_name or values.size <= 100_000
    ):
        finite_values = values[np.isfinite(values)]
        summary["non_finite_count"] = int(values.size - finite_values.size)
        if finite_values.size:
            summary["min"] = scalar(finite_values.min())
            summary["max"] = scalar(finite_values.max())
    if values.size and values.dtype == object:
        summary["sample"] = [scalar(value) for value in values.reshape(-1)[:2]]
    return summary


def summarize_value(value: Any, field_name: str) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return summarize_array(value, field_name)
    if isinstance(value, (list, tuple)):
        summary = {"type": type(value).__name__, "length": len(value)}
        if value:
            summary["sample"] = [scalar(item) for item in value[:2]]
        return summary
    if isinstance(value, dict):
        return {
            "type": "dict",
            "length": len(value),
            "keys": [str(key) for key in list(value)[:30]],
        }
    return {"type": type(value).__name__, "value": scalar(value)}


def main() -> None:
    args = parse_args()
    path = args.pickle_path.resolve()
    size_bytes = path.stat().st_size
    max_size_bytes = int(args.max_size_gb * 1024**3)
    if size_bytes > max_size_bytes:
        raise RuntimeError(
            f"Refusing to load {size_bytes / 1024**3:.2f} GiB; "
            f"limit is {args.max_size_gb:.2f} GiB."
        )

    with path.open("rb") as file:
        payload = pickle.load(file)

    report: dict[str, Any] = {
        "path": str(path),
        "size_bytes": size_bytes,
        "payload_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        report["top_level_keys"] = [str(key) for key in payload]
        report["items"] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                report["items"][str(key)] = {
                    field: summarize_value(field_value, str(field))
                    for field, field_value in value.items()
                }
            else:
                report["items"][str(key)] = summarize_value(value, str(key))
    else:
        report["summary"] = summarize_value(payload, "payload")

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
