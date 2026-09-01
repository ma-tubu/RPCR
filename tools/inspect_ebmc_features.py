from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_DIRECTORIES = (
    "wav2vec-large-c-UTT",
    "deberta-large-4-UTT",
    "manet_UTT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an EBMC-style CMU-MOSI/CMU-MOSEI feature package."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--label-file",
        default="CMUMOSEI_features_raw_2way.pkl",
        help="Label pickle name relative to dataset_root.",
    )
    parser.add_argument(
        "--max-feature-files",
        type=int,
        default=128,
        help="Maximum NPY files sampled per modality; use 0 to scan all files.",
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        help="Optional JSONL/WAV/JPG dataset root to compare by utterance ID.",
    )
    return parser.parse_args()


def python_type(value: Any) -> str:
    return type(value).__name__


def inspect_labels(path: Path) -> tuple[dict[str, Any], list[Any]]:
    with path.open("rb") as file:
        payload = pickle.load(file, encoding="latin1")

    if not isinstance(payload, (list, tuple)) or len(payload) != 7:
        raise ValueError(f"Expected a seven-item sequence, got {python_type(payload)}.")

    video_ids, labels, speakers, sentences, train_ids, valid_ids, test_ids = payload
    split_ids = {
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }
    flat_labels = [float(label) for values in labels.values() for label in values]
    first_video = next(iter(video_ids))

    report = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "payload_type": python_type(payload),
        "payload_length": len(payload),
        "item_types": [python_type(item) for item in payload],
        "video_count": len(video_ids),
        "utterance_count": sum(len(values) for values in video_ids.values()),
        "split_video_counts": {
            split: len(ids) for split, ids in split_ids.items()
        },
        "split_utterance_counts": {
            split: sum(len(video_ids[video_id]) for video_id in ids)
            for split, ids in split_ids.items()
        },
        "label_range": [min(flat_labels), max(flat_labels)],
        "first_video": str(first_video),
        "first_utterance_ids": [str(value) for value in video_ids[first_video][:3]],
        "first_labels": [float(value) for value in labels[first_video][:3]],
        "first_speakers": [str(value) for value in speakers[first_video][:3]],
        "first_sentences": [str(value) for value in sentences[first_video][:2]],
    }
    return report, payload


def inspect_raw_compatibility(
    raw_data_root: Path,
    dataset_root: Path,
    label_payload: list[Any],
) -> dict[str, Any]:
    video_ids, labels, _, _, train_ids, valid_ids, test_ids = label_payload
    label_by_utterance = {
        utterance_id: float(label)
        for video_id, utterance_ids in video_ids.items()
        for utterance_id, label in zip(utterance_ids, labels[video_id])
    }
    split_by_video = {
        str(video_id): split
        for split, video_set in (
            ("train", train_ids),
            ("valid", valid_ids),
            ("test", test_ids),
        )
        for video_id in video_set
    }
    feature_ids = {
        name: {path.stem for path in (dataset_root / "features" / name).glob("*.npy")}
        for name in FEATURE_DIRECTORIES
    }

    report: dict[str, Any] = {"raw_data_root": str(raw_data_root.resolve()), "splits": {}}
    for raw_split, expected_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        rows = [
            json.loads(line)
            for line in (raw_data_root / f"{raw_split}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        utterance_ids = [Path(row["img"]).stem for row in rows]
        annotation_missing = [
            utterance_id
            for utterance_id in utterance_ids
            if utterance_id not in label_by_utterance
        ]
        split_mismatches = []
        score_differences = []
        for row, utterance_id in zip(rows, utterance_ids):
            video_id = str(row.get("meta", {}).get("vid", utterance_id.rsplit("_", 1)[0]))
            actual_split = split_by_video.get(video_id)
            if actual_split is not None and actual_split != expected_split:
                split_mismatches.append(utterance_id)
            if utterance_id in label_by_utterance and "score" in row.get("meta", {}):
                score_differences.append(
                    abs(float(row["meta"]["score"]) - label_by_utterance[utterance_id])
                )

        report["splits"][raw_split] = {
            "sample_count": len(rows),
            "unique_utterance_count": len(set(utterance_ids)),
            "annotation_match_count": len(rows) - len(annotation_missing),
            "annotation_missing_count": len(annotation_missing),
            "annotation_missing_examples": annotation_missing[:10],
            "official_split_mismatch_count": len(split_mismatches),
            "official_split_mismatch_examples": split_mismatches[:10],
            "max_score_abs_difference": max(score_differences, default=None),
            "feature_match_counts": {
                name: sum(utterance_id in ids for utterance_id in utterance_ids)
                for name, ids in feature_ids.items()
            },
        }
    return report


def inspect_feature_directory(path: Path, max_files: int) -> dict[str, Any]:
    files = sorted(path.glob("*.npy"))
    inspected_files = files if max_files == 0 else files[:max_files]
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    non_finite_files = 0

    for feature_path in inspected_files:
        values = np.load(feature_path, mmap_mode="r")
        shape_counts[str(tuple(values.shape))] += 1
        dtype_counts[str(values.dtype)] += 1
        if not np.isfinite(values).all():
            non_finite_files += 1

    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "inspected_file_count": len(inspected_files),
        "shape_counts": dict(shape_counts),
        "dtype_counts": dict(dtype_counts),
        "non_finite_file_count": non_finite_files,
        "sample_files": [feature_path.name for feature_path in files[:3]],
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    label_report, label_payload = inspect_labels(dataset_root / args.label_file)
    report = {
        "dataset_root": str(dataset_root),
        "labels": label_report,
        "features": {
            name: inspect_feature_directory(
                dataset_root / "features" / name, args.max_feature_files
            )
            for name in FEATURE_DIRECTORIES
        },
    }
    if args.raw_data_root is not None:
        report["raw_compatibility"] = inspect_raw_compatibility(
            args.raw_data_root.resolve(), dataset_root, label_payload
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
