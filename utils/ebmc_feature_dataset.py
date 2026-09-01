from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = {
    "audio": "wav2vec-large-c-UTT",
    "text": "deberta-large-4-UTT",
    "vision": "manet_UTT",
}
CHSIMS_FEATURE_NAMES = {
    "audio": "wav2vec-large-c-UTT",
    "text": "chinese-macbert-large-4-UTT",
    "vision": "manet_UTT",
}
EXPECTED_DIMS = {"audio": 512, "text": 1024, "vision": 1024}
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _load_annotation_payload(label_path: Path) -> list[Any]:
    with label_path.open("rb") as file:
        payload = pickle.load(file, encoding="latin1")
    if not isinstance(payload, (list, tuple)) or len(payload) != 7:
        raise ValueError(
            f"Expected seven annotation objects in {label_path}, got {type(payload).__name__}."
        )
    return list(payload)


def _clean_feature(values: np.ndarray, expected_dim: int, path: Path) -> np.ndarray:
    values = np.asarray(values).squeeze()
    if values.shape != (expected_dim,):
        raise ValueError(
            f"Unexpected feature shape at {path}: {values.shape}, expected {(expected_dim,)}."
        )
    return np.nan_to_num(
        values.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_ebmc_feature_cache(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    label_file: str = "CMUMOSEI_features_raw_2way.pkl",
    dataset_name: Optional[str] = None,
    limit_per_split: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path).resolve()
    dataset_name = (dataset_name or dataset_root.name).upper()
    label_path = dataset_root / label_file
    payload = _load_annotation_payload(label_path)
    video_ids, video_labels, _, video_sentences, train_ids, valid_ids, test_ids = payload

    split_video_ids = {
        "train": sorted(train_ids),
        "valid": sorted(valid_ids),
        "test": sorted(test_ids),
    }
    records = []
    split_indices: Dict[str, list[int]] = {split: [] for split in split_video_ids}
    for split, split_videos in split_video_ids.items():
        split_count = 0
        for video_id in split_videos:
            utterance_ids = video_ids[video_id]
            labels = video_labels[video_id]
            sentences = video_sentences[video_id]
            for utterance_id, label, sentence in zip(
                utterance_ids, labels, sentences
            ):
                if limit_per_split is not None and split_count >= limit_per_split:
                    break
                record_index = len(records)
                records.append(
                    {
                        "id": str(utterance_id),
                        "video_id": str(video_id),
                        "sentence": str(sentence),
                        "label": float(label),
                        "split": split,
                    }
                )
                split_indices[split].append(record_index)
                split_count += 1
            if limit_per_split is not None and split_count >= limit_per_split:
                break

    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate utterance IDs found while building the feature cache.")

    feature_arrays = {
        modality: np.empty((len(records), EXPECTED_DIMS[modality]), dtype=np.float32)
        for modality in FEATURE_NAMES
    }
    zero_feature_counts = {modality: 0 for modality in FEATURE_NAMES}
    for record_index, record in enumerate(records):
        utterance_id = record["id"]
        for modality, feature_name in FEATURE_NAMES.items():
            feature_path = (
                dataset_root / "features" / feature_name / f"{utterance_id}.npy"
            )
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing {modality} feature: {feature_path}")
            feature = _clean_feature(
                np.load(feature_path), EXPECTED_DIMS[modality], feature_path
            )
            feature_arrays[modality][record_index] = feature
            if not np.any(feature):
                zero_feature_counts[modality] += 1
        if (record_index + 1) % 1000 == 0 or record_index + 1 == len(records):
            print(f"Loaded {record_index + 1}/{len(records)} utterances", flush=True)

    cache = {
        "version": 2,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "label_path": str(label_path),
        "feature_names": FEATURE_NAMES,
        "expected_dims": EXPECTED_DIMS,
        "limit_per_split": limit_per_split,
        "ids": [record["id"] for record in records],
        "video_ids": [record["video_id"] for record in records],
        "sentences": [record["sentence"] for record in records],
        "labels": torch.tensor(
            [record["label"] for record in records], dtype=torch.float32
        ),
        "audio": torch.from_numpy(feature_arrays["audio"]),
        "text": torch.from_numpy(feature_arrays["text"]),
        "vision": torch.from_numpy(feature_arrays["vision"]),
        "split_indices": {
            split: torch.tensor(indices, dtype=torch.long)
            for split, indices in split_indices.items()
        },
        "zero_feature_counts": zero_feature_counts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(cache, temporary_path)
    os.replace(temporary_path, output_path)

    summary = {
        "output_path": str(output_path),
        "sample_count": len(records),
        "split_counts": {
            split: len(indices) for split, indices in split_indices.items()
        },
        "zero_feature_counts": zero_feature_counts,
        "feature_shapes": {
            modality: list(values.shape)
            for modality, values in feature_arrays.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_chsims_feature_cache(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    manifest_file: str = "manifest/chsims_manifest.csv",
    limit_per_split: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = dataset_root / manifest_file
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CH-SIMS manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "id",
        "fs_id",
        "split",
        "source_video",
        "raw_text",
        "regression_label",
    }
    missing_columns = required_columns.difference(rows[0] if rows else {})
    if missing_columns:
        raise ValueError(
            f"CH-SIMS manifest is missing columns: {sorted(missing_columns)}"
        )

    records = []
    split_indices: Dict[str, list[int]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    split_counts = {split: 0 for split in split_indices}
    for row in rows:
        split = SPLIT_ALIASES.get(str(row["split"]).strip().lower())
        if split is None:
            raise ValueError(f"Unknown CH-SIMS split: {row['split']!r}")
        if limit_per_split is not None and split_counts[split] >= limit_per_split:
            continue
        record_index = len(records)
        records.append(
            {
                "id": str(row["id"]),
                "fs_id": str(row["fs_id"]),
                "video_id": str(row["source_video"]),
                "sentence": str(row["raw_text"]),
                "label": float(row["regression_label"]),
                "classification_label": float(row.get("classification_label") or 0.0),
                "split": split,
            }
        )
        split_indices[split].append(record_index)
        split_counts[split] += 1

    for key in ("id", "fs_id"):
        values = [record[key] for record in records]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate CH-SIMS {key} values found in the manifest.")

    feature_arrays = {
        modality: np.empty((len(records), EXPECTED_DIMS[modality]), dtype=np.float32)
        for modality in CHSIMS_FEATURE_NAMES
    }
    zero_feature_counts = {modality: 0 for modality in CHSIMS_FEATURE_NAMES}
    for record_index, record in enumerate(records):
        file_id = record["fs_id"]
        for modality, feature_name in CHSIMS_FEATURE_NAMES.items():
            feature_path = dataset_root / "features" / feature_name / f"{file_id}.npy"
            if not feature_path.is_file():
                raise FileNotFoundError(f"Missing CH-SIMS {modality} feature: {feature_path}")
            feature = _clean_feature(
                np.load(feature_path), EXPECTED_DIMS[modality], feature_path
            )
            feature_arrays[modality][record_index] = feature
            if not np.any(feature):
                zero_feature_counts[modality] += 1
        if (record_index + 1) % 500 == 0 or record_index + 1 == len(records):
            print(f"Loaded {record_index + 1}/{len(records)} CH-SIMS utterances", flush=True)

    labels = np.asarray([record["label"] for record in records], dtype=np.float32)
    if labels.size and (labels.min() < -1.0001 or labels.max() > 1.0001):
        raise ValueError(
            f"CH-SIMS regression labels must be in [-1,1], got "
            f"[{labels.min()}, {labels.max()}]."
        )
    cache = {
        "version": 3,
        "dataset_name": "CHSIMS",
        "dataset_root": str(dataset_root),
        "label_path": str(manifest_path),
        "feature_names": CHSIMS_FEATURE_NAMES,
        "expected_dims": EXPECTED_DIMS,
        "limit_per_split": limit_per_split,
        "ids": [record["id"] for record in records],
        "file_ids": [record["fs_id"] for record in records],
        "video_ids": [record["video_id"] for record in records],
        "sentences": [record["sentence"] for record in records],
        "classification_labels": torch.tensor(
            [record["classification_label"] for record in records],
            dtype=torch.float32,
        ),
        "labels": torch.from_numpy(labels),
        "audio": torch.from_numpy(feature_arrays["audio"]),
        "text": torch.from_numpy(feature_arrays["text"]),
        "vision": torch.from_numpy(feature_arrays["vision"]),
        "split_indices": {
            split: torch.tensor(indices, dtype=torch.long)
            for split, indices in split_indices.items()
        },
        "zero_feature_counts": zero_feature_counts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(cache, temporary_path)
    os.replace(temporary_path, output_path)
    summary = {
        "output_path": str(output_path),
        "dataset_name": "CHSIMS",
        "sample_count": len(records),
        "split_counts": split_counts,
        "zero_feature_counts": zero_feature_counts,
        "feature_names": CHSIMS_FEATURE_NAMES,
        "feature_shapes": {
            modality: list(values.shape)
            for modality, values in feature_arrays.items()
        },
        "label_range": [float(labels.min()), float(labels.max())],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def load_ebmc_feature_cache(cache_path: str | Path) -> Dict[str, Any]:
    cache_path = Path(cache_path).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {"audio", "text", "vision", "labels", "split_indices"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"Feature cache is missing keys: {sorted(missing)}")
    return cache


class EBMCCachedFeatureDataset(Dataset):
    def __init__(self, cache: Dict[str, Any], split: str) -> None:
        normalized_split = SPLIT_ALIASES.get(split.lower())
        if normalized_split is None:
            raise ValueError(f"Unknown split: {split}")
        self.cache = cache
        self.split = normalized_split
        self.indices = cache["split_indices"][normalized_split]

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cache_index = int(self.indices[index])
        return {
            "vision": self.cache["vision"][cache_index],
            "audio": self.cache["audio"][cache_index],
            "text": self.cache["text"][cache_index],
            "label": self.cache["labels"][cache_index],
            "id": self.cache.get("ids", [cache_index])[cache_index],
        }


def iter_split_sizes(cache: Dict[str, Any]) -> Iterable[tuple[str, int]]:
    for split in ("train", "valid", "test"):
        yield split, int(cache["split_indices"][split].numel())
