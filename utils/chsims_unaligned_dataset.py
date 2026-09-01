from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_SHAPES = {
    "text": (39, 768),
    "text_bert": (3, 39),
    "audio": (400, 33),
    "vision": (55, 709),
}


def _clean_float32(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_standardization(
    features: np.ndarray,
    clip_percentiles: tuple[float, float],
) -> Dict[str, np.ndarray]:
    low_percentile, high_percentile = clip_percentiles
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100.")
    low, high = np.percentile(
        features,
        [low_percentile, high_percentile],
        axis=(0, 1),
    )
    clipped = np.clip(features, low[None, None, :], high[None, None, :])
    mean = clipped.mean(axis=(0, 1))
    std = np.maximum(clipped.std(axis=(0, 1)), 1e-6)
    return {
        "low": low.astype(np.float32),
        "high": high.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def _apply_standardization(
    features: np.ndarray,
    stats: Mapping[str, np.ndarray],
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    features = np.clip(
        features,
        stats["low"][None, None, :],
        stats["high"][None, None, :],
    )
    features = (features - stats["mean"][None, None, :]) / stats["std"][
        None, None, :
    ]
    return np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)


class CHSIMSUnalignedDataset(Dataset):
    def __init__(self, split: str, split_data: Mapping[str, Any]) -> None:
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"Unsupported CH-SIMS split: {split}")
        self.split = split
        self.text = _clean_float32(split_data["text"])
        self.audio = _clean_float32(split_data["audio"])
        self.vision = _clean_float32(split_data["vision"])
        self.text_bert = np.asarray(split_data["text_bert"], dtype=np.int64)
        self.labels = _clean_float32(split_data["regression_labels"]).reshape(-1)
        self.audio_lengths = np.asarray(split_data["audio_lengths"], dtype=np.int64)
        self.vision_lengths = np.asarray(split_data["vision_lengths"], dtype=np.int64)
        self.text_lengths = self.text_bert[:, 1, :].sum(axis=1).astype(np.int64)
        self.ids = [str(value) for value in split_data.get("id", range(len(self.labels)))]
        self.raw_text = [
            str(value) for value in split_data.get("raw_text", [""] * len(self.labels))
        ]
        self._validate()

    def _validate(self) -> None:
        count = len(self.labels)
        lengths = {
            "text": len(self.text),
            "audio": len(self.audio),
            "vision": len(self.vision),
            "text_bert": len(self.text_bert),
            "audio_lengths": len(self.audio_lengths),
            "vision_lengths": len(self.vision_lengths),
            "text_lengths": len(self.text_lengths),
            "ids": len(self.ids),
            "raw_text": len(self.raw_text),
        }
        if any(value != count for value in lengths.values()):
            raise ValueError(f"Inconsistent CH-SIMS {self.split} lengths: {lengths}")
        actual_shapes = {
            "text": tuple(self.text.shape[1:]),
            "text_bert": tuple(self.text_bert.shape[1:]),
            "audio": tuple(self.audio.shape[1:]),
            "vision": tuple(self.vision.shape[1:]),
        }
        mismatches = {
            name: {"expected": EXPECTED_SHAPES[name], "actual": actual_shapes[name]}
            for name in EXPECTED_SHAPES
            if actual_shapes[name] != EXPECTED_SHAPES[name]
        }
        if mismatches:
            raise ValueError(f"Unexpected CH-SIMS feature shapes: {mismatches}")
        if np.min(self.labels) < -1.0001 or np.max(self.labels) > 1.0001:
            raise ValueError(
                f"CH-SIMS labels must be in [-1, 1], got "
                f"[{self.labels.min()}, {self.labels.max()}]."
            )
        length_specs = (
            ("text", self.text_lengths, self.text.shape[1]),
            ("audio", self.audio_lengths, self.audio.shape[1]),
            ("vision", self.vision_lengths, self.vision.shape[1]),
        )
        for name, values, maximum in length_specs:
            if np.min(values) < 1 or np.max(values) > maximum:
                raise ValueError(
                    f"Invalid {name} lengths in {self.split}: "
                    f"[{values.min()}, {values.max()}], maximum={maximum}."
                )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "text_length": torch.tensor(self.text_lengths[index], dtype=torch.long),
            "audio_length": torch.tensor(self.audio_lengths[index], dtype=torch.long),
            "vision_length": torch.tensor(self.vision_lengths[index], dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "id": self.ids[index],
            "raw_text": self.raw_text[index],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "samples": len(self),
            "text_shape": list(self.text.shape),
            "audio_shape": list(self.audio.shape),
            "vision_shape": list(self.vision.shape),
            "label_range": [float(self.labels.min()), float(self.labels.max())],
            "length_ranges": {
                "text": [int(self.text_lengths.min()), int(self.text_lengths.max())],
                "audio": [int(self.audio_lengths.min()), int(self.audio_lengths.max())],
                "vision": [int(self.vision_lengths.min()), int(self.vision_lengths.max())],
            },
        }


def load_chsims_unaligned_datasets(
    data_path: str | Path,
    *,
    standardize: bool = True,
    clip_percentiles: tuple[float, float] = (0.1, 99.9),
) -> Dict[str, CHSIMSUnalignedDataset]:
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"CH-SIMS unaligned feature file not found: {data_path}")
    with data_path.open("rb") as handle:
        payload = pickle.load(handle)
    missing_splits = set(EXPECTED_SPLITS).difference(payload)
    if missing_splits:
        raise KeyError(f"CH-SIMS pickle is missing splits: {sorted(missing_splits)}")

    if standardize:
        stats = {}
        for modality in ("audio", "vision"):
            train_features = _clean_float32(payload["train"][modality])
            stats[modality] = _fit_standardization(
                train_features, clip_percentiles
            )
            for split in EXPECTED_SPLITS:
                payload[split][modality] = _apply_standardization(
                    payload[split][modality], stats[modality]
                )

    datasets = {
        split: CHSIMSUnalignedDataset(split, payload[split])
        for split in EXPECTED_SPLITS
    }
    return datasets
