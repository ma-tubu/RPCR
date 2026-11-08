from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.utils.data import Dataset


SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def load_feature_cache(cache_path: str | Path) -> Dict[str, Any]:
    cache_path = Path(cache_path).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {"audio", "text", "vision", "labels", "split_indices"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"Feature cache is missing keys: {sorted(missing)}")
    return cache


class CachedFeatureDataset(Dataset):
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
            "id": self.cache["ids"][cache_index] if "ids" in self.cache else str(cache_index),
        }


def iter_split_sizes(cache: Dict[str, Any]) -> Iterable[tuple[str, int]]:
    for split in ("train", "valid", "test"):
        yield split, int(cache["split_indices"][split].numel())
