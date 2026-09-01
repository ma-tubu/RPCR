from __future__ import annotations

import argparse
import csv
import json
import pickle
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an utterance-level CH-SIMS manifest from labels and Raw.zip."
    )
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=Path("datasets/CH-SIMS/unaligned_39.pkl"),
        help="MMSA-style CH-SIMS pickle with official splits, ids, text, and labels.",
    )
    parser.add_argument(
        "--raw-zip",
        type=Path,
        default=Path("datasets/CH-SIMS/Raw.zip"),
        help="CH-SIMS Raw.zip containing utterance mp4 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/CHSIMS"),
        help="Output dataset root for GCNet/EBMC-style utterance features.",
    )
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    array = np.asarray(value)
    if array.shape == ():
        return scalar(array.item())
    if array.size == 1:
        return scalar(array.reshape(-1)[0])
    return [scalar(item) for item in array.reshape(-1).tolist()]


def clean_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def split_chsims_id(sample_id: str) -> tuple[str, str, str]:
    if "$_$" not in sample_id:
        raise ValueError(f"Unexpected CH-SIMS id format: {sample_id}")
    source_video, clip_id = sample_id.split("$_$", 1)
    if not source_video or not clip_id:
        raise ValueError(f"Unexpected CH-SIMS id format: {sample_id}")
    return source_video, clip_id, f"{source_video}_{clip_id}"


def build_zip_index(raw_zip: Path) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(raw_zip) as archive:
        for member in archive.namelist():
            path = PurePosixPath(member)
            if path.suffix.lower() != ".mp4":
                continue
            if path.name.startswith("._"):
                continue
            if len(path.parts) < 2:
                continue
            key = (path.parts[-2], path.stem)
            if key in index:
                raise ValueError(
                    f"Duplicate video member for {key}: {index[key]} and {member}"
                )
            index[key] = member
    return index


def label_at(split_data: dict[str, Any], key: str, index: int) -> Any:
    if key not in split_data:
        return ""
    values = split_data[key]
    return scalar(values[index])


def build_rows(payload: dict[str, Any], zip_index: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for split in EXPECTED_SPLITS:
        if split not in payload:
            raise KeyError(f"Missing CH-SIMS split: {split}")
        split_data = payload[split]
        ids = [clean_text(value) for value in split_data["id"]]
        raw_text = [clean_text(value) for value in split_data.get("raw_text", [""] * len(ids))]
        for index, sample_id in enumerate(ids):
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id: {sample_id}")
            source_video, clip_id, fs_id = split_chsims_id(sample_id)
            zip_member = zip_index.get((source_video, clip_id))
            if zip_member is None:
                raise FileNotFoundError(f"Raw video not found in zip for {sample_id}")
            seen_ids.add(sample_id)
            rows.append(
                {
                    "id": sample_id,
                    "fs_id": fs_id,
                    "split": split,
                    "source_video": source_video,
                    "clip_id": clip_id,
                    "raw_zip_member": zip_member,
                    "raw_text": raw_text[index],
                    "regression_label": label_at(split_data, "regression_labels", index),
                    "classification_label": label_at(split_data, "classification_labels", index),
                    "text_label": label_at(split_data, "text_labels", index),
                    "audio_label": label_at(split_data, "audio_labels", index),
                    "vision_label": label_at(split_data, "vision_labels", index),
                }
            )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_root: Path) -> None:
    manifest_root = output_root / "manifest"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "chsims_manifest.csv"
    mapping_path = manifest_root / "id_mapping.json"
    summary_path = manifest_root / "summary.json"

    fields = [
        "id",
        "fs_id",
        "split",
        "source_video",
        "clip_id",
        "raw_zip_member",
        "raw_text",
        "regression_label",
        "classification_label",
        "text_label",
        "audio_label",
        "vision_label",
    ]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    mapping = {
        row["id"]: {
            "fs_id": row["fs_id"],
            "split": row["split"],
            "raw_zip_member": row["raw_zip_member"],
        }
        for row in rows
    }
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in EXPECTED_SPLITS
    }
    source_videos = sorted({row["source_video"] for row in rows})
    summary = {
        "sample_count": len(rows),
        "split_counts": split_counts,
        "source_video_count": len(source_videos),
        "source_videos": source_videos,
        "manifest": str(manifest_path.resolve()),
        "id_mapping": str(mapping_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    pickle_path = args.pickle_path.resolve()
    raw_zip = args.raw_zip.resolve()
    output_root = args.output_root.resolve()
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")
    if not raw_zip.is_file():
        raise FileNotFoundError(f"Raw zip not found: {raw_zip}")

    with pickle_path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload).__name__}")

    zip_index = build_zip_index(raw_zip)
    rows = build_rows(payload, zip_index)
    write_outputs(rows, output_root)

    split_counts = {
        split: sum(row["split"] == split for row in rows) for split in EXPECTED_SPLITS
    }
    print(
        json.dumps(
            {
                "sample_count": len(rows),
                "split_counts": split_counts,
                "source_video_count": len({row["source_video"] for row in rows}),
                "output_root": str(output_root),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
