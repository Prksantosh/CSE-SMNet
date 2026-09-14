"""Video-level normal-only train/validation split utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset, Subset


def _first_path(value: Any) -> Optional[str]:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, dict):
        preferred = (
            "video_id", "video", "video_path", "frame_paths", "frames",
            "paths", "path", "target_path",
        )
        for key in preferred:
            if key in value:
                result = _first_path(value[key])
                if result is not None:
                    return result
        for item in value.values():
            result = _first_path(item)
            if result is not None:
                return result
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _first_path(item)
            if result is not None:
                return result
    return None


def _video_id_from_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    video_path = path if path.suffix == "" else path.parent
    return os.path.normcase(os.path.normpath(str(video_path)))


def infer_video_id(dataset: Dataset, index: int) -> str:
    if hasattr(dataset, "get_video_id") and callable(dataset.get_video_id):
        return str(dataset.get_video_id(index))

    if hasattr(dataset, "video_ids"):
        values = getattr(dataset, "video_ids")
        if len(values) == len(dataset):
            return str(values[index])

    for name in (
        "samples", "sequences", "clips", "frame_sequences", "all_sequences",
        "sample_paths", "frame_paths", "data", "items",
    ):
        if not hasattr(dataset, name):
            continue
        metadata = getattr(dataset, name)
        try:
            if len(metadata) != len(dataset):
                continue
            source_path = _first_path(metadata[index])
        except (TypeError, IndexError, KeyError):
            continue
        if source_path is not None:
            return _video_id_from_path(source_path)

    raise RuntimeError(
        "Could not infer source-video IDs. The dataset should expose "
        "get_video_id(index) so clips from one video stay in one split."
    )


def make_video_level_split(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset, List[str], List[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must lie between 0 and 1.")

    video_to_indices: Dict[str, List[int]] = {}
    for index in range(len(dataset)):
        video_id = infer_video_id(dataset, index)
        video_to_indices.setdefault(video_id, []).append(index)

    video_ids = sorted(video_to_indices)
    if len(video_ids) < 2:
        raise RuntimeError("At least two normal videos are required.")

    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n_val_videos = max(1, round(len(video_ids) * val_ratio))
    n_val_videos = min(n_val_videos, len(video_ids) - 1)

    val_video_ids = sorted(video_ids[:n_val_videos])
    train_video_ids = sorted(video_ids[n_val_videos:])

    train_indices = [i for vid in train_video_ids for i in video_to_indices[vid]]
    val_indices = [i for vid in val_video_ids for i in video_to_indices[vid]]

    if set(train_indices) & set(val_indices):
        raise RuntimeError("Training and validation indices overlap.")
    if not train_indices or not val_indices:
        raise RuntimeError("An empty training/validation subset was created.")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        train_video_ids,
        val_video_ids,
    )


def save_split_manifest(
    path: Path,
    *,
    protocol_version: str,
    seed: int,
    val_ratio: float,
    train_video_ids: Sequence[str],
    val_video_ids: Sequence[str],
    train_clips: int,
    val_clips: int,
    memory_regularization: dict,
) -> None:
    manifest = {
        "protocol_version": protocol_version,
        "seed": seed,
        "validation_ratio": val_ratio,
        "split_level": "video",
        "uses_anomaly_labels": False,
        "training_videos": list(train_video_ids),
        "validation_videos": list(val_video_ids),
        "training_clips": int(train_clips),
        "validation_clips": int(val_clips),
        "memory_regularization": memory_regularization,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
