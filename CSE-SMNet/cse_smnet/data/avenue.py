# -*- coding: utf-8 -*-
"""
Avenue normal-training dataset with explicit source-video metadata.

Expected directory structure
----------------------------
normal/
├── O1/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── O3/
├── O4/
└── O5/

Each sample contains `seq_len` consecutive input frames and the immediately
following target frame. `get_video_id(index)` enables reviewer-compliant
video-level train/validation partitioning.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


def natural_sort_key(path: str):
    """Sort frame names numerically when filenames contain frame numbers."""
    name = os.path.basename(path)
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ]


class AvenueDataset(Dataset):
    """
    Dataset for normal future-frame prediction.

    Returns
    -------
    frames:
        Tensor with shape (T, C, H, W).
    target:
        Tensor with shape (C, H, W).
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 3,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.seq_len = int(seq_len)
        self.transform = transform

        if self.seq_len < 1:
            raise ValueError("seq_len must be at least 1.")

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )

        # Keep sample data and video IDs in separate aligned lists so the
        # public __getitem__ output remains exactly (frames, target).
        self.samples: List[Tuple[List[str], str]] = []
        self.video_ids: List[str] = []

        video_folders = sorted(
            entry
            for entry in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, entry))
        )

        if not video_folders:
            raise RuntimeError(
                f"No video subdirectories were found in {self.root_dir}. "
                "Set root_dir to the parent directory containing O1, O2, ..."
            )

        valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp","*.tif", "*.tiff")

        for video_name in video_folders:
            video_dir = os.path.join(self.root_dir, video_name)

            frame_paths: List[str] = []
            for extension in valid_extensions:
                frame_paths.extend(
                    glob.glob(os.path.join(video_dir, extension))
                )
                frame_paths.extend(
                    glob.glob(os.path.join(video_dir, extension.upper()))
                )

            frame_paths = sorted(
                set(frame_paths),
                key=natural_sort_key,
            )

            required_frames = self.seq_len + 1
            if len(frame_paths) < required_frames:
                print(
                    f"Warning: skipping {video_name}; found "
                    f"{len(frame_paths)} frames but at least "
                    f"{required_frames} are required."
                )
                continue

            # Sliding-window clips remain wholly associated with video_name.
            number_of_samples = len(frame_paths) - self.seq_len

            for start in range(number_of_samples):
                input_paths = frame_paths[
                    start : start + self.seq_len
                ]
                target_path = frame_paths[
                    start + self.seq_len
                ]

                self.samples.append((input_paths, target_path))
                self.video_ids.append(video_name)

        if not self.samples:
            raise RuntimeError(
                f"No valid frame sequences were created from {self.root_dir}."
            )

        if len(self.samples) != len(self.video_ids):
            raise RuntimeError(
                "Internal metadata error: samples and video_ids differ "
                "in length."
            )

        counts = {}
        for video_id in self.video_ids:
            counts[video_id] = counts.get(video_id, 0) + 1

        print(
            f"AvenueDataset: {len(self.samples)} clips from "
            f"{len(counts)} videos: {counts}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_video_id(self, index: int) -> str:
        """
        Return the source video of a clip.

        The training script calls this method to ensure all clips from one
        video remain entirely in either the training or validation subset.
        """
        return self.video_ids[index]

    def __getitem__(self, index: int):
        input_paths, target_path = self.samples[index]

        input_frames = []
        for frame_path in input_paths:
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = torch.from_numpy(
                        __import__("numpy").array(image)
                    ).permute(2, 0, 1).float() / 255.0
                input_frames.append(image)

        with Image.open(target_path) as target_image:
            target_image = target_image.convert("RGB")
            if self.transform is not None:
                target = self.transform(target_image)
            else:
                target = torch.from_numpy(
                    __import__("numpy").array(target_image)
                ).permute(2, 0, 1).float() / 255.0

        frames = torch.stack(input_frames, dim=0)
        return frames, target