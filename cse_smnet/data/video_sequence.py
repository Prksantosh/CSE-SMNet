"""Generic ordered-frame video dataset used for CSE-SMNet evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoSequenceDataset(Dataset):
    """Create future-frame samples from one directory of ordered video frames.

    A sample contains ``seq_len`` observed frames followed by the immediately
    succeeding target frame. Returned frame indices are zero based.
    """

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        root_dir: Path | str,
        seq_len: int = 3,
        image_size: int = 256,
        grayscale_dataset: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.grayscale_dataset = bool(grayscale_dataset)

        if self.seq_len < 1:
            raise ValueError("seq_len must be at least 1.")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Test path is not a directory: {self.root_dir}")

        self.frames = sorted(
            path
            for path in self.root_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS
        )
        if len(self.frames) <= self.seq_len:
            raise ValueError(
                f"Found {len(self.frames)} frames; need more than {self.seq_len}."
            )

    def _read_frame(self, path: Path) -> np.ndarray:
        if self.grayscale_dataset:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read image: {path}")
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            image = np.stack([image, image, image], axis=-1)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0

        return np.transpose(image, (2, 0, 1))

    def __len__(self) -> int:
        return len(self.frames) - self.seq_len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sequence = np.stack(
            [self._read_frame(self.frames[index + offset]) for offset in range(self.seq_len)],
            axis=0,
        )

        last_input_index = index + self.seq_len - 1
        target_index = last_input_index + 1
        target = self._read_frame(self.frames[target_index])

        return {
            "sequence": torch.from_numpy(sequence).float(),
            "target": torch.from_numpy(target).float(),
            "last_input_path": str(self.frames[last_input_index]),
            "last_input_index": last_input_index,
            "target_path": str(self.frames[target_index]),
            "target_index": target_index,
        }
