"""Evaluation configuration for CSE-SMNet."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class EvaluationConfig:
    """Configuration for single-video future-frame anomaly evaluation."""

    test_video_dir: Path
    best_model_path: Path
    results_dir: Path = Path("outputs/evaluation")
    training_checkpoint_path: Optional[Path] = None

    seq_len: int = 3
    image_size: int = 256
    batch_size: int = 1
    num_workers: int = 0
    grayscale_dataset: bool = False

    anomaly_ranges: List[Tuple[int, int]] = field(default_factory=list)
    ground_truth_one_based: bool = True

    train_epochs: int = 100
    memory_temp_start: float = 0.30
    memory_temp_end: float = 0.10
    memory_temp_mode: str = "cosine"
    restore_best_memory_temperature: bool = True
    fallback_inference_memory_temperature: float = 0.20

    save_frame_visualizations: bool = True
    visualization_stride: int = 1
    save_memory_visualizations: bool = True
    max_memory_heatmap_slots: int = 150
    annotate_memory_pca: bool = False

    @property
    def use_ground_truth(self) -> bool:
        return bool(self.anomaly_ranges)
