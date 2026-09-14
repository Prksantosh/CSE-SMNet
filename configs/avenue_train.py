"""Training configuration for CSE-SMNet on Avenue.

Defaults reproduce the validated normal-only training script supplied with the
project. Paths may be overridden from ``train.py`` without changing the
optimization protocol.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AvenueTrainConfig:
    # Reproducibility / data protocol
    seed: int = 42
    val_ratio: float = 0.10
    seq_len: int = 3
    batch_size: int = 4
    epochs: int = 100
    num_workers: int = 0
    protocol_version: str = "normal_only_video_split_usage_regularized_v3"

    train_root: Path = Path("data/train")
    output_dir: Path = Path("outputs/avenue")
    resume: bool = True
    image_size: tuple[int, int] = (256, 256)

    # Prediction loss
    lambda_mse: float = 0.0
    lambda_ssim: float = 0.50
    lambda_temp: float = 0.30
    lambda_grad: float = 0.20

    # Structured prototype-memory regularization
    lambda_compact: float = 0.0
    lambda_separate: float = 1e-3
    lambda_diverse: float = 2e-3
    lambda_usage: float = 2e-4

    memory_temp_start: float = 0.07
    memory_temp_end: float = 0.10
    memory_temp_mode: str = "cosine"
    expected_separation_margin: float = 0.10

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    grad_clip_norm: Optional[float] = 5.0

    # MTAR memory artifacts
    save_memory_vis_on_best: bool = True
    save_memory_artifacts_every_epoch: bool = True
    save_memory_bank_pt: bool = True
    save_memory_bank_npy: bool = True
    save_normalized_memory_bank_npy: bool = True
    max_heatmap_slots: int = 150
    annotate_memory_pca: bool = False

    # Runtime
    device: str = "cuda"

    def memory_regularization_dict(self) -> dict:
        return {
            "lambda_compact": self.lambda_compact,
            "lambda_separate": self.lambda_separate,
            "lambda_diverse": self.lambda_diverse,
            "lambda_usage": self.lambda_usage,
            "temperature_start": self.memory_temp_start,
            "temperature_end": self.memory_temp_end,
            "temperature_mode": self.memory_temp_mode,
        }
