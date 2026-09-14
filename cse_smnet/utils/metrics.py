"""Metrics and anomaly-score helpers for CSE-SMNet."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return per-sample PSNR for tensors scaled to [0, 1]."""
    mse = F.mse_loss(pred, target, reduction="none").mean((1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def psnr_to_anomaly_score(psnr: np.ndarray | float) -> np.ndarray:
    """Convert prediction quality to anomaly evidence: lower PSNR -> higher score."""
    return -np.asarray(psnr, dtype=np.float64)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize one video's anomaly scores to [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.copy()
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def build_frame_labels(
    num_frames: int,
    anomaly_ranges: Iterable[Tuple[int, int]],
    one_based: bool = True,
) -> np.ndarray:
    """Build binary frame labels from inclusive anomalous frame ranges."""
    labels = np.zeros(int(num_frames), dtype=np.int32)
    for start, end in anomaly_ranges:
        if one_based:
            start -= 1
            end -= 1
        start = max(0, int(start))
        end = min(num_frames - 1, int(end))
        if end >= start:
            labels[start : end + 1] = 1
    return labels


def compute_auc_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float]:
    """Return ROC-AUC, approximate EER, and the EER operating threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if len(np.unique(labels)) < 2:
        raise ValueError("AUC/EER require both normal and anomalous labels.")

    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[index] + fnr[index])
    return float(auc), float(eer), float(thresholds[index])
