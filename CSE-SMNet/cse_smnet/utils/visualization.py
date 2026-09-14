"""Evaluation visualizations for CSE-SMNet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def last_input_prediction_difference_map(
    last_input: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Normalized |I_t - I_hat_(t+1)| map used for visualization only."""
    difference = np.abs(last_input - prediction).mean(axis=2)
    low, high = float(difference.min()), float(difference.max())
    if high - low < 1e-12:
        return np.zeros_like(difference, dtype=np.float32)
    return ((difference - low) / (high - low)).astype(np.float32)


def save_frame_visualization(
    save_path: Path,
    last_input: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    psnr: float,
) -> None:
    """Save observed/target/prediction panels and the visualization-only difference map.

    PSNR and anomaly scoring use target(t+1) vs prediction(t+1). The difference,
    heatmap, and overlay panels intentionally use last-input(t) vs prediction(t+1),
    matching the validated experiment script.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    last_u8 = to_uint8(last_input)
    target_u8 = to_uint8(target)
    pred_u8 = to_uint8(prediction)

    difference = last_input_prediction_difference_map(last_input, prediction)
    diff_u8 = (difference * 255.0).astype(np.uint8)
    diff_rgb = cv2.cvtColor(diff_u8, cv2.COLOR_GRAY2RGB)
    heat_bgr = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(last_u8, 0.65, heat_rgb, 0.35, 0)

    items = [
        (last_u8, "Last Input Frame (t)"),
        (target_u8, "Target Frame (t+1)"),
        (pred_u8, "Predicted Frame (t+1)"),
        (diff_rgb, "Difference Map (Pred vs Last Input)"),
        (heat_rgb, "Heatmap"),
        (overlay, "Overlay"),
    ]

    title_height = 36
    panels = []
    for image_rgb, title in items:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        panel = np.full(
            (title_height + image_bgr.shape[0], image_bgr.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        panel[title_height:] = image_bgr
        cv2.putText(
            panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.50, (0, 0, 0), 1, cv2.LINE_AA,
        )
        panels.append(panel)

    canvas = np.concatenate(panels, axis=1)
    strip = np.full((40, canvas.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        strip, f"PSNR={psnr:.3f} dB", (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 2, cv2.LINE_AA,
    )
    canvas = np.concatenate([canvas, strip], axis=0)
    cv2.imwrite(str(save_path), canvas)


def save_score_plot(
    scores: np.ndarray,
    labels: Optional[np.ndarray],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(scores))
    plt.figure(figsize=(13, 4.5))
    plt.plot(x, scores, linewidth=1.5, label="Anomaly score over time")
    if labels is not None and len(labels) == len(scores):
        plt.fill_between(
            x, 0, 1, where=(labels == 1), alpha=0.15,
            label="Ground-truth anomaly",
        )
    plt.xlabel("Evaluation sample / target-frame order")
    plt.ylabel("Normalized anomaly score")
    plt.title("Frame-level anomaly score")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_memory_usage_plot(summary: Dict[str, Any], save_path: Path) -> None:
    direct = summary.get("direct", {}).get("usage")
    temporal = summary.get("temporal", {}).get("usage")
    if direct is None and temporal is None:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    if direct is not None:
        plt.plot(np.arange(len(direct)), direct, linewidth=1.5, label="Direct retrieval")
    if temporal is not None:
        plt.plot(np.arange(len(temporal)), temporal, linewidth=1.5, label="Temporal retrieval")
    plt.xlabel("Memory slot index")
    plt.ylabel("Mean attention probability")
    plt.title("Whole-test-video MTAR memory-slot usage")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
