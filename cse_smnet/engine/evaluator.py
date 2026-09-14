"""Evaluation engine for CSE-SMNet future-frame video anomaly detection."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cse_smnet.utils.checkpoint import extract_state_dict
from cse_smnet.utils.memory import (
    MemoryUsageAccumulator,
    find_mtar_module,
    resolve_inference_memory_temperature,
)
from cse_smnet.utils.metrics import (
    build_frame_labels,
    compute_auc_eer,
    compute_psnr,
    normalize_scores,
    psnr_to_anomaly_score,
)
from cse_smnet.utils.visualization import (
    save_frame_visualization,
    save_memory_usage_plot,
    save_score_plot,
)


class CSESMNetEvaluator:
    """Evaluate one ordered-frame video with a trained CSE-SMNet checkpoint."""

    def __init__(
        self,
        *,
        model: nn.Module,
        dataset,
        device: torch.device,
        config,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.cfg = config
        self.results_dir = Path(config.results_dir)
        self.frame_vis_dir = self.results_dir / "frame_visualizations"
        self.memory_vis_dir = self.results_dir / "memory_visualizations"

    def _load_model(self) -> nn.Module:
        checkpoint_path = Path(self.cfg.best_model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(loaded)
        self.model.load_state_dict(state_dict, strict=True)
        return find_mtar_module(self.model)

    def evaluate(self) -> Dict[str, Any]:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_frame_visualizations:
            self.frame_vis_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_memory_visualizations:
            self.memory_vis_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        memory_module = self._load_model()
        inference_temperature = resolve_inference_memory_temperature(
            memory_module,
            training_checkpoint_path=self.cfg.training_checkpoint_path,
            total_epochs=self.cfg.train_epochs,
            default_start_temperature=self.cfg.memory_temp_start,
            default_end_temperature=self.cfg.memory_temp_end,
            default_mode=self.cfg.memory_temp_mode,
            restore_best_temperature=self.cfg.restore_best_memory_temperature,
            fallback_temperature=self.cfg.fallback_inference_memory_temperature,
        )
        self.model.eval()

        labels_all: Optional[np.ndarray] = None
        if self.cfg.use_ground_truth:
            labels_all = build_frame_labels(
                len(self.dataset.frames),
                self.cfg.anomaly_ranges,
                one_based=self.cfg.ground_truth_one_based,
            )

        raw_scores: List[float] = []
        psnr_values: List[float] = []
        target_indices: List[int] = []
        rows: List[Dict[str, Any]] = []
        usage_accumulator = MemoryUsageAccumulator()

        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                sequence = batch["sequence"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                prediction = self.model(sequence).clamp(0.0, 1.0)
                target_eval = target.clamp(0.0, 1.0)
                psnr_batch = compute_psnr(prediction, target_eval)
                usage_accumulator.update(memory_module.memory_diagnostics())

                for sample_index in range(sequence.size(0)):
                    psnr_value = float(psnr_batch[sample_index].item())
                    raw_score = float(psnr_to_anomaly_score(psnr_value))
                    target_index = int(batch["target_index"][sample_index])
                    last_input_index = int(batch["last_input_index"][sample_index])

                    if target_index != last_input_index + 1:
                        raise RuntimeError(
                            "Temporal alignment error: expected the target to be "
                            "exactly one frame after the final observed input."
                        )

                    raw_scores.append(raw_score)
                    psnr_values.append(psnr_value)
                    target_indices.append(target_index)

                    label: int | str = ""
                    if labels_all is not None:
                        label = int(labels_all[target_index])

                    rows.append({
                        "sample_index": len(raw_scores) - 1,
                        "last_input_frame_index_zero_based": last_input_index,
                        "last_input_frame_number_one_based": last_input_index + 1,
                        "last_input_path": batch["last_input_path"][sample_index],
                        "target_frame_index_zero_based": target_index,
                        "target_frame_number_one_based": target_index + 1,
                        "target_path": batch["target_path"][sample_index],
                        "psnr_db": psnr_value,
                        "raw_anomaly_score_neg_psnr": raw_score,
                        "label": label,
                    })

                    global_index = len(raw_scores) - 1
                    if (
                        self.cfg.save_frame_visualizations
                        and global_index % self.cfg.visualization_stride == 0
                    ):
                        last_input = sequence[sample_index, -1].cpu().numpy().transpose(1, 2, 0)
                        target_np = target_eval[sample_index].cpu().numpy().transpose(1, 2, 0)
                        pred_np = prediction[sample_index].cpu().numpy().transpose(1, 2, 0)
                        save_frame_visualization(
                            self.frame_vis_dir
                            / f"{global_index:05d}_input_{last_input_index + 1:05d}_target_pred_{target_index + 1:05d}.png",
                            last_input,
                            target_np,
                            pred_np,
                            psnr_value,
                        )

                if batch_index == 0 or (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
                    print(
                        f"[{batch_index + 1:05d}/{len(loader):05d}] "
                        f"PSNR={float(psnr_batch.mean().item()):.4f} dB"
                    )

        raw_scores_np = np.asarray(raw_scores, dtype=np.float64)
        psnr_np = np.asarray(psnr_values, dtype=np.float64)
        normalized_scores = normalize_scores(raw_scores_np)
        target_indices_np = np.asarray(target_indices, dtype=np.int64)

        np.save(self.results_dir / "raw_scores.npy", raw_scores_np)
        np.save(self.results_dir / "normalized_scores.npy", normalized_scores)
        np.save(self.results_dir / "psnr_values.npy", psnr_np)
        np.save(self.results_dir / "target_frame_indices.npy", target_indices_np)

        evaluation_labels: Optional[np.ndarray] = None
        if labels_all is not None:
            evaluation_labels = labels_all[target_indices_np]
            np.save(self.results_dir / "evaluation_labels.npy", evaluation_labels)

        for row, score in zip(rows, normalized_scores):
            row["normalized_anomaly_score"] = float(score)

        if rows:
            with (self.results_dir / "per_frame_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        metrics: Dict[str, Any] = {
            "num_test_frames": len(self.dataset.frames),
            "num_evaluation_samples": len(self.dataset),
            "sequence_length": self.cfg.seq_len,
            "mean_psnr_db": float(psnr_np.mean()),
            "std_psnr_db": float(psnr_np.std()),
            "min_psnr_db": float(psnr_np.min()),
            "max_psnr_db": float(psnr_np.max()),
            "memory_temperature": inference_temperature,
            "anomaly_score": "minmax(-PSNR(target, prediction))",
        }

        if evaluation_labels is not None:
            auc, eer, threshold = compute_auc_eer(normalized_scores, evaluation_labels)
            metrics.update({"auc": auc, "eer": eer, "eer_threshold": threshold})
            print("\nFrame-level evaluation")
            print(f"AUC : {auc:.6f} ({auc * 100:.2f}%)")
            print(f"EER : {eer:.6f} ({eer * 100:.2f}%)")
            print(f"EER threshold: {threshold:.6f}")

        usage_summary = usage_accumulator.finalize()
        metrics["memory_usage"] = usage_summary

        with (self.results_dir / "test_metrics_and_memory.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metrics, file, indent=2)

        save_score_plot(
            normalized_scores,
            evaluation_labels,
            self.results_dir / "anomaly_score_plot.png",
        )
        save_memory_usage_plot(
            usage_summary,
            self.results_dir / "test_video_memory_slot_usage.png",
        )

        if self.cfg.save_memory_visualizations:
            memory_module.save_memory_visualizations(
                output_dir=self.memory_vis_dir,
                max_heatmap_slots=self.cfg.max_memory_heatmap_slots,
                annotate_points=self.cfg.annotate_memory_pca,
                save_json=True,
            )

        print("\nEvaluation completed.")
        print(f"Mean PSNR: {metrics['mean_psnr_db']:.4f} dB")
        print(f"Results: {self.results_dir}")
        return metrics
