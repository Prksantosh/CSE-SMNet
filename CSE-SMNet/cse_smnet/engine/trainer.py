"""Training engine for CSE-SMNet future-frame prediction."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from cse_smnet.utils.memory import (
    get_memory_losses,
    refresh_memory_statistics,
    safe_memory_diagnostics,
    save_memory_bank_snapshot,
    save_memory_diagnostics_json,
    save_memory_visualizations,
)
from cse_smnet.utils.metrics import compute_psnr
from cse_smnet.utils.reproducibility import capture_rng_state, restore_rng_state


HISTORY_FIELDS = [
    "epoch", "lr",
    "train_total_loss", "train_prediction_loss", "train_mse_loss",
    "train_ssim_loss", "train_temp_loss", "train_grad_loss",
    "train_memory_loss", "train_memory_compactness", "train_memory_separation",
    "train_memory_diversity", "train_memory_usage",
    "val_total_loss", "val_prediction_loss", "val_mse_loss",
    "val_ssim_loss", "val_temp_loss", "val_grad_loss",
    "val_memory_loss", "val_memory_compactness", "val_memory_separation",
    "val_memory_diversity", "val_memory_usage", "val_mean_psnr",
]


def unpack_normal_batch(batch):
    if len(batch) < 2:
        raise ValueError("Expected at least (frames, target).")
    return batch[0], batch[1]


class CSESMNetTrainer:
    """Normal-only training engine with MTAR memory regularization."""

    def __init__(
        self,
        *,
        model: nn.Module,
        memory_module: nn.Module,
        criterion,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config,
        train_video_ids: Sequence[str],
        val_video_ids: Sequence[str],
    ) -> None:
        self.model = model
        self.memory_module = memory_module
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = config
        self.train_video_ids = list(train_video_ids)
        self.val_video_ids = list(val_video_ids)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.output_dir / "best_model.pth"
        self.last_model_path = self.output_dir / "last_model.pth"
        self.checkpoint_path = self.output_dir / "checkpoint.pth"
        self.history_path = self.output_dir / "training_history.csv"
        self.latest_memory_diag_path = self.output_dir / "memory_diagnostics_latest.json"
        self.memory_epochs_dir = self.output_dir / "memory_artifacts_per_epoch"
        self.best_memory_vis_dir = self.output_dir / "memory_visualizations_best"
        self.final_memory_vis_dir = self.output_dir / "memory_visualizations_final"

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _memory_losses(self) -> Dict[str, torch.Tensor]:
        return get_memory_losses(
            self.memory_module,
            lambda_compact=self.cfg.lambda_compact,
            lambda_separate=self.cfg.lambda_separate,
            lambda_diverse=self.cfg.lambda_diverse,
            lambda_usage=self.cfg.lambda_usage,
        )

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        totals = {
            "total_loss": 0.0,
            "prediction_loss": 0.0,
            "mse_loss": 0.0,
            "ssim_loss": 0.0,
            "temp_loss": 0.0,
            "grad_loss": 0.0,
            "memory_loss": 0.0,
            "memory_compactness": 0.0,
            "memory_separation": 0.0,
            "memory_diversity": 0.0,
            "memory_usage": 0.0,
        }
        psnr_sum = 0.0
        sample_count = 0
        batch_count = 0

        with torch.inference_mode():
            for batch in self.val_loader:
                frames, target = unpack_normal_batch(batch)
                frames = frames.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = self.model(frames)
                prev_frame = frames[:, -1]
                pred_losses = self.criterion(pred, target, prev_frame)
                mem_losses = self._memory_losses()

                prediction_loss = pred_losses["total_loss"]
                total_loss = prediction_loss + mem_losses["total"]

                totals["total_loss"] += float(total_loss.item())
                totals["prediction_loss"] += float(prediction_loss.item())
                totals["mse_loss"] += float(pred_losses["mse_loss"].item())
                totals["ssim_loss"] += float(pred_losses["ssim_loss"].item())
                totals["temp_loss"] += float(pred_losses["temp_loss"].item())
                totals["grad_loss"] += float(pred_losses["grad_loss"].item())
                totals["memory_loss"] += float(mem_losses["total"].item())
                totals["memory_compactness"] += float(mem_losses["compactness"].item())
                totals["memory_separation"] += float(mem_losses["separation"].item())
                totals["memory_diversity"] += float(mem_losses["diversity"].item())
                totals["memory_usage"] += float(mem_losses["usage"].item())

                psnr_sum += float(compute_psnr(pred, target).sum().item())
                sample_count += int(target.size(0))
                batch_count += 1

        if batch_count == 0:
            raise RuntimeError("The validation loader is empty.")

        metrics = {key: value / batch_count for key, value in totals.items()}
        metrics["mean_psnr"] = psnr_sum / sample_count
        return metrics

    def _append_history(self, row: Dict[str, Any]) -> None:
        exists = self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})

    def resume_if_available(self) -> None:
        if not self.cfg.resume or not self.checkpoint_path.exists():
            return

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("protocol_version") != self.cfg.protocol_version:
            raise RuntimeError(
                "Checkpoint uses a different training protocol.\n"
                f"Checkpoint protocol: {checkpoint.get('protocol_version')}\n"
                f"Current protocol   : {self.cfg.protocol_version}"
            )
        if checkpoint.get("seed") != self.cfg.seed:
            raise RuntimeError("Checkpoint and current seeds differ.")

        saved_memory_cfg = checkpoint.get("memory_regularization", {})
        current_memory_cfg = self.cfg.memory_regularization_dict()
        if saved_memory_cfg and saved_memory_cfg != current_memory_cfg:
            raise RuntimeError(
                "Structured-memory hyperparameters differ from checkpoint.\n"
                f"Checkpoint: {saved_memory_cfg}\nCurrent: {current_memory_cfg}"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_val_loss = float(checkpoint["best_val_loss"])
        self.best_epoch = int(checkpoint.get("best_epoch", -1))
        restore_rng_state(checkpoint.get("rng_state"))

        print(
            f"Resumed at epoch {self.start_epoch + 1}; "
            f"best normal validation total loss={self.best_val_loss:.6f}."
        )

    def _train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals = {
            "total_loss": 0.0,
            "prediction_loss": 0.0,
            "mse_loss": 0.0,
            "ssim_loss": 0.0,
            "temp_loss": 0.0,
            "grad_loss": 0.0,
            "memory_loss": 0.0,
            "memory_compactness": 0.0,
            "memory_separation": 0.0,
            "memory_diversity": 0.0,
            "memory_usage": 0.0,
        }
        train_batches = 0

        for batch in self.train_loader:
            frames, target = unpack_normal_batch(batch)
            frames = frames.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(frames)
            prev_frame = frames[:, -1]
            pred_losses = self.criterion(pred, target, prev_frame)
            memory_losses = self._memory_losses()

            prediction_loss = pred_losses["total_loss"]
            total_loss = prediction_loss + memory_losses["total"]
            total_loss.backward()

            if self.cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.grad_clip_norm,
                )
            self.optimizer.step()

            totals["total_loss"] += float(total_loss.detach().item())
            totals["prediction_loss"] += float(prediction_loss.detach().item())
            totals["mse_loss"] += float(pred_losses["mse_loss"].detach().item())
            totals["ssim_loss"] += float(pred_losses["ssim_loss"].detach().item())
            totals["temp_loss"] += float(pred_losses["temp_loss"].detach().item())
            totals["grad_loss"] += float(pred_losses["grad_loss"].detach().item())
            totals["memory_loss"] += float(memory_losses["total"].detach().item())
            totals["memory_compactness"] += float(memory_losses["compactness"].detach().item())
            totals["memory_separation"] += float(memory_losses["separation"].detach().item())
            totals["memory_diversity"] += float(memory_losses["diversity"].detach().item())
            totals["memory_usage"] += float(memory_losses["usage"].detach().item())
            train_batches += 1

        if train_batches == 0:
            raise RuntimeError("The training loader is empty.")
        return {key: value / train_batches for key, value in totals.items()}

    def _save_epoch_memory_artifacts(
        self,
        epoch: int,
        current_memory_temperature: float,
    ) -> None:
        if not self.cfg.save_memory_artifacts_every_epoch:
            return

        epoch_memory_dir = self.memory_epochs_dir / f"epoch_{epoch + 1:03d}"
        refresh_memory_statistics(
            self.model,
            self.val_loader,
            self.device,
            max_batches=1,
        )
        save_memory_bank_snapshot(
            self.memory_module,
            epoch_memory_dir,
            epoch_number=epoch + 1,
            memory_temperature=current_memory_temperature,
            save_pt=self.cfg.save_memory_bank_pt,
            save_npy=self.cfg.save_memory_bank_npy,
            save_normalized_npy=self.cfg.save_normalized_memory_bank_npy,
        )
        save_memory_diagnostics_json(
            self.memory_module,
            epoch_memory_dir / "memory_diagnostics.json",
        )
        save_memory_visualizations(
            self.memory_module,
            epoch_memory_dir,
            max_heatmap_slots=self.cfg.max_heatmap_slots,
            annotate_points=self.cfg.annotate_memory_pca,
        )

    @staticmethod
    def _print_epoch(
        epoch: int,
        total_epochs: int,
        lr: float,
        memory_temperature: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        print("\n" + "=" * 110)
        print(
            f"Epoch [{epoch + 1}/{total_epochs}] | LR: {lr:.6e} | "
            f"MemTemp: {memory_temperature:.4f}"
        )
        print("-" * 110)
        print(
            "TRAIN | "
            f"Total: {train_metrics['total_loss']:.6f} | "
            f"Pred: {train_metrics['prediction_loss']:.6f} | "
            f"MSE: {train_metrics['mse_loss']:.6f} | "
            f"SSIM: {train_metrics['ssim_loss']:.6f} | "
            f"Temp: {train_metrics['temp_loss']:.6f} | "
            f"Grad: {train_metrics['grad_loss']:.6f}"
        )
        print(
            "      | "
            f"Mem: {train_metrics['memory_loss']:.6f} | "
            f"Compact: {train_metrics['memory_compactness']:.6f} | "
            f"Separate: {train_metrics['memory_separation']:.6f} | "
            f"Diverse: {train_metrics['memory_diversity']:.6f} | "
            f"Usage: {train_metrics['memory_usage']:.6f}"
        )
        print("-" * 110)
        print(
            "VAL   | "
            f"Total: {val_metrics['total_loss']:.6f} | "
            f"Pred: {val_metrics['prediction_loss']:.6f} | "
            f"MSE: {val_metrics['mse_loss']:.6f} | "
            f"SSIM: {val_metrics['ssim_loss']:.6f} | "
            f"Temp: {val_metrics['temp_loss']:.6f} | "
            f"Grad: {val_metrics['grad_loss']:.6f} | "
            f"PSNR: {val_metrics['mean_psnr']:.4f} dB"
        )
        print(
            "      | "
            f"Mem: {val_metrics['memory_loss']:.6f} | "
            f"Compact: {val_metrics['memory_compactness']:.6f} | "
            f"Separate: {val_metrics['memory_separation']:.6f} | "
            f"Diverse: {val_metrics['memory_diversity']:.6f} | "
            f"Usage: {val_metrics['memory_usage']:.6f}"
        )
        print("=" * 110)

    def fit(self) -> None:
        self.resume_if_available()

        anneal_fn = getattr(self.memory_module, "anneal_memory_temperature", None)
        if not callable(anneal_fn):
            raise RuntimeError("MTAR must expose anneal_memory_temperature().")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            current_memory_temperature = anneal_fn(
                epoch=epoch,
                total_epochs=self.cfg.epochs,
                start_temperature=self.cfg.memory_temp_start,
                end_temperature=self.cfg.memory_temp_end,
                mode=self.cfg.memory_temp_mode,
            )

            train_metrics = self._train_one_epoch()
            val_metrics = self.validate()
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self._print_epoch(
                epoch,
                self.cfg.epochs,
                current_lr,
                current_memory_temperature,
                train_metrics,
                val_metrics,
            )

            memory_diagnostics = safe_memory_diagnostics(self.memory_module)
            save_memory_diagnostics_json(
                self.memory_module,
                self.latest_memory_diag_path,
            )
            self._save_epoch_memory_artifacts(epoch, current_memory_temperature)

            self._append_history({
                "epoch": epoch + 1,
                "lr": current_lr,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            })

            torch.save(self.model.state_dict(), self.last_model_path)

            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = float(val_metrics["total_loss"])
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.best_model_path)
                print(
                    f"New best model saved at epoch {epoch + 1} | "
                    f"normal validation total loss={self.best_val_loss:.6f}"
                )
                save_memory_diagnostics_json(
                    self.memory_module,
                    self.output_dir / "memory_diagnostics_best.json",
                )
                if self.cfg.save_memory_vis_on_best:
                    save_memory_bank_snapshot(
                        self.memory_module,
                        self.best_memory_vis_dir,
                        epoch_number=epoch + 1,
                        memory_temperature=current_memory_temperature,
                        save_pt=self.cfg.save_memory_bank_pt,
                        save_npy=self.cfg.save_memory_bank_npy,
                        save_normalized_npy=self.cfg.save_normalized_memory_bank_npy,
                    )
                    save_memory_visualizations(
                        self.memory_module,
                        self.best_memory_vis_dir,
                        max_heatmap_slots=self.cfg.max_heatmap_slots,
                        annotate_points=self.cfg.annotate_memory_pca,
                    )

            self.scheduler.step()
            torch.save(
                {
                    "protocol_version": self.cfg.protocol_version,
                    "seed": self.cfg.seed,
                    "validation_ratio": self.cfg.val_ratio,
                    "split_level": "video",
                    "uses_anomaly_labels": False,
                    "total_epochs": self.cfg.epochs,
                    "memory_regularization": self.cfg.memory_regularization_dict(),
                    "train_video_ids": self.train_video_ids,
                    "val_video_ids": self.val_video_ids,
                    "epoch": epoch,
                    "best_epoch": self.best_epoch,
                    "best_val_loss": self.best_val_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "memory_diagnostics": memory_diagnostics,
                    "memory_temperature": current_memory_temperature,
                },
                self.checkpoint_path,
            )

        self._finalize_best_model()

    def _finalize_best_model(self) -> None:
        print("\nTraining completed.")
        print(
            f"Best epoch: {self.best_epoch + 1} | "
            f"best normal validation total loss: {self.best_val_loss:.6f}"
        )
        if not self.best_model_path.exists():
            return

        best_state = torch.load(
            self.best_model_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(best_state)
        refresh_memory_statistics(
            self.model,
            self.val_loader,
            self.device,
            max_batches=1,
        )
        save_memory_diagnostics_json(
            self.memory_module,
            self.output_dir / "memory_diagnostics_final_best.json",
        )
        save_memory_bank_snapshot(
            self.memory_module,
            self.final_memory_vis_dir,
            epoch_number=self.best_epoch + 1,
            memory_temperature=None,
            save_pt=self.cfg.save_memory_bank_pt,
            save_npy=self.cfg.save_memory_bank_npy,
            save_normalized_npy=self.cfg.save_normalized_memory_bank_npy,
        )
        save_memory_visualizations(
            self.memory_module,
            self.final_memory_vis_dir,
            max_heatmap_slots=self.cfg.max_heatmap_slots,
            annotate_points=self.cfg.annotate_memory_pca,
        )

        print(f"Best model              : {self.best_model_path}")
        print(f"Last model              : {self.last_model_path}")
        print(f"Resumable checkpoint    : {self.checkpoint_path}")
        print(f"Training history        : {self.history_path}")
        print(f"Per-epoch memory results: {self.memory_epochs_dir}")
