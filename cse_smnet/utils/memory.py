"""Utilities for CSE-SMNet's MTAR structured prototype memory."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _supports_structured_memory(module: nn.Module) -> bool:
    return (
        callable(getattr(module, "memory_regularization_loss", None))
        and callable(getattr(module, "memory_diagnostics", None))
    )


def find_mtar_module(model: nn.Module) -> nn.Module:
    """Locate the outermost MTAR-compatible structured-memory module."""
    if _supports_structured_memory(model):
        return model

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and _supports_structured_memory(module)
    ]
    if not candidates:
        raise RuntimeError(
            "MTAR structured-memory module was not found inside CSESMNet. "
            "Expected memory_regularization_loss() and memory_diagnostics()."
        )

    candidates.sort(key=lambda item: (item[0].count("."), len(item[0])))
    selected_name, selected_module = candidates[0]
    print(f"MTAR structured-memory module detected: model.{selected_name}")
    return selected_module


def get_memory_losses(
    memory_module: nn.Module,
    *,
    lambda_compact: float,
    lambda_separate: float,
    lambda_diverse: float,
    lambda_usage: float,
) -> Dict[str, torch.Tensor]:
    losses = memory_module.memory_regularization_loss(
        lambda_compact=lambda_compact,
        lambda_separate=lambda_separate,
        lambda_diverse=lambda_diverse,
        lambda_usage=lambda_usage,
    )
    required = {"total", "compactness", "separation", "diversity", "usage"}
    missing = required.difference(losses)
    if missing:
        raise KeyError(f"memory_regularization_loss() is missing keys: {missing}")
    return losses


def safe_memory_diagnostics(memory_module: nn.Module) -> Dict[str, Any]:
    try:
        return memory_module.memory_diagnostics()
    except Exception as exc:  # diagnostics must never destroy a training run
        return {"error": f"{type(exc).__name__}: {exc}"}


def save_memory_diagnostics_json(memory_module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(safe_memory_diagnostics(memory_module), file, indent=2)


def save_memory_visualizations(
    memory_module: nn.Module,
    output_dir: Path,
    *,
    max_heatmap_slots: int,
    annotate_points: bool,
) -> None:
    visualizer = getattr(memory_module, "save_memory_visualizations", None)
    if not callable(visualizer):
        print("Warning: MTAR does not expose save_memory_visualizations(); skipped.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        visualizer(
            output_dir=output_dir,
            max_heatmap_slots=max_heatmap_slots,
            annotate_points=annotate_points,
            save_json=True,
        )
    except Exception as exc:
        print(
            "Warning: MTAR visualization failed, but training continues: "
            f"{type(exc).__name__}: {exc}"
        )


def _looks_like_memory_bank_parameter(name: str, tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        return False
    normalized_name = name.lower().replace("-", "_")
    leaf = normalized_name.split(".")[-1]
    if leaf in {"memory", "memory_bank", "prototype", "prototypes", "prototype_bank", "memory_slots"}:
        return True
    return any(
        pattern in normalized_name
        for pattern in ("prototype_memory.memory", "prototype_bank", "memory_bank", "memory_slots")
    )


def get_memory_bank_tensors(memory_module: nn.Module) -> Dict[str, torch.Tensor]:
    banks = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in memory_module.named_parameters()
        if _looks_like_memory_bank_parameter(name, parameter)
    }
    if not banks:
        available_2d = [
            name for name, parameter in memory_module.named_parameters()
            if parameter.ndim == 2
        ]
        raise RuntimeError(
            "No L x D prototype-memory parameter could be identified. "
            f"Available 2-D parameters include: {available_2d[:20]}"
        )
    return banks


def _safe_bank_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")
    return safe.strip("_") or "memory_bank"


def save_memory_bank_snapshot(
    memory_module: nn.Module,
    output_dir: Path,
    *,
    epoch_number: int,
    memory_temperature: Optional[float],
    save_pt: bool,
    save_npy: bool,
    save_normalized_npy: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    banks = get_memory_bank_tensors(memory_module)

    raw_pt: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Any] = {
        "epoch": int(epoch_number),
        "memory_temperature": None if memory_temperature is None else float(memory_temperature),
        "number_of_unique_memory_banks": len(banks),
        "banks": {},
    }

    for bank_index, (name, bank) in enumerate(banks.items(), start=1):
        normalized = F.normalize(bank, p=2, dim=1, eps=1e-12)
        raw_pt[name] = bank
        slot_norms = torch.linalg.vector_norm(bank, ord=2, dim=1)
        stem = _safe_bank_filename(name)
        canonical_single = len(banks) == 1

        if save_npy:
            np.save(output_dir / f"{stem}_raw.npy", bank.numpy())
            if canonical_single:
                np.save(output_dir / "memory_bank.npy", bank.numpy())
        if save_normalized_npy:
            np.save(output_dir / f"{stem}_normalized.npy", normalized.numpy())
            if canonical_single:
                np.save(output_dir / "memory_bank_normalized.npy", normalized.numpy())

        metadata["banks"][name] = {
            "bank_index": bank_index,
            "shape": [int(bank.shape[0]), int(bank.shape[1])],
            "num_slots_L": int(bank.shape[0]),
            "embedding_dim_D": int(bank.shape[1]),
            "dtype_saved": str(bank.dtype),
            "raw_mean": float(bank.mean().item()),
            "raw_std": float(bank.std(unbiased=False).item()),
            "raw_min": float(bank.min().item()),
            "raw_max": float(bank.max().item()),
            "mean_slot_l2_norm": float(slot_norms.mean().item()),
            "min_slot_l2_norm": float(slot_norms.min().item()),
            "max_slot_l2_norm": float(slot_norms.max().item()),
        }

    if save_pt:
        torch.save(
            {
                "epoch": int(epoch_number),
                "memory_temperature": None if memory_temperature is None else float(memory_temperature),
                "memory_banks": raw_pt,
            },
            output_dir / "memory_bank.pt",
        )

    with (output_dir / "memory_bank_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


@torch.no_grad()
def refresh_memory_statistics(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 1,
) -> None:
    model.eval()
    for batch_index, batch in enumerate(loader):
        frames = batch[0].to(device, non_blocking=True)
        _ = model(frames)
        if batch_index + 1 >= max_batches:
            break


def scheduled_memory_temperature(
    epoch_zero_based: int,
    total_epochs: int,
    start_temperature: float,
    end_temperature: float,
    mode: str,
) -> float:
    """Reconstruct the memory temperature used at a particular training epoch."""
    if total_epochs <= 1:
        progress = 1.0
    else:
        progress = epoch_zero_based / float(total_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)

    mode = mode.lower()
    if mode == "linear":
        return start_temperature + progress * (end_temperature - start_temperature)
    if mode == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_temperature + (start_temperature - end_temperature) * weight
    raise ValueError(f"Unknown memory temperature mode: {mode}")


def resolve_inference_memory_temperature(
    memory_module: nn.Module,
    *,
    training_checkpoint_path: Optional[Path],
    total_epochs: int,
    default_start_temperature: float,
    default_end_temperature: float,
    default_mode: str,
    restore_best_temperature: bool,
    fallback_temperature: float,
) -> float:
    """Restore MTAR retrieval temperature from training metadata when available."""
    setter = getattr(memory_module, "set_memory_temperature", None)
    if not callable(setter):
        raise RuntimeError("MTAR must expose set_memory_temperature().")

    temperature = float(fallback_temperature)
    checkpoint_path = None if training_checkpoint_path is None else Path(training_checkpoint_path)

    if restore_best_temperature and checkpoint_path is not None and checkpoint_path.exists():
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        best_epoch = int(metadata.get("best_epoch", -1))
        memory_cfg = metadata.get("memory_regularization", {})
        start_temp = float(memory_cfg.get("temperature_start", default_start_temperature))
        end_temp = float(memory_cfg.get("temperature_end", default_end_temperature))
        mode = str(memory_cfg.get("temperature_mode", default_mode))

        if best_epoch >= 0:
            schedule_epochs = int(metadata.get("total_epochs", total_epochs))
            temperature = scheduled_memory_temperature(
                best_epoch, schedule_epochs, start_temp, end_temp, mode
            )
            print(f"Best training epoch: {best_epoch + 1}")
        else:
            temperature = float(metadata.get("memory_temperature", fallback_temperature))
    else:
        print(
            "Training metadata unavailable; using fallback inference "
            f"temperature={temperature:.4f}"
        )

    setter(temperature)
    print(f"Inference MTAR memory temperature: {temperature:.6f}")
    return float(temperature)


class MemoryUsageAccumulator:
    """Aggregate direct and temporal MTAR prototype usage over a test video."""

    def __init__(self) -> None:
        self.direct_sum: Optional[np.ndarray] = None
        self.temporal_sum: Optional[np.ndarray] = None
        self.direct_count = 0
        self.temporal_count = 0

    def update(self, diagnostics: Dict[str, Any]) -> None:
        layer = diagnostics.get("layer1", diagnostics)
        direct = layer.get("direct_usage")
        temporal = layer.get("temporal_usage")

        if direct is not None:
            array = np.asarray(direct, dtype=np.float64)
            self.direct_sum = array if self.direct_sum is None else self.direct_sum + array
            self.direct_count += 1

        if temporal is not None:
            array = np.asarray(temporal, dtype=np.float64)
            self.temporal_sum = array if self.temporal_sum is None else self.temporal_sum + array
            self.temporal_count += 1

    @staticmethod
    def _stats(usage: Optional[np.ndarray]) -> Dict[str, Any]:
        if usage is None:
            return {}
        usage = np.asarray(usage, dtype=np.float64)
        usage = usage / max(float(usage.sum()), 1e-12)
        entropy = -np.sum(usage * np.log(usage + 1e-12))
        return {
            "usage": usage.tolist(),
            "effective_slots": float(np.exp(entropy)),
            "usage_entropy": float(entropy),
            "max_slot_usage": float(usage.max()),
            "min_slot_usage": float(usage.min()),
            "dominant_slot": int(np.argmax(usage)),
        }

    def finalize(self) -> Dict[str, Any]:
        direct = None if self.direct_sum is None else self.direct_sum / max(self.direct_count, 1)
        temporal = None if self.temporal_sum is None else self.temporal_sum / max(self.temporal_count, 1)
        return {
            "direct": self._stats(direct),
            "temporal": self._stats(temporal),
            "direct_batches": self.direct_count,
            "temporal_batches": self.temporal_count,
        }
