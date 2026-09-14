"""Checkpoint helpers shared by CSE-SMNet training and evaluation."""
from __future__ import annotations

from typing import Any, Dict

import torch


def extract_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    """Return a model state dict from raw weights or a training checkpoint."""
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    if isinstance(loaded, dict) and any(torch.is_tensor(value) for value in loaded.values()):
        return loaded
    raise TypeError(
        "Checkpoint must be a raw state_dict or contain 'model_state_dict'."
    )
