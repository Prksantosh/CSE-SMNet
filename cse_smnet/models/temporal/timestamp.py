"""Temporal layout transformation used by CSE-SMNet."""

from __future__ import annotations

import torch
import torch.nn as nn


class TimestampTransform(nn.Module):
    """Convert frame-wise encoder features into a spatiotemporal tensor.

    The encoder processes video frames by merging batch and temporal dimensions.
    This parameter-free module restores the temporal axis before MTAR processing.

    Input shape:
        ``(B * T, C, H, W)``

    Output shape:
        ``(B, C, T, H, W)``

    Notes
    -----
    This operation intentionally preserves the exact layout transformation used
    by the trained CSE-SMNet implementation.
    """

    def forward(
        self,
        x: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        """Restore the temporal dimension and move it after channels."""
        channels, height, width = x.shape[1:]

        x = x.view(
            batch_size,
            sequence_length,
            channels,
            height,
            width,
        )

        x = x.permute(0, 2, 1, 3, 4)

        return x
