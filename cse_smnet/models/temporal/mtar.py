
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Small utilities
# ============================================================================

class LayerNormLastDim(nn.Module):
    """LayerNorm applied to the last tensor dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _zero_like_parameter(parameter: torch.Tensor) -> torch.Tensor:
    """Return a differentiable scalar zero on the same device/dtype."""
    return parameter.sum() * 0.0


# ============================================================================
# Structured learnable prototype memory
# ============================================================================

class PrototypeMemoryBank(nn.Module):
    """
    Learnable prototype memory used by both MTAR retrieval branches.

    Parameters
    ----------
    num_slots:
        Number of memory slots L.
    embedding_dim:
        Prototype dimension D.
    temperature:
        Temperature for cosine-softmax addressing.
    top_k:
        Optional sparse retrieval. If None, all slots participate. If an integer,
        only the top-k similarities are retained before softmax.
    eps:
        Numerical stability constant used during L2 normalization.
    """

    def __init__(
        self,
        num_slots: int,
        embedding_dim: int,
        temperature: float = 0.30,
        top_k: Optional[int] = None,
        diversity_margin: float = 0.20,
        eps: float = 1e-8,
    ):
        super().__init__()

        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and (top_k < 1 or top_k > num_slots):
            raise ValueError("top_k must satisfy 1 <= top_k <= num_slots")
        if not 0.0 <= diversity_margin < 1.0:
            raise ValueError("diversity_margin must satisfy 0 <= margin < 1")

        self.num_slots = int(num_slots)
        self.embedding_dim = int(embedding_dim)
        self.temperature = float(temperature)
        self.top_k = top_k
        self.diversity_margin = float(diversity_margin)
        self.eps = float(eps)

        # Ordinary trainable parameters. They are NOT dynamically written from
        # individual samples and are NOT updated at inference time.
        self.memory = nn.Parameter(torch.empty(num_slots, embedding_dim))
        self.reset_parameters()

        # Last-forward diagnostics. These are plain attributes, not persistent
        # buffers, so checkpoints stay compact and backward compatible.
        self.last_usage: Optional[torch.Tensor] = None
        self.last_attention_entropy: Optional[torch.Tensor] = None
        self.last_effective_slots: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        """Xavier-uniform initialization for the prototype matrix."""
        nn.init.xavier_uniform_(self.memory)

    def normalized_memory(self) -> torch.Tensor:
        """Return row-wise L2-normalized prototypes, shape (L, D)."""
        return F.normalize(self.memory, p=2, dim=-1, eps=self.eps)


    def set_temperature(self, temperature: float) -> None:
        """Set cosine-softmax retrieval temperature."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def anneal_temperature(
        self,
        epoch: int,
        total_epochs: int,
        start_temperature: float = 0.30,
        end_temperature: float = 0.10,
        mode: str = "cosine",
    ) -> float:
        """Anneal retrieval temperature to avoid premature one-slot collapse."""
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")
        if start_temperature <= 0 or end_temperature <= 0:
            raise ValueError("temperatures must be > 0")

        if total_epochs == 1:
            progress = 1.0
        else:
            progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)

        mode = mode.lower()
        if mode == "linear":
            temperature = start_temperature + progress * (
                end_temperature - start_temperature
            )
        elif mode == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * progress))
            temperature = end_temperature + (
                start_temperature - end_temperature
            ) * w
        else:
            raise ValueError("mode must be 'linear' or 'cosine'")

        self.set_temperature(temperature)
        return self.temperature

    def cosine_similarity(self, query: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarities between arbitrary query tensors and memory slots.

        query: (..., D)
        returns: (..., L)
        """
        if query.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Query dimension {query.shape[-1]} does not match memory "
                f"dimension {self.embedding_dim}."
            )

        q = F.normalize(query, p=2, dim=-1, eps=self.eps)
        m = self.normalized_memory()
        return torch.matmul(q, m.t())

    def _attention_from_similarity(self, similarity: torch.Tensor) -> torch.Tensor:
        logits = similarity / self.temperature

        if self.top_k is not None and self.top_k < self.num_slots:
            top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, top_indices, top_values)
            logits = masked_logits

        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def _update_usage_diagnostics(self, attention: torch.Tensor) -> None:
        """Track average slot usage and entropy from the most recent read."""
        if attention.numel() == 0:
            return

        flat = attention.detach().reshape(-1, self.num_slots)
        usage = flat.mean(dim=0)
        usage = usage / usage.sum().clamp_min(self.eps)

        entropy = -(usage * (usage + self.eps).log()).sum()
        effective_slots = entropy.exp()

        self.last_usage = usage.cpu()
        self.last_attention_entropy = entropy.cpu()
        self.last_effective_slots = effective_slots.cpu()

    def retrieve(
        self,
        query: torch.Tensor,
        update_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a weighted normal prototype representation.

        query: (..., D)
        returns:
            retrieved: (..., D)
            attention: (..., L)
            similarity: (..., L)
        """
        similarity = self.cosine_similarity(query)
        attention = self._attention_from_similarity(similarity)

        # Reading normalized prototypes makes the scale of the retrieved prior
        # stable and keeps addressing geometry consistent with the losses.
        retrieved = torch.matmul(attention, self.normalized_memory())

        if update_diagnostics:
            self._update_usage_diagnostics(attention)

        return retrieved, attention, similarity

    def utilization_loss(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Branch-level batch utilization regularization.

        Computes KL(u || Uniform), where u is the mean slot usage over all
        queries in the supplied attention tensor:

            L_use = sum_k u_k log(u_k * L)

        The loss is 0 for perfectly broad aggregate use and approaches log(L)
        for complete one-slot collapse. Individual query attention is not forced
        to be uniform.
        """
        if attention.numel() == 0:
            return _zero_like_parameter(self.memory)

        flat = attention.reshape(-1, self.num_slots)
        usage = flat.mean(dim=0)
        usage = usage / usage.sum().clamp_min(self.eps)

        return (
            usage
            * ((usage + self.eps).log() + math.log(float(self.num_slots)))
        ).sum()

    def assignment_losses(
        self,
        query: torch.Tensor,
        margin: float = 0.10,
    ) -> Dict[str, torch.Tensor]:
        """
        Query-to-prototype compactness and separation losses.

        Compactness:
            pulls each normal query toward its closest prototype.

        Separation:
            requires the best prototype similarity to exceed the second-best
            similarity by at least ``margin``.
        """
        if margin < 0:
            raise ValueError("margin must be >= 0")

        similarity = self.cosine_similarity(query)
        flat = similarity.reshape(-1, self.num_slots)

        if flat.numel() == 0:
            z = _zero_like_parameter(self.memory)
            return {"compactness": z, "separation": z}

        best = flat.max(dim=-1).values
        compactness = (1.0 - best).mean()

        if self.num_slots >= 2:
            top2 = torch.topk(flat, k=2, dim=-1).values
            best_sim = top2[:, 0]
            second_sim = top2[:, 1]
            separation = F.relu(margin + second_sim - best_sim).mean()
        else:
            separation = _zero_like_parameter(self.memory)

        return {
            "compactness": compactness,
            "separation": separation,
        }

    def diversity_loss(self) -> torch.Tensor:
        """
        Margin-based prototype redundancy penalty.

        Only excessive absolute cosine similarity above ``diversity_margin`` is
        penalized. This prevents duplicate slots without forcing all related
        normal-motion prototypes to become nearly orthogonal.

            L_div = mean( ReLU(|cos(m_i,m_j)| - rho)^2 ),  i != j
        """
        if self.num_slots < 2:
            return _zero_like_parameter(self.memory)

        m = self.normalized_memory()
        gram = torch.matmul(m, m.t())
        mask = ~torch.eye(
            self.num_slots,
            device=gram.device,
            dtype=torch.bool,
        )
        off_diag = gram[mask].abs()
        penalty = F.relu(off_diag - self.diversity_margin)
        return penalty.pow(2).mean()

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        """Return interpretable memory-collapse and usage diagnostics."""
        m = self.normalized_memory()

        if self.num_slots >= 2:
            gram = torch.matmul(m, m.t())
            mask = ~torch.eye(self.num_slots, device=gram.device, dtype=torch.bool)
            off = gram[mask]
            mean_abs_cos = off.abs().mean().item()
            max_abs_cos = off.abs().max().item()
            mean_sq_cos = off.pow(2).mean().item()
        else:
            mean_abs_cos = 0.0
            max_abs_cos = 0.0
            mean_sq_cos = 0.0

        result = {
            "num_slots": float(self.num_slots),
            "embedding_dim": float(self.embedding_dim),
            "mean_abs_offdiag_cosine": mean_abs_cos,
            "max_abs_offdiag_cosine": max_abs_cos,
            "mean_squared_offdiag_cosine": mean_sq_cos,
            "diversity_margin": float(self.diversity_margin),
            "temperature": float(self.temperature),
        }

        if self.last_usage is not None:
            result.update({
                "effective_slots": float(self.last_effective_slots.item()),
                "usage_entropy": float(self.last_attention_entropy.item()),
                "max_slot_usage": float(self.last_usage.max().item()),
                "min_slot_usage": float(self.last_usage.min().item()),
            })

        return result


# ============================================================================
# Memory-guided temporal attention branch
# ============================================================================

class MemoryGuidedTemporalAttention(nn.Module):
    """
    Temporal self-attention augmented by the SAME prototype bank used by the
    direct memory-retrieval branch of the enclosing recurrent cell.
    """

    def __init__(
        self,
        channels: int,
        prototype_memory: PrototypeMemoryBank,
        num_heads: int = 4,
        dropout: float = 0.1,
        separation_margin: float = 0.10,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        if prototype_memory.embedding_dim != channels:
            raise ValueError(
                "prototype_memory.embedding_dim must equal channels in the current "
                "MTAR implementation."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.prototype_memory = prototype_memory
        self.memory_slots = prototype_memory.num_slots
        self.separation_margin = separation_margin

        self.norm = LayerNormLastDim(channels)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)

        self.mem_k_proj = nn.Linear(channels, channels, bias=False)
        self.mem_v_proj = nn.Linear(channels, channels, bias=False)

        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(channels * 2, channels)

        self.last_compactness: Optional[torch.Tensor] = None
        self.last_separation: Optional[torch.Tensor] = None
        self.last_memory_usage: Optional[torch.Tensor] = None
        self.last_usage_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, C, T, H, W)."""
        B, C, T, H, W = x.shape
        N = H * W

        # Every spatial location forms an independent temporal sequence.
        x_tokens = (
            x.permute(0, 3, 4, 2, 1)
             .contiguous()
             .view(B * N, T, C)
        )
        x_norm = self.norm(x_tokens)

        q_base = self.q_proj(x_norm)                         # (B*N, T, C)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Shared, normalized prototype bank.
        mem_base = self.prototype_memory.normalized_memory() # (L, C)
        mem = mem_base.unsqueeze(0).expand(B * N, -1, -1)   # (B*N, L, C)
        mem_k = self.mem_k_proj(mem)
        mem_v = self.mem_v_proj(mem)

        # Temporal attention is augmented by memory keys/values.
        k_all = torch.cat([k, mem_k], dim=1)
        v_all = torch.cat([v, mem_v], dim=1)

        q = (
            q_base.view(B * N, T, self.num_heads, self.head_dim)
                  .transpose(1, 2)
        )
        k_all = (
            k_all.view(B * N, T + self.memory_slots, self.num_heads, self.head_dim)
                 .transpose(1, 2)
        )
        v_all = (
            v_all.view(B * N, T + self.memory_slots, self.num_heads, self.head_dim)
                 .transpose(1, 2)
        )

        attn = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_all)
        out = (
            out.transpose(1, 2)
               .contiguous()
               .view(B * N, T, C)
        )

        # Explicit normality-prior retrieval uses cosine addressing.
        mem_read, mem_attn, _ = self.prototype_memory.retrieve(q_base)

        # Query-aware memory losses for this branch. They are stored so the outer
        # module can add them to the training objective without changing forward().
        losses = self.prototype_memory.assignment_losses(
            q_base,
            margin=self.separation_margin,
        )
        self.last_compactness = losses["compactness"]
        self.last_separation = losses["separation"]
        self.last_usage_loss = self.prototype_memory.utilization_loss(mem_attn)
        self.last_memory_usage = mem_attn.detach().mean(dim=(0, 1)).cpu()

        # Adaptive fusion between temporal context and explicit memory retrieval.
        fused = torch.cat([out, mem_read], dim=-1)
        gate = torch.sigmoid(self.gate_proj(fused))
        out = gate * out + (1.0 - gate) * mem_read
        out = self.out_proj(out)

        return (
            out.view(B, H, W, T, C)
               .permute(0, 4, 3, 1, 2)
               .contiguous()
        )


# ============================================================================
# Memory-augmented ConvLSTM cell
# ============================================================================

class MTARCell(nn.Module):
    """
    Conv2D recurrent cell with two memory-based retrieval paths:

    Path A: hidden-state -> direct prototype retrieval.
    Path B: memory-guided temporal attention using the same prototype bank.

    A single shared bank inside the cell prevents unnecessary duplicate banks
    from learning conflicting notions of normality.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        mem_slots: int = 150,
        kernel_size: int = 3,
        num_heads: int = 4,
        memory_temperature: float = 0.30,
        memory_top_k: Optional[int] = None,
        separation_margin: float = 0.10,
        diversity_margin: float = 0.20,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.mem_slots = mem_slots
        self.separation_margin = separation_margin

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

        # One prototype bank is shared by the direct and temporal memory paths.
        self.prototype_memory = PrototypeMemoryBank(
            num_slots=mem_slots,
            embedding_dim=hidden_channels,
            temperature=memory_temperature,
            top_k=memory_top_k,
            diversity_margin=diversity_margin,
        )

        self.query_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.mem_proj = nn.Linear(hidden_channels, hidden_channels)
        self.fusion = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1)

        self.temporal_attention = MemoryGuidedTemporalAttention(
            channels=hidden_channels,
            prototype_memory=self.prototype_memory,
            num_heads=num_heads,
            dropout=dropout,
            separation_margin=separation_margin,
        )

        self.memory_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Loss terms from the most recent forward pass.
        self.last_direct_compactness: Optional[torch.Tensor] = None
        self.last_direct_separation: Optional[torch.Tensor] = None
        self.last_direct_usage: Optional[torch.Tensor] = None
        self.last_direct_usage_loss: Optional[torch.Tensor] = None

    @property
    def memory(self) -> torch.nn.Parameter:
        """Compatibility accessor for older code that used cell.memory."""
        return self.prototype_memory.memory

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:      (B, Cin, T, H, W)
        h_prev: (B, Ch,  T, H, W)
        c_prev: (B, Ch,  T, H, W)
        """
        B, C, T, H, W = x.shape

        x2d = (
            x.permute(0, 2, 1, 3, 4)
             .contiguous()
             .view(B * T, C, H, W)
        )
        h2d = (
            h_prev.permute(0, 2, 1, 3, 4)
                  .contiguous()
                  .view(B * T, self.hidden_channels, H, W)
        )
        c2d = (
            c_prev.permute(0, 2, 1, 3, 4)
                  .contiguous()
                  .view(B * T, self.hidden_channels, H, W)
        )

        # ConvLSTM update.
        combined = torch.cat([x2d, h2d], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c2d + i * g
        h = o * torch.tanh(c)

        # ------------------------------------------------------------------
        # Direct memory retrieval branch
        # ------------------------------------------------------------------
        query_map = self.query_conv(h)                       # (B*T, Ch, H, W)
        query = query_map.mean(dim=(2, 3))                  # (B*T, Ch)

        mem_read_vec, direct_attn, _ = self.prototype_memory.retrieve(query)
        direct_losses = self.prototype_memory.assignment_losses(
            query,
            margin=self.separation_margin,
        )
        self.last_direct_compactness = direct_losses["compactness"]
        self.last_direct_separation = direct_losses["separation"]
        self.last_direct_usage_loss = self.prototype_memory.utilization_loss(
            direct_attn
        )
        self.last_direct_usage = direct_attn.detach().mean(dim=0).cpu()

        # Learned projection is applied AFTER prototype retrieval; addressing
        # itself remains cosine-normalized and directly interpretable.
        mem_read_vec = self.mem_proj(mem_read_vec)
        mem_read = (
            mem_read_vec.view(B * T, self.hidden_channels, 1, 1)
                        .expand(-1, -1, H, W)
        )

        h_mem = self.fusion(torch.cat([h, mem_read], dim=1))

        # ------------------------------------------------------------------
        # Memory-guided temporal retrieval branch
        # ------------------------------------------------------------------
        h_mem_5d = (
            h_mem.view(B, T, self.hidden_channels, H, W)
                 .permute(0, 2, 1, 3, 4)
                 .contiguous()
        )
        h_attn = self.temporal_attention(h_mem_5d)
        h_attn = (
            h_attn.permute(0, 2, 1, 3, 4)
                  .contiguous()
                  .view(B * T, self.hidden_channels, H, W)
        )

        # Adaptive fusion of the two memory-aware branches.
        gate = self.memory_gate(torch.cat([h_mem, h_attn], dim=1))
        h_final = gate * h_mem + (1.0 - gate) * h_attn

        h_final = (
            h_final.view(B, T, self.hidden_channels, H, W)
                   .permute(0, 2, 1, 3, 4)
                   .contiguous()
        )
        c = (
            c.view(B, T, self.hidden_channels, H, W)
             .permute(0, 2, 1, 3, 4)
             .contiguous()
        )

        return h_final, c

    def memory_regularization_loss(
        self,
        lambda_compact: float = 1.0,
        lambda_separate: float = 1.0,
        lambda_diverse: float = 1.0,
        lambda_usage: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Structured-memory objective for the single MTAR cell.

        Compactness and separation are averaged across the direct and temporal
        retrieval branches. Diversity is computed once on the shared prototype
        bank. Utilization is regularized separately for both branches before
        averaging, so collapse in one branch cannot be hidden by the other.
        """
        z = _zero_like_parameter(self.prototype_memory.memory)

        direct_c = (
            self.last_direct_compactness
            if self.last_direct_compactness is not None else z
        )
        direct_s = (
            self.last_direct_separation
            if self.last_direct_separation is not None else z
        )
        direct_u = (
            self.last_direct_usage_loss
            if self.last_direct_usage_loss is not None else z
        )

        temporal_c = (
            self.temporal_attention.last_compactness
            if self.temporal_attention.last_compactness is not None else z
        )
        temporal_s = (
            self.temporal_attention.last_separation
            if self.temporal_attention.last_separation is not None else z
        )
        temporal_u = (
            self.temporal_attention.last_usage_loss
            if self.temporal_attention.last_usage_loss is not None else z
        )

        compactness = 0.5 * (direct_c + temporal_c)
        separation = 0.5 * (direct_s + temporal_s)
        utilization = 0.5 * (direct_u + temporal_u)
        diversity = self.prototype_memory.diversity_loss()

        total = (
            lambda_compact * compactness
            + lambda_separate * separation
            + lambda_diverse * diversity
            + lambda_usage * utilization
        )

        return {
            "total": total,
            "compactness": compactness,
            "separation": separation,
            "diversity": diversity,
            "usage": utilization,
            "direct_compactness": direct_c,
            "direct_separation": direct_s,
            "direct_usage_loss": direct_u,
            "temporal_compactness": temporal_c,
            "temporal_separation": temporal_s,
            "temporal_usage_loss": temporal_u,
        }

    @staticmethod
    def _usage_statistics(
        usage: Optional[torch.Tensor],
        eps: float = 1e-8,
    ) -> Dict[str, Union[float, list]]:
        if usage is None:
            return {}

        u = usage.detach().float().cpu()
        u = u / u.sum().clamp_min(eps)
        entropy = -(u * (u + eps).log()).sum()
        effective = entropy.exp()

        return {
            "usage": u.tolist(),
            "effective_slots": float(effective.item()),
            "usage_entropy": float(entropy.item()),
            "max_slot_usage": float(u.max().item()),
            "min_slot_usage": float(u.min().item()),
        }

    @torch.no_grad()
    def memory_diagnostics(self) -> Dict[str, Union[float, list, dict]]:
        d = self.prototype_memory.diagnostics()

        direct_stats = self._usage_statistics(self.last_direct_usage)
        temporal_stats = self._usage_statistics(
            self.temporal_attention.last_memory_usage
        )

        if direct_stats:
            d["direct_usage"] = direct_stats["usage"]
            d["direct_effective_slots"] = direct_stats["effective_slots"]
            d["direct_usage_entropy"] = direct_stats["usage_entropy"]
            d["direct_max_slot_usage"] = direct_stats["max_slot_usage"]
            d["direct_min_slot_usage"] = direct_stats["min_slot_usage"]

        if temporal_stats:
            d["temporal_usage"] = temporal_stats["usage"]
            d["temporal_effective_slots"] = temporal_stats["effective_slots"]
            d["temporal_usage_entropy"] = temporal_stats["usage_entropy"]
            d["temporal_max_slot_usage"] = temporal_stats["max_slot_usage"]
            d["temporal_min_slot_usage"] = temporal_stats["min_slot_usage"]

        effective_values = []
        max_usage_values = []

        if direct_stats:
            effective_values.append(direct_stats["effective_slots"])
            max_usage_values.append(direct_stats["max_slot_usage"])

        if temporal_stats:
            effective_values.append(temporal_stats["effective_slots"])
            max_usage_values.append(temporal_stats["max_slot_usage"])

        if effective_values:
            d["effective_slots"] = float(min(effective_values))
        if max_usage_values:
            d["max_slot_usage"] = float(max(max_usage_values))

        return d

class MTAR(nn.Module):
    """
    Single-cell MTAR recurrent module with structured prototype memory.

    The public API is intentionally kept compatible with the previous
    two-cell wrapper:
      - forward(...)
      - memory_regularization_loss(...)
      - get_memory_banks(...)
      - memory_diagnostics(...)
      - save_memory_visualizations(...)

    For backward compatibility with existing training/visualization code,
    diagnostic dictionaries still use the key ``"layer1"`` even though the
    module now contains only one recurrent cell.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        mem_slots: int = 150,
        num_heads: int = 4,
        memory_temperature: float = 0.30,
        memory_top_k: Optional[int] = None,
        separation_margin: float = 0.10,
        diversity_margin: float = 0.20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell1 = MTARCell(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            mem_slots=mem_slots,
            num_heads=num_heads,
            memory_temperature=memory_temperature,
            memory_top_k=memory_top_k,
            separation_margin=separation_margin,
            diversity_margin=diversity_margin,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        Args
        ----
        x:
            Input feature tensor of shape ``(B, C, T, H, W)``.
        return_aux:
            If ``False`` (default), return only the single-cell hidden output.
            If ``True``, also return memory losses and diagnostics.

        Returns
        -------
        h:
            Output hidden tensor of shape
            ``(B, hidden_channels, T, H, W)``.
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected x with shape (B, C, T, H, W), got {tuple(x.shape)}"
            )

        B, _, T, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # --------------------------------------------------------------
        # Single recurrent state
        # --------------------------------------------------------------
        h = torch.zeros(
            B,
            self.cell1.hidden_channels,
            T,
            H,
            W,
            device=device,
            dtype=dtype,
        )
        c = torch.zeros_like(h)

        # --------------------------------------------------------------
        # Single MTAR cell
        # --------------------------------------------------------------
        h, c = self.cell1(x, h, c)

        if not return_aux:
            return h

        return h, {
            "memory_losses": self.memory_regularization_loss(),
            "memory_diagnostics": self.memory_diagnostics(),
        }

    def memory_regularization_loss(
        self,
        lambda_compact: float = 1.0,
        lambda_separate: float = 1.0,
        lambda_diverse: float = 1.0,
        lambda_usage: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Return single-cell structured-memory regularization.

        The original public keys are preserved, with the new branch-aware
        utilization component added.
        """
        losses = self.cell1.memory_regularization_loss(
            lambda_compact=lambda_compact,
            lambda_separate=lambda_separate,
            lambda_diverse=lambda_diverse,
            lambda_usage=lambda_usage,
        )

        return {
            "compactness": losses["compactness"],
            "separation": losses["separation"],
            "diversity": losses["diversity"],
            "usage": losses["usage"],
            "direct_usage_loss": losses["direct_usage_loss"],
            "temporal_usage_loss": losses["temporal_usage_loss"],
            "total": losses["total"],
            "layer1_total": losses["total"],
        }

    def set_memory_temperature(self, temperature: float) -> None:
        """Set shared memory retrieval temperature."""
        self.cell1.prototype_memory.set_temperature(temperature)

    def anneal_memory_temperature(
        self,
        epoch: int,
        total_epochs: int,
        start_temperature: float = 0.30,
        end_temperature: float = 0.10,
        mode: str = "cosine",
    ) -> float:
        """Anneal shared retrieval temperature and return the current value."""
        return self.cell1.prototype_memory.anneal_temperature(
            epoch=epoch,
            total_epochs=total_epochs,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
            mode=mode,
        )

    @property
    def memory_temperature(self) -> float:
        return float(self.cell1.prototype_memory.temperature)

    @torch.no_grad()
    def get_memory_banks(
        self,
        normalized: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Return a detached copy of the single structured prototype bank.

        The key ``"layer1"`` is intentionally preserved so existing
        visualization and diagnostic code does not require modification.
        """
        if normalized:
            memory = self.cell1.prototype_memory.normalized_memory()
        else:
            memory = self.cell1.prototype_memory.memory

        return {
            "layer1": memory.detach().cpu().clone(),
        }

    @torch.no_grad()
    def memory_diagnostics(
        self,
    ) -> Dict[str, Dict[str, Union[float, list]]]:
        """
        Return diagnostics for the single prototype-memory bank.

        The wrapper key remains ``"layer1"`` for compatibility.
        """
        return {
            "layer1": self.cell1.memory_diagnostics(),
        }

    @torch.no_grad()
    def save_memory_visualizations(
        self,
        output_dir: Union[str, os.PathLike] = "memory_visualizations",
        max_heatmap_slots: int = 200,
        annotate_points: bool = False,
        save_json: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """
        Save visualizations for the single MTAR prototype-memory bank.

        Generated files
        ---------------
        1. ``layer1_pca.png``
           2-D PCA projection of prototype slots.

        2. ``layer1_cosine_similarity.png``
           Pairwise cosine-similarity matrix. For large banks, an evenly
           sampled subset of at most ``max_heatmap_slots`` slots is shown.

        3. ``layer1_slot_usage.png``
           Direct-retrieval and temporal-retrieval usage from the latest
           forward pass, when available.

        4. ``memory_diagnostics.json``
           Numeric redundancy/utilization diagnostics.

        PCA uses NumPy SVD, so scikit-learn is not required.
        Matplotlib is imported lazily and is required only for visualization.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Memory visualization requires matplotlib and numpy."
            ) from exc

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        banks = self.get_memory_banks(normalized=True)
        diagnostics = self.memory_diagnostics()
        files: Dict[str, Dict[str, str]] = {}

        for layer_name, tensor in banks.items():
            x = tensor.numpy()
            n_slots = x.shape[0]
            layer_files: Dict[str, str] = {}

            # ----------------------------------------------------------
            # PCA prototype map
            # ----------------------------------------------------------
            centered = x - x.mean(axis=0, keepdims=True)

            if centered.shape[0] >= 2 and centered.shape[1] >= 2:
                # SVD-based PCA: scores = U * S.
                u, s, _ = np.linalg.svd(
                    centered,
                    full_matrices=False,
                )
                coords = u[:, :2] * s[:2]
            else:
                coords = np.zeros(
                    (n_slots, 2),
                    dtype=x.dtype,
                )
                if n_slots:
                    coords[:, 0] = centered[:, 0]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=28,
                alpha=0.8,
            )
            ax.set_title(
                f"{layer_name}: prototype memory PCA"
            )
            ax.set_xlabel("Principal component 1")
            ax.set_ylabel("Principal component 2")
            ax.grid(True, alpha=0.25)

            if annotate_points and n_slots <= 200:
                for idx, (px, py) in enumerate(coords):
                    ax.annotate(
                        str(idx),
                        (px, py),
                        fontsize=6,
                        alpha=0.7,
                    )

            pca_path = out_dir / f"{layer_name}_pca.png"
            fig.tight_layout()
            fig.savefig(
                pca_path,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
            layer_files["pca"] = str(pca_path)

            # ----------------------------------------------------------
            # Cosine-similarity heatmap
            # ----------------------------------------------------------
            if n_slots > max_heatmap_slots:
                indices = np.linspace(
                    0,
                    n_slots - 1,
                    max_heatmap_slots,
                    dtype=int,
                )
                heat = x[indices]
                heat_title = (
                    f"{layer_name}: cosine similarity "
                    f"({max_heatmap_slots}/{n_slots} slots)"
                )
            else:
                heat = x
                heat_title = (
                    f"{layer_name}: prototype cosine similarity"
                )

            cosine = heat @ heat.T

            fig, ax = plt.subplots(figsize=(8, 7))
            image = ax.imshow(
                cosine,
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
            )
            ax.set_title(heat_title)
            ax.set_xlabel("Memory slot")
            ax.set_ylabel("Memory slot")
            fig.colorbar(
                image,
                ax=ax,
                label="Cosine similarity",
            )

            cos_path = (
                out_dir
                / f"{layer_name}_cosine_similarity.png"
            )
            fig.tight_layout()
            fig.savefig(
                cos_path,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
            layer_files["cosine_similarity"] = str(cos_path)

            # ----------------------------------------------------------
            # Latest attention usage
            # ----------------------------------------------------------
            layer_diag = diagnostics[layer_name]
            direct = layer_diag.get("direct_usage")
            temporal = layer_diag.get("temporal_usage")

            if direct is not None or temporal is not None:
                fig, ax = plt.subplots(figsize=(10, 4.8))
                slots = np.arange(n_slots)

                if direct is not None:
                    ax.plot(
                        slots,
                        np.asarray(direct),
                        label="Direct retrieval",
                    )

                if temporal is not None:
                    ax.plot(
                        slots,
                        np.asarray(temporal),
                        label="Temporal retrieval",
                    )

                ax.set_title(
                    f"{layer_name}: memory-slot usage"
                )
                ax.set_xlabel("Memory slot index")
                ax.set_ylabel("Mean attention probability")
                ax.grid(True, alpha=0.25)
                ax.legend()

                usage_path = (
                    out_dir
                    / f"{layer_name}_slot_usage.png"
                )
                fig.tight_layout()
                fig.savefig(
                    usage_path,
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)
                layer_files["slot_usage"] = str(usage_path)

            files[layer_name] = layer_files

        if save_json:
            json_path = out_dir / "memory_diagnostics.json"

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    diagnostics,
                    f,
                    indent=2,
                )

            files["diagnostics"] = {
                "json": str(json_path)
            }

        return files
