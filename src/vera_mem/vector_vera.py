"""Memory-conditioned extension of vector-scaled random low-rank adaptation.

This is an experimental extension, not the published standard VeRA algorithm.
The layer input creates a query; retrieved external values supply the rank-wise
scaling of frozen random projections. Input-derived keys and values can be
written to an external bank after an observation. This module neither owns nor
silently updates that bank: the runner enforces read-before-write causality.

For each token, with a softmax over its top-k cosine matches:

    value = sum_i attention_i * stored_value_i
    delta = b * B((A x) * value)

A and B are frozen buffers. The shared output scaling b and the query, key, and
value encoders are trainable. Offline episodic training can differentiate through
selected scores and values. Hard top-k index selection itself is not smooth.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _temperature(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("temperature must be finite and positive")
    return result


class VectorVeRA(nn.Module):
    """Per-token sparse value retrieval conditions a frozen low-rank residual.

    ``x`` has shape ``[..., in_features]``. ``keys`` and ``values`` have shapes
    ``[N, key_dim]`` and ``[N, rank]``. ``forward`` is the differentiable offline
    tensor path. Online CPU retrieval can use ``encode_query`` followed by
    ``delta_from_value`` with only the sparse mixed value copied to the GPU.
    No persistence or detach is performed here. Encoding returns float32,
    while the residual is returned in the input dtype.

    The zero-initialized ``b`` makes initialization an exact no-op. On the first
    supervised step only ``b`` receives a nonzero task gradient; subsequent
    steps can train the other encoders. The auxiliary contrastive loss can train
    query/key addressing from the start, independently of ``b``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        key_dim: int = 64,
        top_k: int = 4,
        temperature: float = 0.2,
        seed: int = 42,
        value_centering: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = _positive_integer("in_features", in_features)
        self.out_features = _positive_integer("out_features", out_features)
        self.rank = _positive_integer("rank", rank)
        self.key_dim = _positive_integer("key_dim", key_dim)
        self.top_k = _positive_integer("top_k", top_k)
        self.temperature = _temperature(temperature)
        self.seed = seed
        if not isinstance(value_centering, bool):
            raise TypeError("value_centering must be a bool")
        self.value_centering = value_centering
        # Save/restore CPU RNG and avoid torch.manual_seed, which also changes
        # CUDA generators. Construction must not alter the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.register_buffer(
                "A", torch.randn(rank, in_features, dtype=torch.float32) / math.sqrt(in_features)
            )
            self.register_buffer(
                "B", torch.randn(out_features, rank, dtype=torch.float32) / math.sqrt(rank)
            )
            self.Wq = nn.Linear(in_features, key_dim, bias=False, dtype=torch.float32)
            self.Wk = nn.Linear(in_features, key_dim, bias=False, dtype=torch.float32)
            self.Wv = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
            with torch.no_grad():
                self.Wk.weight.copy_(self.Wq.weight)
        self.b = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        # Keep the default state_dict identical to pre-centering checkpoints.
        if self.value_centering:
            self.register_buffer("value_center", torch.zeros(in_features, dtype=torch.float32))

    def _normalized_input(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected x[..., {self.in_features}], got {tuple(x.shape)}")
        value = x.to(dtype=self.Wq.weight.dtype)
        # RMS-normalize without mean subtraction; zero inputs remain zero.
        return F.normalize(value, p=2, dim=-1, eps=1e-12) * math.sqrt(self.in_features)

    def encode_query(self, x: Tensor) -> Tensor:
        return F.normalize(self.Wq(self._normalized_input(x)), dim=-1, eps=1e-12)

    def encode_key(self, x: Tensor) -> Tensor:
        return F.normalize(self.Wk(self._normalized_input(x)), dim=-1, eps=1e-12)

    def encode_value(self, x: Tensor) -> Tensor:
        normalized = self._normalized_input(x)
        if self.value_centering:
            normalized = normalized - self.value_center
        return torch.tanh(self.Wv(normalized))

    @torch.no_grad()
    def fit_value_center(self, training_support_inputs: Tensor) -> "VectorVeRA":
        """Fit a fixed value-only mean from offline training observations.

        Call once before offline optimization, using only training support
        features; never include development/test/online observations. The caller
        owns this split policy. Query/key encoders and the A x projection are
        unaffected. The fitted buffer is saved with the module and is not a
        parameter or an online running statistic.
        """
        if not self.value_centering:
            raise RuntimeError("Enable value_centering before fitting a value center")
        if training_support_inputs.ndim != 2 or training_support_inputs.shape[0] == 0:
            raise ValueError("Training supports must be nonempty [N, in_features]")
        normalized = self._normalized_input(training_support_inputs)
        if not torch.isfinite(normalized).all():
            raise ValueError("Training supports must contain finite values")
        self.value_center.copy_(normalized.mean(dim=0))
        return self

    def delta_from_value(self, x: Tensor, mixed_value: Tensor) -> Tensor:
        """Compute a residual after an external bank has performed retrieval.

        ``mixed_value[..., rank]`` may broadcast over the input's leading token
        dimensions. Both inputs must already reside on the module device. This
        keeps CPU bank lookup independent of the GPU projection implementation,
        while preserving gradients if the caller supplies differentiable values.
        """
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected x[..., {self.in_features}], got {tuple(x.shape)}")
        if mixed_value.ndim < 1 or mixed_value.shape[-1] != self.rank:
            raise ValueError(f"Expected mixed_value[..., {self.rank}], got {tuple(mixed_value.shape)}")
        if x.device != self.b.device or mixed_value.device != x.device:
            raise ValueError("x, mixed_value, and VectorVeRA must be on the same device")
        try:
            value = torch.broadcast_to(mixed_value, (*x.shape[:-1], self.rank))
        except RuntimeError as error:
            raise ValueError("mixed_value must broadcast over x's token dimensions") from error
        projected = F.linear(x.to(dtype=self.A.dtype), self.A)
        return (F.linear(projected * value.to(dtype=projected.dtype), self.B) * self.b).to(dtype=x.dtype)

    def forward(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected x[..., {self.in_features}], got {tuple(x.shape)}")
        if keys.ndim != 2 or keys.shape[1] != self.key_dim:
            raise ValueError(f"Expected keys[N, {self.key_dim}], got {tuple(keys.shape)}")
        if values.ndim != 2 or values.shape != (keys.shape[0], self.rank):
            raise ValueError(f"Expected values[{keys.shape[0]}, {self.rank}], got {tuple(values.shape)}")
        if x.device != self.b.device or keys.device != x.device or values.device != x.device:
            raise ValueError("x, keys, values, and VectorVeRA must be on the same device")
        if keys.shape[0] == 0:
            residual = x.new_zeros((*x.shape[:-1], self.out_features))
            if not return_info:
                return residual
            selection_shape = (*x.shape[:-1], 0)
            return residual, {
                "indices": torch.empty(selection_shape, dtype=torch.long, device=x.device),
                "scores": torch.empty(selection_shape, dtype=self.b.dtype, device=x.device),
                "weights": torch.empty(selection_shape, dtype=self.b.dtype, device=x.device),
            }

        query = self.encode_query(x)
        normalized_keys = F.normalize(keys.to(dtype=query.dtype), dim=-1, eps=1e-12)
        similarities = query @ normalized_keys.T
        scores, indices = similarities.topk(min(self.top_k, keys.shape[0]), dim=-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        selected_values = values.to(dtype=query.dtype)[indices]
        retrieved = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
        residual = self.delta_from_value(x, retrieved)
        if return_info:
            return residual, {"indices": indices, "scores": scores, "weights": weights}
        return residual

    def contrastive_loss(
        self, query_inputs: Tensor, support_inputs: Tensor, temperature: float = 0.1
    ) -> Tensor:
        """Symmetric row-aligned query/key InfoNCE for offline address learning.

        Inputs have shape ``[N, in_features]`` and row i is the positive pair.
        Other rows are negatives, so a useful batch needs at least two distinct
        facts. Positives must be paired by training-only metadata; this function
        cannot validate data split or label-visibility policy.
        """
        if query_inputs.ndim != 2 or support_inputs.ndim != 2:
            raise ValueError("Contrastive inputs must both be [N, in_features]")
        if query_inputs.shape != support_inputs.shape or query_inputs.shape[0] == 0:
            raise ValueError("Contrastive inputs need matching nonempty shapes")
        query = self.encode_query(query_inputs)
        key = self.encode_key(support_inputs)
        logits = query @ key.T / _temperature(temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def trainable_numel(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def configuration(self) -> dict[str, Any]:
        """Constructor arguments to store beside ``state_dict`` checkpoints."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "key_dim": self.key_dim,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "seed": self.seed,
            "value_centering": self.value_centering,
        }
