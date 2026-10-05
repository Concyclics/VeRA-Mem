"""Centered addressing and nonsaturating values for the VeRA memory branch.

The computation and retrieval contract are inherited from ``VectorVeRA``:
every actual layer input generates a query, sparse retrieved values become
rank-wise VeRA parameters, and observation inputs generate writable key/value
pairs. This class does not own or update a VDB.

The statistics must be fitted explicitly on OFFLINE TRAINING features before
encoding. Query and observation distributions have separate fixed centers.
Fitting never optimizes model parameters; encoding never changes statistics.
The caller remains responsible for ensuring that supplied features belong to
the training split. No dataset membership can be inferred from a tensor.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .vector_vera import VectorVeRA


class StableVectorVeRA(VectorVeRA):
    """Use centered, renormalized inputs and RMS-normalized linear values.

    For query/support domain ``d`` with its training center ``mu_d``::

        u = RMSnormalize(x)
        z_d = RMSnormalize(u - mu_d)
        q = L2normalize(Wq z_query)
        k = L2normalize(Wk z_support)
        v = RMSnormalize(Wv z_support)

    Unlike a tanh writer, this value encoder has no elementwise saturation.
    Its per-vector RMS is bounded by one, rather than each coordinate being
    bounded by one. A collapsed or uninformative writer is still possible and
    must be diagnosed; the normalization is not a claim of useful memory.

    ``A``, ``B``, and the computation ``b * B((A x) * retrieved_value)`` remain
    exactly the original mechanism. The residual uses the actual unnormalized
    layer input, and zero-initialized ``b`` preserves an exact initial no-op.
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
        input_epsilon: float = 1e-6,
        value_epsilon: float = 1e-5,
    ) -> None:
        for name, value in (("input_epsilon", input_epsilon), ("value_epsilon", value_epsilon)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(
            in_features=in_features, out_features=out_features, rank=rank,
            key_dim=key_dim, top_k=top_k, temperature=temperature, seed=seed,
            value_centering=False,
        )
        self.input_epsilon = float(input_epsilon)
        self.value_epsilon = float(value_epsilon)
        self.register_buffer("query_center", torch.zeros(in_features, dtype=torch.float32))
        self.register_buffer("support_center", torch.zeros(in_features, dtype=torch.float32))
        self.register_buffer("statistics_fitted", torch.tensor(False))
        # A Python flag avoids device synchronization on every token. The
        # persistent boolean buffer restores this flag when loading state.
        self._statistics_ready = False

    @staticmethod
    def _rms_normalize(value: Tensor, epsilon: float) -> Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)

    def _domain_input(self, x: Tensor, center: Tensor) -> Tensor:
        if not self._statistics_ready:
            raise RuntimeError("Call fit_statistics on offline training features before encoding")
        # Keep the original first normalization so statistics are independent
        # of positive rescaling of the raw backbone activations.
        normalized = self._normalized_input(x)
        return self._rms_normalize(normalized - center, self.input_epsilon)

    def encode_query(self, x: Tensor) -> Tensor:
        return F.normalize(self.Wq(self._domain_input(x, self.query_center)), dim=-1, eps=1e-12)

    def encode_key(self, x: Tensor) -> Tensor:
        return F.normalize(self.Wk(self._domain_input(x, self.support_center)), dim=-1, eps=1e-12)

    def encode_value(self, x: Tensor) -> Tensor:
        projected = self.Wv(self._domain_input(x, self.support_center))
        return self._rms_normalize(projected, self.value_epsilon)

    @torch.no_grad()
    def fit_statistics(self, query_inputs: Tensor, support_inputs: Tensor) -> "StableVectorVeRA":
        """Fit once from training queries and observations, with no online updates.

        Arrays are independently nonempty ``[N, in_features]``; their lengths
        need not agree. Query and support means are not pooled across domains.
        Both arrays are validated before either buffer is changed. Re-fitting
        a trained/loaded module is prohibited: create a new experimental module
        if different training statistics are required.
        """
        if self._statistics_ready:
            raise RuntimeError("Statistics are already fitted and immutable; construct a new module to refit")
        centers = []
        for name, inputs in (("query_inputs", query_inputs), ("support_inputs", support_inputs)):
            if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] != self.in_features:
                raise ValueError(f"{name} must be nonempty [N, {self.in_features}]")
            inputs = inputs.to(device=self.query_center.device, dtype=self.Wq.weight.dtype)
            if not torch.isfinite(inputs).all():
                raise ValueError(f"{name} must contain finite values")
            normalized = self._normalized_input(inputs)
            if not torch.isfinite(normalized).all():
                raise ValueError(f"{name} normalization produced nonfinite values")
            centers.append(normalized.mean(dim=0))
        self.query_center.copy_(centers[0])
        self.support_center.copy_(centers[1])
        self.statistics_fitted.fill_(True)
        self._statistics_ready = True
        return self

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )
        self._statistics_ready = bool(self.statistics_fitted.item())

    def configuration(self) -> dict[str, Any]:
        result = super().configuration()
        result.pop("value_centering")
        result.update(input_epsilon=self.input_epsilon, value_epsilon=self.value_epsilon)
        return result
