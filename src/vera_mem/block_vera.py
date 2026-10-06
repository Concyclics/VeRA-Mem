"""Whole-fact values as a residual dynamic low-rank operator.

All modes choose one fact by QK logsumexp and fetch its complete three rows.
They use uniform row weights, not the previous QKV within-fact softmax. With
z=A x and m=mean(V), every mode retains diag(m) z. The two matched outer arms
add either w(m) u(m)^T z or mean_s w(v_s) u(v_s)^T z, where u(v) is the L2
normalized affine input factor and w(v) is the linear output factor. Their
extra terms have rank at most one/three; the TOTAL operator retains rank up
to the common latent width. It is not a higher-rank replacement for B or A.

Uniform values and discrete group selection provide no CE gradient to Q/K.
The inherited dense group-address loss is the explicit Q/K training route.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .qkv_vera import QKVVeRA, QKVVectorDB

BLOCK_MODES = {"diagonal": 0, "pooled_outer": 1, "block_outer": 2}


def _retrieve_block(query, keys, values, temperature):
    """Select by keys but preserve all selected V rows and their gradients."""
    leading = query.shape[:-1]
    if keys.shape[-2] == 0:
        empty = query.new_empty((*leading, 0))
        return dict(indices=empty.long(), scores=empty, weights=empty,
                    selected_values=values.new_empty((*leading, 0, values.shape[-1])),
                    selected_fact=torch.full(leading, -1, dtype=torch.long, device=query.device),
                    group_score=query.new_zeros(leading), queries=query)
    q = F.normalize(query.float(), dim=-1, eps=1e-12)
    k = F.normalize(keys.float(), dim=-1, eps=1e-12)
    similarities = q @ k.T if keys.ndim == 2 else torch.bmm(q, k.transpose(1, 2))
    group_logits = torch.logsumexp(similarities.reshape(*leading, keys.shape[-2] // 3, 3) / temperature, -1)
    group_score, fact = group_logits.max(-1)
    indices = 3 * fact.unsqueeze(-1) + torch.arange(3, device=query.device)
    scores = similarities.gather(-1, indices)
    if values.ndim == 2:
        selected = values[indices]
    else:
        rows = torch.arange(values.shape[0], device=query.device)[:, None, None]
        selected = values[rows, indices]
    return dict(indices=indices, scores=scores, weights=torch.full_like(scores, 1 / 3),
                selected_values=selected, selected_fact=fact, group_score=group_score, queries=query)


class BlockVeRA(QKVVeRA):
    """One shared VeRA projection with optional complete-block outer residual.

    Inputs are actual layer activations. Raw writer features remain detached,
    while the shared value writer and selected rows are differentiable offline.
    CPU online use requires BlockBackend; the old mixed-value hook cannot carry
    a matrix and is explicitly rejected for either outer mode.
    """

    def __init__(self, *args, block_mode="diagonal", slot_weighting="uniform",
                 block_version=1, factor_epsilon=1e-6, residual_outer=True, **kwargs):
        if (block_mode not in BLOCK_MODES or slot_weighting != "uniform"
                or type(block_version) is not int or block_version != 1 or residual_outer is not True):
            raise ValueError("Require declared block mode, uniform weights, residual outer and version 1")
        if isinstance(factor_epsilon, bool) or not math.isfinite(factor_epsilon) or factor_epsilon <= 0:
            raise ValueError("factor_epsilon must be finite and positive")
        kwargs.setdefault("read_mode", "grouped"); kwargs.setdefault("readout", "vera")
        if kwargs["read_mode"] != "grouped" or kwargs["readout"] != "vera":
            raise ValueError("Block readers require grouped retrieval and the shared VeRA A/B path")
        super().__init__(*args, **kwargs)
        self.block_mode, self.slot_weighting = block_mode, slot_weighting
        self.block_version, self.factor_epsilon, self.residual_outer = 1, float(factor_epsilon), True
        self.P_in = self.P_out = None
        if block_mode != "diagonal":
            # nn.Linear initialization must not advance the caller's RNG, even
            # though these random initial weights are immediately overwritten.
            with torch.random.fork_rng(devices=[]):
                self.P_in = nn.Linear(self.rank, self.rank, bias=True)
                self.P_out = nn.Linear(self.rank, self.rank, bias=False)
            generator = torch.Generator(device="cpu").manual_seed(self.seed + 104729)
            bias = torch.randn(self.rank, generator=generator)
            bias = bias / bias.square().mean().sqrt()
            with torch.no_grad():
                self.P_in.weight.copy_(torch.eye(self.rank)); self.P_out.weight.copy_(torch.eye(self.rank))
                self.P_in.bias.copy_(bias)
        self.register_buffer("block_architecture", torch.tensor([1, BLOCK_MODES[block_mode], 1], dtype=torch.int64))

    def configuration(self):
        return dict(super().configuration(), block_mode=self.block_mode, slot_weighting=self.slot_weighting,
                    block_version=self.block_version, factor_epsilon=self.factor_epsilon, residual_outer=True)

    def _validate_architecture(self, state_dict, prefix=""):
        super()._validate_architecture(state_dict, prefix)
        value = state_dict.get(prefix + "block_architecture")
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.shape != (3,)
                or value.detach().cpu().tolist() != [1, BLOCK_MODES[self.block_mode], 1]):
            raise RuntimeError("Checkpoint block architecture differs")

    def block_factors(self, values):
        """Row-wise u,w; pooled callers must supply their mean as one row."""
        if self.P_in is None or self.P_out is None:
            raise ValueError("Diagonal control has no outer factors")
        if values.ndim < 1 or values.shape[-1] != self.rank or values.device != self.b.device:
            raise ValueError("Outer factors require [..., rank] values on module device")
        values = values.to(self.P_in.weight.dtype)
        return F.normalize(self.P_in(values), dim=-1, eps=self.factor_epsilon), self.P_out(values)

    def latent_from_block(self, x, selected_values):
        """Return h before the common B/b; matrix rows are never mean-only state."""
        if (x.ndim < 1 or x.shape[-1] != self.in_features or selected_values.ndim < 2
                or selected_values.shape[-1] != self.rank or selected_values.shape[-2] not in (0, 3)
                or x.device != self.b.device or selected_values.device != x.device):
            raise ValueError("Require actual x[..., D] and zero/three selected rows [..., S, rank] on module device")
        try:
            rows = torch.broadcast_to(selected_values, (*x.shape[:-1], selected_values.shape[-2], self.rank))
        except RuntimeError as error:
            raise ValueError("Selected block must broadcast over actual query positions") from error
        z = F.linear(x.to(self.A.dtype), self.A)
        if rows.shape[-2] == 0:
            return z * 0
        rows = rows.to(z.dtype); mean = rows.mean(-2)
        diagonal = z * mean
        if self.block_mode == "diagonal":
            return diagonal
        factors = mean.unsqueeze(-2) if self.block_mode == "pooled_outer" else rows
        u, w = self.block_factors(factors)
        coefficients = (u * z.unsqueeze(-2)).sum(-1)
        outer = (w * coefficients.unsqueeze(-1)).mean(-2)
        return diagonal + outer

    def delta_from_block(self, x, selected_values):
        return (F.linear(self.latent_from_block(x, selected_values), self.B) * self.b).to(x.dtype)

    def delta_from_value(self, x, mixed_value):
        if self.block_mode != "diagonal":
            raise RuntimeError("Outer readers require full selected_values; use BlockBackend/delta_from_block")
        return super().delta_from_value(x, mixed_value)

    def forward(self, x, keys=None, values=None, return_info=False):
        if keys is None and values is None:
            keys, values = self.encoded_feature_bank()
        if (keys is None or values is None or keys.ndim not in (2, 3) or keys.shape[-1] != self.key_dim
                or keys.shape[-2] % 3 or values.shape != (*keys.shape[:-1], self.rank)):
            raise ValueError("Tensor banks must contain matching complete three-slot groups")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("Wrong actual layer-input feature width")
        if keys.ndim == 3 and (x.ndim != 3 or x.shape[0] != keys.shape[0]):
            raise ValueError("Independent banks require matching x[batch, sequence, D]")
        if x.device != self.b.device or keys.device != x.device or values.device != x.device:
            raise ValueError("Input, parameters and tensor banks must share a device")
        query = self.encode_query(x) if keys.shape[-2] else x.new_zeros((*x.shape[:-1], self.key_dim))
        info = _retrieve_block(query, keys, values, self.temperature)
        info.update(fact_indices=info["indices"] // 3, slot_ids=info["indices"] % 3)
        delta = self.delta_from_block(x, info["selected_values"])
        return (delta, info) if return_info else delta

    def new_store(self, *, record_routes=False):
        return BlockVectorDB(self.key_dim, self.rank, self.temperature, self.top_k, record_routes=record_routes)

    def _check_store(self, store):
        super()._check_store(store)
        if not isinstance(store, BlockVectorDB) or store.slot_weighting != self.slot_weighting:
            raise ValueError("CPU block store configuration differs")

    @torch.no_grad()
    def cpu_delta(self, x, store, *, teacher=False, return_info=False):
        if teacher:
            delta, info = x.new_zeros((*x.shape[:-1], self.out_features)), {"teacher_bypass": True}
        else:
            self._check_store(store)
            query = self.encode_query(x).detach().cpu() if len(store) else x.new_zeros((*x.shape[:-1], self.key_dim), device="cpu")
            info = store.search(query)
            delta = self.delta_from_block(x, info["selected_values"].to(x.device))
        return (delta, info) if return_info else delta


class BlockVectorDB(QKVVectorDB):
    """CPU whole-group atomic writes and full block reads with replay queries."""
    PROTOCOL = "block-grouped-vdb-v1"

    def __init__(self, key_dim, value_dim, temperature=.2, top_k=4, *, slots=3,
                 read_mode="grouped", slot_weighting="uniform", block_version=1, record_routes=False):
        if read_mode != "grouped" or slot_weighting != "uniform" or type(block_version) is not int or block_version != 1:
            raise ValueError("Block store requires grouped selection, uniform weights, version 1")
        super().__init__(key_dim, value_dim, temperature, top_k, slots=slots, read_mode=read_mode, record_routes=record_routes)
        self.slot_weighting, self.block_version = "uniform", 1

    def search(self, queries):
        bank, facts = self._state
        query = torch.as_tensor(queries).detach().to(device="cpu", dtype=torch.float32)
        if query.ndim < 1 or query.shape[-1] != self.key_dim or not bool(torch.isfinite(query).all()):
            raise ValueError("CPU queries must be finite [..., key_dim]")
        if len(bank) and query.numel():
            norms = query.norm(dim=-1)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
                raise ValueError("Nonempty-bank queries must have finite nonzero norms")
        with torch.no_grad():
            info = _retrieve_block(query, bank._keys, bank._values, bank.temperature)
        width = info["indices"].shape[-1]; count = query.reshape(-1, self.key_dim).shape[0]
        indices = info["indices"].reshape(count, width).tolist()
        info["ids"] = [[bank.ids[i] for i in row] for row in indices]
        info["fact_ids"] = [[facts[i // 3] for i in row] for row in indices]
        info["slot_ids"] = info["indices"] % 3
        if self.record_routes:
            self.route_log.append(dict(query_shape=list(query.shape), **{
                k: v.clone() for k, v in info.items() if isinstance(v, torch.Tensor)},
                fact_ids=copy.deepcopy(info["fact_ids"])))
        return info

    def snapshot(self):
        return dict(super().snapshot(), slot_weighting="uniform", block_version=1)

    def hash(self):
        metadata = json.dumps(dict(slot_weighting="uniform", block_version=1), sort_keys=True)
        return hashlib.sha256((super().hash() + metadata).encode()).hexdigest()

    @classmethod
    def from_snapshot(cls, state, *, record_routes=False):
        fields = {"protocol", "slots", "fact_ids", "store", "read_mode", "slot_weighting", "block_version"}
        if (not isinstance(state, dict) or set(state) != fields or state["protocol"] != cls.PROTOCOL
                or state["read_mode"] != "grouped" or state["slot_weighting"] != "uniform"
                or type(state["block_version"]) is not int or state["block_version"] != 1):
            raise ValueError("Invalid block-store snapshot")
        legacy = {k: v for k, v in state.items() if k not in ("slot_weighting", "block_version")}
        legacy["protocol"] = QKVVectorDB.PROTOCOL
        base = QKVVectorDB.from_snapshot(legacy)
        result = cls(**state["store"]["config"], slots=state["slots"], record_routes=record_routes)
        result._state = base._state
        return result
