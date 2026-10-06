"""Matched flat and fact-then-slot reads over three contextual content slots.

Both modes have identical trainable parameters and stored K/V bytes. Flat reads
top-4 slots; grouped selects the fact with largest logsumexp(slot cosine / T),
then softmax-mixes all three slots of that fact. This is NOT address/content
feature separation. Top-1 fact selection has no differentiable selection path:
CE trains selected slot scores/values, while dense group_address_loss supplies
address gradients outside the selected fact. All modes retain one rank-wise
VeRA gate. Exact CPU lookup scans keys, but mixes only the selected values.
"""
from __future__ import annotations

import copy
import hashlib
import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .reconstruction_vera import ReconstructionVeRA, GroupedVectorDB

READ_MODES = {"flat": 0, "grouped": 1}
READOUTS = {"vera": 0, "additive": 1}


def _mode(value):
    if value not in READ_MODES:
        raise ValueError("read_mode must be flat or grouped")
    return value


def _grouped_read(query, keys, values, temperature):
    """Tensor read; inputs have validated dimensions and complete fact groups."""
    leading = query.shape[:-1]
    if keys.shape[-2] == 0:
        empty = torch.empty((*leading, 0), device=query.device)
        return values.new_zeros((*leading, values.shape[-1])), dict(
            indices=empty.long(), scores=empty, weights=empty,
            selected_fact=torch.full(leading, -1, dtype=torch.long, device=query.device),
            group_score=query.new_zeros(leading))
    q = F.normalize(query.float(), dim=-1, eps=1e-12)
    k = F.normalize(keys.float(), dim=-1, eps=1e-12)
    similarities = q @ k.T if keys.ndim == 2 else torch.bmm(q, k.transpose(1, 2))
    fact_logits = torch.logsumexp(similarities.reshape(*leading, keys.shape[-2] // 3, 3) / temperature, dim=-1)
    group_score, fact = fact_logits.max(dim=-1)
    indices = fact.unsqueeze(-1) * 3 + torch.arange(3, device=query.device)
    scores = similarities.gather(-1, indices)
    weights = torch.softmax(scores / temperature, dim=-1)
    if values.ndim == 2:
        selected_values = values[indices]
    else:
        rows = torch.arange(values.shape[0], device=query.device)[:, None, None]
        selected_values = values[rows, indices]
    mixed = (selected_values.float() * weights.unsqueeze(-1)).sum(-2)
    return mixed, dict(indices=indices, scores=scores, weights=weights,
                       selected_fact=fact, group_score=group_score)


class QKVVeRA(ReconstructionVeRA):
    """Three-slot reconstruction with a matched learnable positional key term.

    Writer input is detached; Wk and slot_position remain differentiable. The
    positional term is added to the linear key BEFORE its final L2 norm.
    encode_key accepts [..., L, D] with L divisible by three, including grouped
    [..., 3, D] and flattened [..., N*3, D]. It never changes query encoding.
    Grouped route order is slot 0/1/2, not descending cosine order.
    """

    def __init__(self, in_features, out_features, rank=64, key_dim=64, top_k=4,
                 temperature=.2, seed=42, input_epsilon=1e-6, value_epsilon=1e-5,
                 *, slots=3, train_B=True, writer_mode="masked_mean", value_mlp_hidden=0,
                 architecture_version=1, reconstruction_version=1,
                 read_mode="flat", readout="vera", qkv_version=1):
        if type(slots) is not int or slots != 3 or type(qkv_version) is not int or qkv_version != 1:
            raise ValueError("QKV requires three slots and version 1")
        self.read_mode = _mode(read_mode)
        if readout not in READOUTS:
            raise ValueError("readout must be vera or additive")
        self.readout = readout
        if top_k != 4:
            raise ValueError("Matched QKV control requires flat top_k=4")
        super().__init__(in_features, out_features, rank, key_dim, top_k, temperature, seed,
                         input_epsilon, value_epsilon, slots=slots, train_B=train_B,
                         writer_mode=writer_mode, value_mlp_hidden=value_mlp_hidden,
                         architecture_version=architecture_version, reconstruction_version=reconstruction_version)
        self.slot_position = nn.Parameter(torch.zeros(3, key_dim, dtype=torch.float32))
        self.register_buffer("qkv_architecture", torch.tensor([1, READ_MODES[read_mode], READOUTS[readout]], dtype=torch.int64))

    def configuration(self):
        return dict(super().configuration(), read_mode=self.read_mode, readout=self.readout, qkv_version=1)

    def _validate_architecture(self, state_dict, prefix=""):
        super()._validate_architecture(state_dict, prefix)
        value = state_dict.get(prefix + "qkv_architecture")
        if (not isinstance(value, Tensor) or value.dtype != torch.int64 or value.shape != (3,)
                or value.detach().cpu().tolist() != [1, READ_MODES[self.read_mode], READOUTS[self.readout]]):
            raise RuntimeError("Checkpoint QKV read architecture differs")

    def delta_from_value(self, x, mixed_value):
        if self.readout == "vera":
            return super().delta_from_value(x, mixed_value)
        # Diagnostic only: A remains the identically initialized UNUSED buffer.
        # The main VeRA results always retain the multiplicative A x branch.
        if x.ndim < 1 or x.shape[-1] != self.in_features or mixed_value.ndim < 1 or mixed_value.shape[-1] != self.rank:
            raise ValueError("Wrong additive input/value feature dimensions")
        if x.device != self.b.device or mixed_value.device != x.device:
            raise ValueError("Input, value and parameters must share a device")
        try:
            mixed = torch.broadcast_to(mixed_value, (*x.shape[:-1], self.rank))
        except RuntimeError as error:
            raise ValueError("Value must broadcast over input positions") from error
        return (F.linear(mixed.to(self.B.dtype), self.B) * self.b).to(x.dtype)

    def encode_key(self, x):
        if x.ndim < 2 or x.shape[-1] != self.in_features or x.shape[-2] == 0 or x.shape[-2] % 3:
            raise ValueError("Slot keys require [..., L, D], with positive L divisible by three")
        encoded = self.Wk(self._domain_input(x.detach(), self.support_center))
        positions = torch.arange(x.shape[-2], device=encoded.device) % 3
        return F.normalize(encoded + self.slot_position[positions], dim=-1, eps=1e-12)

    def forward(self, x, keys=None, values=None, return_info=False):
        if self.read_mode == "flat":
            return super().forward(x, keys, values, return_info=return_info)
        if keys is None and values is None:
            keys, values = self.encoded_feature_bank()
        if (keys is None or values is None or keys.ndim not in (2, 3)
                or keys.shape[-1] != self.key_dim or keys.shape[-2] % 3
                or values.shape != (*keys.shape[:-1], self.rank)):
            raise ValueError("Tensor banks must contain matching complete three-slot groups")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("Wrong actual layer-input feature width")
        if keys.ndim == 3 and (x.ndim != 3 or x.shape[0] != keys.shape[0]):
            raise ValueError("Independent banks require matching x[batch, sequence, D]")
        if x.device != self.b.device or keys.device != x.device or values.device != x.device:
            raise ValueError("Input, parameters and tensor banks must share a device")
        # Like the old empty-bank path, do not require fitted query statistics.
        query = self.encode_query(x) if keys.shape[-2] else x.new_zeros((*x.shape[:-1], self.key_dim))
        mixed, info = _grouped_read(query, keys, values, self.temperature)
        residual = self.delta_from_value(x, mixed)
        info.update(fact_indices=info["indices"] // 3, slot_indices=info["indices"] % 3)
        return (residual, info) if return_info else residual

    def new_store(self, *, record_routes=False):
        return QKVVectorDB(self.key_dim, self.rank, self.temperature, self.top_k,
                           slots=3, read_mode=self.read_mode, record_routes=record_routes)

    def _check_store(self, store):
        super()._check_store(store)
        if not isinstance(store, QKVVectorDB) or store.read_mode != self.read_mode:
            raise ValueError("CPU store read mode differs from module")


class QKVVectorDB(GroupedVectorDB):
    """CPU-only, atomic fact replacement with a snapshot-bound read policy.

    The inherited immutable (bank, fact_ids) tuple is captured once per search.
    Keys/values are the same flattened three-slot payload in both read modes.
    This wrapper accesses the captured PersistentVectorDB tensor storage to
    gather selected values without copying the full value bank on each read.
    """

    PROTOCOL = "qkv-grouped-vdb-v1"

    def __init__(self, key_dim, value_dim, temperature=.2, top_k=4, *, slots=3,
                 read_mode="flat", record_routes=False):
        if type(slots) is not int or slots != 3 or top_k != 4:
            raise ValueError("QKV store requires three slots and top_k=4")
        self.read_mode = _mode(read_mode)
        super().__init__(key_dim, value_dim, temperature, top_k, slots=slots, record_routes=record_routes)

    def search(self, queries):
        if self.read_mode == "flat":
            return super().search(queries)
        bank, facts = self._state
        query = torch.as_tensor(queries).detach().to(device="cpu", dtype=torch.float32)
        if query.ndim < 1 or query.shape[-1] != self.key_dim or not bool(torch.isfinite(query).all()):
            raise ValueError("CPU queries must be finite [..., key_dim]")
        if len(bank) and query.numel():
            norms = query.norm(dim=-1)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
                raise ValueError("Nonempty-bank queries must have finite nonzero norms")
        with torch.no_grad():
            mixed, info = _grouped_read(query, bank._keys, bank._values, bank.temperature)
        info["mixed_value"] = mixed
        width = info["indices"].shape[-1]
        count = query.reshape(-1, self.key_dim).shape[0]
        flat = info["indices"].reshape(count, width).tolist()
        info["ids"] = [[bank.ids[i] for i in row] for row in flat]
        info["fact_ids"] = [[facts[i // 3] for i in row] for row in flat]
        info["slot_ids"] = info["indices"] % 3
        if self.record_routes:
            self.route_log.append(dict(query_shape=list(query.shape), **{
                k:info[k].clone() for k in ("indices", "scores", "weights", "slot_ids", "selected_fact", "group_score")},
                fact_ids=copy.deepcopy(info["fact_ids"])))
        return info

    def snapshot(self):
        return dict(super().snapshot(), read_mode=self.read_mode)

    def hash(self):
        metadata = json.dumps(dict(read_mode=self.read_mode), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((super().hash() + metadata).encode()).hexdigest()

    @classmethod
    def from_snapshot(cls, state, *, record_routes=False):
        if (not isinstance(state, dict) or set(state) != {"protocol", "slots", "fact_ids", "store", "read_mode"}
                or state["protocol"] != cls.PROTOCOL):
            raise ValueError("Invalid QKV store snapshot")
        mode = _mode(state["read_mode"])
        legacy = {k:v for k,v in state.items() if k != "read_mode"}
        legacy["protocol"] = GroupedVectorDB.PROTOCOL
        base = GroupedVectorDB.from_snapshot(legacy)
        result = cls(**state["store"]["config"], slots=state["slots"], read_mode=mode, record_routes=record_routes)
        result._state = base._state
        return result
