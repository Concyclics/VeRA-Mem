"""Content reconstruction through sparse, grouped VDB values and one VeRA layer.

The two independent controls are one/three content slots and fixed/trainable
shared B. Both use the same Wk/Wv, rank, top-k and actual-input query encoder.
There are no foundation prototypes, slot offsets, answer-ID query filters, or
text retrieval. Three slots increase stored payload and change writer pooling;
they do not increase the rank of the shared readout. A remains fixed.

The tensor bank is the differentiable TRAIN path. The grouped CPU store is the
detached online path. Existing CounterfactualBackend teacher scopes disable
the entire vector branch, so teacher calls neither encode nor read these banks.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from .interface_variants import InterfaceVectorVeRA, masked_mean_support
from .vector_store import PersistentVectorDB


def _slots(value):
    if type(value) is not int or value not in (1, 3):
        raise ValueError("slots must be 1 or 3")
    return value


def _binary(mask, shape, device, name):
    if mask.shape != shape or mask.device != device:
        raise ValueError(f"{name} shape/device mismatch")
    if mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be boolean or binary integer")
    if bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError(f"{name} must contain only zero and one")
    return mask.bool()


def pool_content_slots(token_features: Tensor, content_mask: Tensor, *, slots: int = 1,
                       slot_masks: Tensor | None = None) -> Tensor:
    """Pool explicit content only, returning ``[..., slots, D]`` in float32.

    ``slot_masks[..., slots, tokens]`` must partition content exactly. Supply
    tokenizer-derived three-word spans for the word-slot experiment. Without
    masks, three slots split selected token positions into consecutive thirds
    (sizes differ by at most one); this is a different, explicit pooling choice.
    The one-slot mean weights tokens equally; it is generally NOT the unweighted
    mean of three word-slot means when words have unequal token counts.
    """
    slots = _slots(slots)
    # Reuse the established finite-selected-feature/binary-mask contract.
    whole = masked_mean_support(token_features, content_mask)
    content = content_mask.bool()
    if slot_masks is None:
        if slots == 1:
            return whole.unsqueeze(-2)
        counts = content.sum(-1)
        if bool((counts < slots).any()):
            raise ValueError("At least three content tokens are required for thirds")
        ordinal = content.long().cumsum(-1) - 1
        # torch.tensor_split convention: distribute the remainder to early slots.
        first = counts // 3 + (counts % 3 > 0)
        second = first + counts // 3 + (counts % 3 > 1)
        slot_masks = torch.stack((content & (ordinal < first[..., None]),
                                 content & (ordinal >= first[..., None]) & (ordinal < second[..., None]),
                                 content & (ordinal >= second[..., None])), dim=-2)
    else:
        slot_masks = _binary(slot_masks, (*content.shape[:-1], slots, content.shape[-1]),
                             token_features.device, "slot_masks")
        if not torch.equal(slot_masks.long().sum(-2), content.long()):
            raise ValueError("slot_masks must partition exactly the content mask without overlap")
    expanded = token_features.unsqueeze(-3).expand(*token_features.shape[:-2], slots,
                                                  *token_features.shape[-2:])
    return masked_mean_support(expanded, slot_masks)


class ReconstructionVeRA(InterfaceVectorVeRA):
    """Shared content writer with independent writer-slot/readout controls.

    ``set_feature_bank`` takes flattened [N*S,D] or [B,N*S,D] raw features.
    Subsequent ``forward(x, None, None)`` re-encodes them on EVERY call, retaining
    Wk/Wv gradients without retaining a stale optimization graph. Raw writer
    features are detached; query inputs preserve their actual layer graph.
    The bank is transient, absent from state_dict, and must be reset after moving
    a module to another device. Explicit K/V banks remain backend-compatible.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 64,
                 key_dim: int = 64, top_k: int = 4, temperature: float = .2,
                 seed: int = 42, input_epsilon: float = 1e-6, value_epsilon: float = 1e-5,
                 *, slots: int = 1, train_B: bool = False,
                 writer_mode: str = "masked_mean", value_mlp_hidden: int = 0,
                 architecture_version: int = 1, reconstruction_version: int = 1):
        self.slots = _slots(slots)
        if (writer_mode != "masked_mean" or type(value_mlp_hidden) is not int or value_mlp_hidden != 0
                or type(reconstruction_version) is not int or reconstruction_version != 1):
            raise ValueError("Reconstruction requires pooled linear writer and version 1")
        super().__init__(in_features, out_features, rank, key_dim, top_k, temperature,
                         seed, input_epsilon, value_epsilon, writer_mode=writer_mode,
                         train_B=train_B, value_mlp_hidden=0,
                         architecture_version=architecture_version)
        self.reconstruction_version = reconstruction_version
        self.register_buffer("reconstruction_architecture", torch.tensor([1, self.slots], dtype=torch.int64))
        self._feature_bank: tuple[Tensor, Tensor] | None = None

    def _validate_architecture(self, state_dict, prefix=""):
        super()._validate_architecture(state_dict, prefix)
        value = state_dict.get(prefix + "reconstruction_architecture")
        if (not isinstance(value, Tensor) or value.dtype != torch.int64 or value.shape != (2,)
                or value.detach().cpu().tolist() != [1, self.slots]):
            raise RuntimeError("Checkpoint reconstruction architecture differs")

    def configuration(self):
        return dict(super().configuration(), slots=self.slots,
                    reconstruction_version=self.reconstruction_version)

    def encode_key(self, x: Tensor) -> Tensor:
        return super().encode_key(x.detach())

    def encode_value(self, x: Tensor) -> Tensor:
        return super().encode_value(x.detach())

    def encode_bank(self, key_features: Tensor, value_features: Tensor | None = None):
        """Encode grouped [N,S,D] or [B,N,S,D] into flattened K/V banks."""
        value_features = key_features if value_features is None else value_features
        if (key_features.ndim not in (3, 4) or key_features.shape[-2:] != (self.slots, self.in_features)
                or key_features.shape[-3] == 0 or value_features.shape != key_features.shape):
            raise ValueError("Grouped features must be matching [N,S,D] or [B,N,S,D]")
        if any(not feature.is_floating_point() or not bool(torch.isfinite(feature).all())
               for feature in (key_features, value_features)):
            raise ValueError("Writer features must be finite floating point")
        flat_shape = (*key_features.shape[:-3], key_features.shape[-3] * self.slots, self.in_features)
        return self.encode_key(key_features.reshape(flat_shape)), self.encode_value(value_features.reshape(flat_shape))

    def set_feature_bank(self, key_features: Tensor, value_features: Tensor | None = None):
        """Replace the transient TRAIN bank, copying detached feature tensors."""
        value_features = key_features if value_features is None else value_features
        if (key_features.ndim not in (2, 3) or key_features.shape[-1] != self.in_features
                or key_features.shape[-2] % self.slots or value_features.shape != key_features.shape):
            raise ValueError("Flattened features must be matching [N*S,D] or [B,N*S,D]")
        copies = []
        for feature in (key_features, value_features):
            if not feature.is_floating_point() or not bool(torch.isfinite(feature).all()):
                raise ValueError("Writer features must be finite floating point")
            copies.append(feature.detach().to(device=self.b.device, dtype=self.Wk.weight.dtype).clone())
        self._feature_bank = tuple(copies)
        return self

    def clear_feature_bank(self):
        self._feature_bank = None

    def encoded_feature_bank(self):
        if self._feature_bank is None:
            raise RuntimeError("Set a feature bank or supply explicit tensor K/V")
        keys, values = self._feature_bank
        return self.encode_key(keys), self.encode_value(values)

    def forward(self, x: Tensor, keys: Tensor | None = None, values: Tensor | None = None,
                return_info: bool = False):
        if keys is None and values is None:
            keys, values = self.encoded_feature_bank()
        if keys is None or values is None or keys.ndim not in (2, 3) or keys.shape[-2] % self.slots:
            raise ValueError("Supply complete fact groups in both tensor banks")
        result, info = super().forward(x, keys, values, return_info=True)
        info = dict(info, fact_indices=info["indices"] // self.slots,
                    slot_indices=info["indices"] % self.slots)
        return (result, info) if return_info else result

    def group_address_loss(self, query_inputs: Tensor, target_fact_indices: Tensor,
                           keys: Tensor | None = None, *, batch_indices: Tensor | None = None,
                           reduction: str = "mean") -> Tensor:
        """Dense address supervision: -log(sum probability of a fact's slots).

        This auxiliary objective is deliberately dense; actual VeRA reads stay
        top-k sparse. For flattened answer-position inputs with independent
        [B,N*S,K] banks, provide one batch_indices entry per prediction token.
        target_fact_indices are local FACT indices, never flattened slot IDs.
        """
        if keys is None:
            if self._feature_bank is None:
                raise RuntimeError("No feature bank for group address supervision")
            keys = self.encode_key(self._feature_bank[0])
        if keys.ndim not in (2, 3) or keys.shape[-1] != self.key_dim or keys.shape[-2] == 0 or keys.shape[-2] % self.slots:
            raise ValueError("Address bank must contain nonempty complete fact groups")
        query = self.encode_query(query_inputs)
        normalized = torch.nn.functional.normalize(keys.float(), dim=-1, eps=1e-12)
        if keys.ndim == 2:
            if batch_indices is not None:
                raise ValueError("batch_indices are only used with independent banks")
            logits = query @ normalized.T
        elif query.ndim == 3 and batch_indices is None and query.shape[0] == keys.shape[0]:
            logits = torch.bmm(query, normalized.transpose(1, 2))
        elif query.ndim == 2 and batch_indices is not None:
            if (batch_indices.shape != query.shape[:-1] or batch_indices.dtype != torch.long
                    or bool(((batch_indices < 0) | (batch_indices >= keys.shape[0])).any())):
                raise ValueError("Invalid per-token batch_indices")
            logits = torch.einsum("tk,tnk->tn", query, normalized[batch_indices])
        else:
            raise ValueError("Independent address banks need matching batch or per-token batch_indices")
        targets = torch.as_tensor(target_fact_indices, device=query.device)
        if targets.dtype != torch.long:
            raise ValueError("target_fact_indices must be int64")
        try:
            targets = torch.broadcast_to(targets, logits.shape[:-1])
        except RuntimeError as error:
            raise ValueError("target fact indices must broadcast over query positions") from error
        if bool(((targets < 0) | (targets >= keys.shape[-2] // self.slots)).any()):
            raise ValueError("Target fact outside bank")
        selected = targets[..., None] * self.slots + torch.arange(self.slots, device=query.device)
        logits = logits.float() / self.temperature
        loss = torch.logsumexp(logits, -1) - torch.logsumexp(logits.gather(-1, selected), -1)
        if reduction == "none":
            return loss
        if reduction == "mean" and loss.numel():
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError("reduction must be none, mean (nonempty), or sum")

    def new_store(self, *, record_routes: bool = False):
        return GroupedVectorDB(self.key_dim, self.rank, self.temperature, self.top_k,
                               slots=self.slots, record_routes=record_routes)

    @torch.no_grad()
    def write_group(self, store, fact_id: str, key_features: Tensor,
                    value_features: Tensor | None = None, *, timestamp=0):
        self._check_store(store)
        if key_features.shape != (self.slots, self.in_features):
            raise ValueError("One fact must supply [S,D] features")
        values = key_features if value_features is None else value_features
        keys, values = self.encode_bank(key_features.detach().to(self.b.device).unsqueeze(0),
                                       values.detach().to(self.b.device).unsqueeze(0))
        store.write_group(fact_id, keys, values, timestamp)

    def _check_store(self, store):
        if (not isinstance(store, GroupedVectorDB) or store.slots != self.slots
                or (store.key_dim, store.value_dim, store.top_k, store.temperature)
                != (self.key_dim, self.rank, self.top_k, self.temperature)):
            raise ValueError("Grouped store configuration differs from module")

    @torch.no_grad()
    def cpu_delta(self, x: Tensor, store, *, teacher: bool = False, return_info: bool = False):
        """Online detached CPU search, or strict zero-access teacher bypass."""
        if teacher:
            delta = x.new_zeros((*x.shape[:-1], self.out_features))
            info = {"teacher_bypass": True}
        else:
            self._check_store(store)
            info = store.search(self.encode_query(x).detach().cpu())
            delta = self.delta_from_value(x, info["mixed_value"].to(x.device))
        return (delta, info) if return_info else delta


class GroupedVectorDB:
    """Atomic whole-fact replacement over the existing exact CPU VDB.

    A fact always owns S consecutive slot records. write_group validates a
    private copy and publishes one (store, fact_ids) state, including timestamps.
    A failed or stale write changes nothing. Readers capture one fixed state;
    concurrent writers still require caller synchronization. Search never
    accepts a target fact ID. Route logs are optional, detached diagnostics.
    """

    PROTOCOL = "reconstruction-grouped-vdb-v1"

    def __init__(self, key_dim: int, value_dim: int, temperature: float = .2,
                 top_k: int = 4, *, slots: int = 1, record_routes: bool = False):
        self.slots = _slots(slots)
        self._state = (PersistentVectorDB(key_dim, value_dim, temperature, top_k), ())
        self.record_routes = bool(record_routes)
        self.route_log: list[dict] = []

    @staticmethod
    def slot_id(fact_id, slot):
        return json.dumps([fact_id, slot], ensure_ascii=False, separators=(",", ":"))

    def __len__(self):
        return len(self._state[0])

    @property
    def fact_ids(self):
        return self._state[1]

    @property
    def record_fact_ids(self):
        return tuple(fact for fact in self.fact_ids for _ in range(self.slots))

    @property
    def record_slot_ids(self):
        return tuple(slot for _ in self.fact_ids for slot in range(self.slots))

    def __getattr__(self, name):
        if name in {"key_dim", "value_dim", "temperature", "top_k", "keys", "values", "ids", "timestamps"}:
            return getattr(self._state[0], name)
        raise AttributeError(name)

    def write_group(self, fact_id: str, keys: Tensor, values: Tensor, timestamp):
        if not isinstance(fact_id, str) or not fact_id:
            raise ValueError("fact_id must be a nonempty string")
        base, facts = self._state
        if keys.shape != (self.slots, base.key_dim) or values.shape != (self.slots, base.value_dim):
            raise ValueError("Whole-fact writes require exactly S key/value slots")
        candidate = copy.deepcopy(base)
        for slot in range(self.slots):
            candidate.write(self.slot_id(fact_id, slot), keys[slot], values[slot], timestamp)
        self._state = (candidate, facts if fact_id in facts else (*facts, fact_id))

    def search(self, queries: Tensor):
        bank, facts = self._state
        info = bank.search(queries)
        count, width = len(info["ids"]), info["indices"].shape[-1]
        flat = info["indices"].reshape(count, width)
        info["fact_ids"] = [[facts[index // self.slots] for index in row] for row in flat.tolist()]
        info["slot_ids"] = info["indices"] % self.slots
        if self.record_routes:
            self.route_log.append(dict(query_shape=list(queries.shape), indices=info["indices"].clone(),
                weights=info["weights"].clone(), scores=info["scores"].clone(),
                fact_ids=copy.deepcopy(info["fact_ids"]), slot_ids=info["slot_ids"].clone()))
        return info

    def clear_routes(self):
        self.route_log.clear()

    def resident_bytes(self):
        return self._state[0].resident_bytes()

    def bytes(self):
        return self.resident_bytes()

    def snapshot(self):
        bank, facts = self._state
        return dict(protocol=self.PROTOCOL, slots=self.slots, fact_ids=list(facts), store=bank.snapshot())

    def hash(self):
        bank, facts = self._state
        metadata = json.dumps(dict(protocol=self.PROTOCOL, slots=self.slots, fact_ids=facts),
                              sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256((metadata + bank.hash()).encode()).hexdigest()

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(self.snapshot(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @classmethod
    def from_snapshot(cls, state, *, record_routes=False):
        if (not isinstance(state, dict) or set(state) != {"protocol", "slots", "fact_ids", "store"}
                or state["protocol"] != cls.PROTOCOL):
            raise ValueError("Invalid grouped-store snapshot")
        buffer = io.BytesIO()
        torch.save(state["store"], buffer)
        buffer.seek(0)
        bank = PersistentVectorDB.load(buffer)
        result = cls(**state["store"]["config"], slots=state["slots"], record_routes=record_routes)
        facts = state["fact_ids"]
        if (not isinstance(facts, list) or any(not isinstance(fact, str) or not fact for fact in facts)
                or len(facts) != len(set(facts))):
            raise ValueError("Invalid fact IDs")
        expected = tuple(result.slot_id(fact, slot) for fact in facts for slot in range(result.slots))
        if bank.ids != expected:
            raise ValueError("Missing/reordered/foreign slots in fact groups")
        if any(len(set(bank.timestamps[i:i+result.slots])) != 1 for i in range(0, len(bank), result.slots)):
            raise ValueError("Partially updated fact timestamps")
        result._state = (bank, tuple(facts))
        return result

    @classmethod
    def load(cls, path, *, record_routes=False):
        return cls.from_snapshot(torch.load(path, map_location="cpu", weights_only=True), record_routes=record_routes)
