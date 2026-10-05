"""Small persistent CPU vector store for token-level memory experiments.

This is an exact flat reference index, not a production database or concurrent
service. Scoring one query scans O(N * key_dim) values, followed by top-k
selection. Key/value payloads are authoritative CPU float32 tensors. Search is
read-only and intentionally detached from autograd; offline differentiable
alignment training should use its own tensor path before populating this store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _timestamp(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp must be a finite int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    return value


def _vector(value: Tensor, dimension: int, name: str) -> Tensor:
    vector = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError(f"{name} must have shape ({dimension},)")
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite values")
    return vector.contiguous().clone()


class PersistentVectorDB:
    """CPU cosine top-k search with explicit latest-timestamp upserts.

    Older writes for an existing ID are rejected. Equal-timestamp writes use
    last-write-wins semantics, allowing deterministic replay/upsert in input
    order. IDs identify records for storage and audit; search only consumes
    query tensors and cannot filter using a known answer or evidence ID.

    The caller owns synchronization: concurrent writes/save/search are not
    supported. A generation request should observe a fixed bank snapshot.
    """

    FORMAT_VERSION = 1

    def __init__(
        self, key_dim: int, value_dim: int, temperature: float = 0.2, top_k: int = 4
    ) -> None:
        self.key_dim = _positive_integer("key_dim", key_dim)
        self.value_dim = _positive_integer("value_dim", value_dim)
        self.top_k = _positive_integer("top_k", top_k)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a finite positive number")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        self._keys = torch.empty((0, self.key_dim), dtype=torch.float32, device="cpu")
        self._values = torch.empty((0, self.value_dim), dtype=torch.float32, device="cpu")
        self._ids: list[str] = []
        self._timestamps: list[int | float] = []
        self._id_to_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    @property
    def timestamps(self) -> tuple[int | float, ...]:
        return tuple(self._timestamps)

    @property
    def keys(self) -> Tensor:
        return self._keys.clone()

    @property
    def values(self) -> Tensor:
        return self._values.clone()

    def write(self, id: str, key: Tensor, value: Tensor, timestamp: int | float) -> None:
        if not isinstance(id, str) or not id:
            raise ValueError("id must be a nonempty string")
        time = _timestamp(timestamp)
        position = self._id_to_index.get(id)
        if position is not None and time < self._timestamps[position]:
            raise ValueError(f"Stale write for {id!r}: timestamp precedes stored observation")
        key_copy = _vector(key, self.key_dim, "key")
        norm = key_copy.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("key must have a finite nonzero norm")
        key_copy = key_copy / norm
        value_copy = _vector(value, self.value_dim, "value")
        if position is None:
            # Finish all validation before mutating authoritative state.
            self._keys = torch.cat((self._keys, key_copy.unsqueeze(0)), dim=0)
            self._values = torch.cat((self._values, value_copy.unsqueeze(0)), dim=0)
            self._id_to_index[id] = len(self._ids)
            self._ids.append(id)
            self._timestamps.append(time)
        else:
            self._keys[position].copy_(key_copy)
            self._values[position].copy_(value_copy)
            self._timestamps[position] = time

    def search(self, queries: Tensor) -> dict[str, Any]:
        """Return CPU tensors preserving all query leading dimensions.

        ``ids`` is a list of hit-ID lists in flattened query order. Tensor keys
        are mixed_value [..., value_dim] and indices/weights/scores [..., k],
        with k=min(top_k, bank_size). Empty banks return k=0 and zero mixtures.
        Equal-score tie order follows torch.topk and is not a semantic policy.
        """
        query = torch.as_tensor(queries).detach().to(device="cpu", dtype=torch.float32)
        if query.ndim < 1 or query.shape[-1] != self.key_dim:
            raise ValueError(f"queries must have final dimension {self.key_dim}")
        if not torch.isfinite(query).all():
            raise ValueError("queries must be finite")
        leading_shape = tuple(query.shape[:-1])
        flat = query.reshape(-1, self.key_dim)
        count = flat.shape[0]
        k = min(self.top_k, len(self))
        if k == 0 or count == 0:
            return {
                "mixed_value": torch.zeros((*leading_shape, self.value_dim), dtype=torch.float32),
                "indices": torch.empty((*leading_shape, k), dtype=torch.long),
                "weights": torch.empty((*leading_shape, k), dtype=torch.float32),
                "scores": torch.empty((*leading_shape, k), dtype=torch.float32),
                "ids": [[] for _ in range(count)],
            }
        norms = flat.norm(dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Nonempty-bank queries must have finite nonzero norms")
        similarities = (flat / norms) @ self._keys.T
        scores, indices = torch.topk(similarities, k=k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        selected_values = self._values[indices]
        mixed = (selected_values * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "mixed_value": mixed.reshape(*leading_shape, self.value_dim).clone(),
            "indices": indices.reshape(*leading_shape, k).clone(),
            "weights": weights.reshape(*leading_shape, k).clone(),
            "scores": scores.reshape(*leading_shape, k).clone(),
            "ids": [[self._ids[index] for index in row] for row in indices.tolist()],
        }

    def snapshot(self) -> dict[str, Any]:
        """An isolated, weights_only-compatible state suitable for torch.save."""
        return {
            "format_version": self.FORMAT_VERSION,
            "config": {"key_dim": self.key_dim, "value_dim": self.value_dim,
                       "temperature": self.temperature, "top_k": self.top_k},
            "ids": list(self._ids),
            "timestamps": list(self._timestamps),
            "keys": self._keys.clone(),
            "values": self._values.clone(),
        }

    def hash(self) -> str:
        """Stable content hash including config, order, timestamps and payload."""
        metadata = {"format_version": self.FORMAT_VERSION,
                    "config": {"key_dim": self.key_dim, "value_dim": self.value_dim,
                               "temperature": self.temperature, "top_k": self.top_k},
                    "ids": self._ids, "timestamps": self._timestamps}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":"), allow_nan=False).encode("utf-8"))
        # Pure torch + stdlib: no optional NumPy dependency for byte hashing.
        for tensor in (self._keys, self._values):
            digest.update(bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist()))
        return digest.hexdigest()

    def resident_bytes(self) -> dict[str, int]:
        """Actual key/value CPU tensor payload, excluding Python object overhead."""
        key_bytes = self._keys.numel() * self._keys.element_size()
        value_bytes = self._values.numel() * self._values.element_size()
        return {"key_bytes": key_bytes, "value_bytes": value_bytes,
                "total_bytes": key_bytes + value_bytes}

    def bytes(self) -> dict[str, int]:
        return self.resident_bytes()

    def save(self, path: str | Path) -> None:
        """Atomically replace path with a tensor/primitive-only CPU checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(self.snapshot(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> PersistentVectorDB:
        state = torch.load(path, map_location="cpu", weights_only=True)
        required = {"format_version", "config", "ids", "timestamps", "keys", "values"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Invalid vector-store checkpoint fields")
        if state["format_version"] != cls.FORMAT_VERSION:
            raise ValueError("Unsupported vector-store checkpoint version")
        config = state["config"]
        if not isinstance(config, dict) or set(config) != {"key_dim", "value_dim", "temperature", "top_k"}:
            raise ValueError("Invalid vector-store configuration")
        store = cls(**config)
        ids, timestamps, keys, values = (state[name] for name in ("ids", "timestamps", "keys", "values"))
        if not isinstance(ids, list) or any(not isinstance(id, str) or not id for id in ids):
            raise ValueError("Invalid stored IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate stored IDs")
        if not isinstance(timestamps, list) or len(timestamps) != len(ids):
            raise ValueError("Invalid stored timestamps")
        for timestamp in timestamps:
            _timestamp(timestamp)
        for tensor, dim, name in ((keys, store.key_dim, "keys"), (values, store.value_dim, "values")):
            if not isinstance(tensor, Tensor) or tensor.dtype != torch.float32:
                raise ValueError(f"Stored {name} must be float32 tensors")
            if tuple(tensor.shape) != (len(ids), dim) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid stored {name} shape or contents")
        if len(ids) and not torch.allclose(keys.norm(dim=-1), torch.ones(len(ids)), atol=1e-5, rtol=1e-5):
            raise ValueError("Stored keys must be unit-normalized")
        # Do not renormalize on reload: preserve exact stored bits/content hash.
        store._keys = keys.detach().cpu().contiguous().clone()
        store._values = values.detach().cpu().contiguous().clone()
        store._ids = list(ids)
        store._timestamps = list(timestamps)
        store._id_to_index = {id: index for index, id in enumerate(ids)}
        return store
