"""Small, auditable memory components for the VeRA-Mem experiments.

These modules do not implement the separate VeRA random-projection method.
The caller controls observation timing: keys must be made from the query before
the answer is revealed, and writes must happen after prediction. Retrieval and
routing never update learned parameters or stored observations.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _copy_value(value: Any) -> Any:
    """Detach observations; memory-bank writes are not a training graph."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return copy.deepcopy(value)


def _unit_vector(value: Tensor, expected_dim: int | None = None) -> Tensor:
    vector = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if vector.ndim == 2 and 1 in vector.shape:
        vector = vector.reshape(-1)
    if vector.ndim != 1 or not vector.numel():
        raise ValueError("Keys and queries must be nonempty vectors")
    if expected_dim is not None and vector.numel() != expected_dim:
        raise ValueError(f"Expected dimension {expected_dim}, got {vector.numel()}")
    norm = vector.norm()
    if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Keys and queries must be finite and have nonzero norm")
    return (vector / norm).clone()


class LoRABank(nn.Module):
    """Independent LoRA slots with equal active rank and explicit storage cost.

    ``forward`` returns only the residual, so the caller adds it to the frozen
    projection output. Each slot starts as an exact zero residual. Computation
    uses the parameter dtype (float32 by default), then returns the input dtype.
    Only the selected slot participates in autograd.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        slots: int = 1,
        alpha: float | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_features = _positive_int("in_features", in_features)
        self.out_features = _positive_int("out_features", out_features)
        self.rank = _positive_int("rank", rank)
        self.slots = _positive_int("slots", slots)
        self.alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        self.scaling = self.alpha / rank
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for _ in range(slots):
            a = torch.empty(rank, in_features, dtype=torch.float32)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            self.A.append(nn.Parameter(a))
            self.B.append(nn.Parameter(torch.zeros(out_features, rank)))

    def _validate_slot(self, slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        if not 0 <= slot < self.slots:
            raise IndexError(f"slot {slot} is outside [0, {self.slots})")

    def forward(self, x: Tensor, slot: int = 0) -> Tensor:
        self._validate_slot(slot)
        value = F.linear(x.to(dtype=self.A[slot].dtype), self.A[slot])
        return (F.linear(value, self.B[slot]) * self.scaling).to(dtype=x.dtype)

    def parameters_for_slot(self, slot: int) -> list[nn.Parameter]:
        self._validate_slot(slot)
        return [self.A[slot], self.B[slot]]

    def trainable_numel(self, active: bool = False) -> int:
        """Number of trainable elements; ``active=True`` counts one slot."""
        if active:
            return sum(p.numel() for p in self.parameters_for_slot(0) if p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LatentResidual(nn.Module):
    """Trainable low-rank value encoder and residual reader.

    ``encode`` preserves gradients for offline episodic training. To populate
    an inference bank, the caller should detach its outputs (``ExactMemory``
    does this automatically). A zero-initialized reader makes the initial
    residual zero. Consequently the encoder receives a nonzero learning signal
    only after the reader starts learning. Frozen backbone weights must still
    allow activation gradients after the injection layer during training.
    """

    def __init__(self, out_features: int, rank: int = 16, seed: int = 42) -> None:
        super().__init__()
        self.out_features = _positive_int("out_features", out_features)
        self.rank = _positive_int("rank", rank)
        # fork_rng avoids changing the experiment's global random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = nn.Linear(out_features, rank, bias=False, dtype=torch.float32)
            self.reader = nn.Linear(rank, out_features, bias=False, dtype=torch.float32)
            nn.init.zeros_(self.reader.weight)

    def encode(self, hidden: Tensor) -> Tensor:
        return self.encoder(hidden.to(dtype=self.encoder.weight.dtype))

    def forward(self, value_vector: Tensor) -> Tensor:
        return self.reader(value_vector.to(dtype=self.reader.weight.dtype))

    def trainable_numel(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class ExactMemory:
    """CPU float32 cosine search with idempotent, latest-observation writes.

    Rewriting an id replaces its value and key. Retrieval uses a stable
    insertion-order tie break and returns independent copies. This reference
    implementation is intentionally an exact scan, not a scalable ANN index.
    Timestamps are metadata; a caller must enforce its temporal write policy.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._dimension: int | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def write(self, id: str, key: Tensor, value: Any, timestamp: int | float) -> None:
        normalized = _unit_vector(key, self._dimension)
        if not isinstance(id, str) or not id:
            raise ValueError("id must be a nonempty string")
        self._entries[id] = {
            "id": id,
            "key": normalized,
            "value": _copy_value(value),
            "timestamp": copy.deepcopy(timestamp),
        }
        self._dimension = normalized.numel()

    def retrieve(self, query: Tensor, top_k: int = 1) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not self._entries or top_k == 0:
            return []
        normalized = _unit_vector(query, self._dimension)
        records = list(self._entries.values())
        scores = torch.stack([record["key"] for record in records]) @ normalized
        order = sorted(range(len(records)), key=lambda index: (-float(scores[index]), index))
        return [
            {
                "id": records[index]["id"],
                "score": float(scores[index]),
                "value": _copy_value(records[index]["value"]),
                "timestamp": copy.deepcopy(records[index]["timestamp"]),
            }
            for index in order[:top_k]
        ]

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "entries": [_copy_value(record) for record in self._entries.values()]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported ExactMemory state version")
        restored = ExactMemory()
        for record in state["entries"]:
            if record["id"] in restored._entries:
                raise ValueError("Duplicate id in ExactMemory state")
            restored.write(record["id"], record["key"], record["value"], record["timestamp"])
        self._entries = restored._entries
        self._dimension = restored._dimension


class FrozenRouter:
    """Seeded spherical k-means fitted once on answer-free priming features.

    Routing and fitting run on detached CPU float32 features. Fit is a one-time
    operation to avoid accidental changes during the online stream. ``centroids``
    returns a copy; checkpoint restoration creates a new frozen router.
    """

    def __init__(self, slots: int = 4, seed: int = 42) -> None:
        self.slots = _positive_int("slots", slots)
        self.seed = seed
        self._centroids: Tensor | None = None

    @property
    def centroids(self) -> Tensor | None:
        return None if self._centroids is None else self._centroids.clone()

    def fit(self, features: Tensor, max_iterations: int = 40) -> "FrozenRouter":
        if self._centroids is not None:
            raise RuntimeError("FrozenRouter is already fitted; create a new router to refit")
        _positive_int("max_iterations", max_iterations)
        data = torch.as_tensor(features).detach().cpu().float()
        if data.ndim != 2 or data.shape[0] < self.slots or data.shape[1] == 0:
            raise ValueError("features must be [N, D] with N >= slots and D > 0")
        norms = data.norm(dim=-1)
        if not torch.isfinite(data).all() or (norms <= 0).any():
            raise ValueError("Every feature must be finite and have nonzero norm")
        data = F.normalize(data, dim=-1)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        selected = [int(torch.randint(len(data), (1,), generator=generator))]
        for _ in range(1, self.slots):
            distances = (1 - data @ data[selected].T).clamp_min(0).min(dim=1).values
            distances[selected] = 0
            if float(distances.sum()) > 1e-12:
                selected.append(int(torch.multinomial(distances.square(), 1, generator=generator)))
            else:
                selected.append(next(index for index in range(len(data)) if index not in selected))
        centroids = data[selected].clone()
        for _ in range(max_iterations):
            similarities = data @ centroids.T
            labels = similarities.argmax(dim=-1)
            updated = torch.empty_like(centroids)
            # Re-seed an empty/cancelling cluster with a poorly represented row.
            candidates = torch.argsort(similarities.max(dim=-1).values, stable=True).tolist()
            used_replacements: set[int] = set()
            for slot in range(self.slots):
                members = data[labels == slot]
                mean = members.mean(dim=0) if len(members) else torch.zeros(data.shape[1])
                if mean.norm() <= 1e-12:
                    index = next(index for index in candidates if index not in used_replacements)
                    used_replacements.add(index)
                    updated[slot] = data[index]
                else:
                    updated[slot] = F.normalize(mean, dim=0)
            if torch.allclose(updated, centroids, atol=1e-6, rtol=0):
                centroids = updated
                break
            centroids = updated
        self._centroids = centroids.clone()
        return self

    def route(self, feature: Tensor) -> int:
        if self._centroids is None:
            raise RuntimeError("Fit the router before routing")
        normalized = _unit_vector(feature, self._centroids.shape[1])
        return int((self._centroids @ normalized).argmax())

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "slots": self.slots, "seed": self.seed, "centroids": self.centroids}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._centroids is not None:
            raise RuntimeError("Load a checkpoint into a new, unfitted router")
        if state.get("version") != 1 or state["slots"] != self.slots:
            raise ValueError("Incompatible FrozenRouter state")
        centers = state["centroids"]
        if centers is not None:
            if centers.ndim != 2 or centers.shape[0] != self.slots:
                raise ValueError("Invalid centroid shape")
            self._centroids = torch.stack([_unit_vector(row) for row in centers])
        self.seed = state["seed"]


class LexicalMemory:
    """Transparent TF-IDF cosine baseline for small text-memory pilots.

    This is lexical retrieval, not a semantic embedding model or BM25. The
    vocabulary and smoothed IDF are recomputed over current documents at read
    time. Tokenization is Unicode word tokenization with case folding; term
    frequency is the raw count and idf = log((N + 1) / (df + 1)) + 1.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _tokens(text: str) -> Counter[str]:
        return Counter(re.findall(r"\w+", text.casefold()))

    def write(self, id: str, text: str, timestamp: int | float) -> None:
        if not isinstance(id, str) or not id or not isinstance(text, str):
            raise ValueError("id must be a nonempty string and text must be a string")
        self._entries[id] = {"id": id, "text": text, "timestamp": copy.deepcopy(timestamp)}

    def retrieve(self, question: str, top_k: int = 1) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not self._entries or top_k == 0:
            return []
        records = list(self._entries.values())
        documents = [self._tokens(record["text"]) for record in records]
        document_frequency = Counter(token for document in documents for token in document)
        idf = {token: math.log((len(documents) + 1) / (count + 1)) + 1
               for token, count in document_frequency.items()}
        query = {token: count * idf[token] for token, count in self._tokens(question).items() if token in idf}
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        scores = []
        for document in documents:
            weighted = {token: count * idf[token] for token, count in document.items()}
            norm = math.sqrt(sum(value * value for value in weighted.values()))
            dot = sum(weighted.get(token, 0) * value for token, value in query.items())
            scores.append(dot / (norm * query_norm) if norm and query_norm else 0.0)
        order = sorted(range(len(records)), key=lambda index: (-scores[index], index))
        return [
            {**copy.deepcopy(records[index]), "value": records[index]["text"], "score": scores[index]}
            for index in order[:top_k]
        ]

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "entries": copy.deepcopy(list(self._entries.values()))}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported LexicalMemory state version")
        restored = LexicalMemory()
        for record in state["entries"]:
            if record["id"] in restored._entries:
                raise ValueError("Duplicate id in LexicalMemory state")
            restored.write(record["id"], record["text"], record["timestamp"])
        self._entries = restored._entries
