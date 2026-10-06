"""Two independently addressed banks feeding one rank-wise VeRA residual.

The base dictionary is a shared, offline-learned parameter collection. Episodic
keys/values remain caller-owned observations; this module never persists or
optimizes a test-time record. Each bank has its own retrieval quota and softmax.
The resulting rank vectors are added *before* the single existing VeRA gate.

Training may use dense retrieval or a straight-through mixture: the latter has
sparse forward values and dense backward derivatives. It incurs dense backward
work and is not a sparse-training-cost claim. Evaluation always uses sparse
retrieval, even if a training routing mode remains selected.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .interface_variants import InterfaceVectorVeRA
from .vector_store import PersistentVectorDB


ROUTING_MODES = {"sparse": 0, "dense": 1, "straight_through": 2}
DICTIONARY_VERSION = 1


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@torch.no_grad()
def merge_bank(keys: Tensor, values: Tensor, num_clusters: int, *, split: str,
               weights: Tensor | None = None, seed: int = 42,
               max_iterations: int = 40) -> dict[str, Any]:
    """Weighted spherical k-means and weighted, **unnormalized** value means.

    Inputs must be encoded training keys/values, not raw hidden features. An
    explicit ``split='train'`` is required; the caller must additionally verify
    provenance because tensor contents cannot identify dataset membership.
    This offline CPU operation has no autograd graph or global RNG side effect.

    ``within_value_variance[c]`` is the weighted mean of coordinate-mean squared
    distances to cluster c's value mean. Counts are original row counts, whereas
    weight_sums retain supplied multiplicities. Colliding/antipodal keys are
    handled deterministically, including nonempty-cluster repair.

    This compression is NOT function-preserving: merging changes softmax mass
    and top-k multiplicity. Counts are metadata, not an implicit log-count bias.
    Do not merge conflicting episodic facts merely because their keys are near.
    """
    if split != "train":
        raise ValueError("Dictionary initialization/merging accepts split='train' only")
    _positive_int(num_clusters, "num_clusters")
    _positive_int(max_iterations, "max_iterations")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (not isinstance(keys, Tensor) or not isinstance(values, Tensor)
            or keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]
            or min(*keys.shape, values.shape[1]) <= 0 or keys.shape[0] < num_clusters
            or not keys.is_floating_point() or not values.is_floating_point()):
        raise ValueError("keys/values must be nonempty floating [N,D] tensors, N >= num_clusters")
    k, v = keys.detach().cpu().float(), values.detach().cpu().float()
    norms = k.norm(dim=-1)
    if (not bool(torch.isfinite(k).all() and torch.isfinite(v).all() and torch.isfinite(norms).all())
            or bool((norms == 0).any())):
        raise ValueError("Finite values and finite nonzero keys are required")
    k = F.normalize(k, dim=-1)
    if weights is None:
        w = torch.ones(len(k))
    else:
        if not isinstance(weights, Tensor) or weights.shape != (len(k),):
            raise ValueError("weights must have shape [N]")
        w = weights.detach().cpu().float()
        if not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
            raise ValueError("weights must be finite and strictly positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = [int(torch.multinomial(w, 1, generator=generator))]
    # Update only the new center's distances: recomputing against every previous
    # center at every draw would make k-means++ O(N * K^2 * key_dim).
    nearest = (1 - k @ k[chosen[0]]).clamp_min(0)
    for _ in range(1, num_clusters):
        # For unit keys, 1-cosine is proportional to squared Euclidean distance,
        # so k-means++ samples nearest itself, not its square (which is D^4).
        probability = nearest * w
        probability[chosen] = 0
        if probability.sum() > 0:
            chosen.append(int(torch.multinomial(probability, 1, generator=generator)))
        else:
            chosen.append(next(i for i in range(len(k)) if i not in chosen))
        nearest = torch.minimum(nearest, (1 - k @ k[chosen[-1]]).clamp_min(0))
    centers, previous = k[chosen].clone(), None
    for iteration in range(max_iterations):
        similarity = k @ centers.T
        labels = similarity.argmax(dim=1)
        counts = torch.bincount(labels, minlength=num_clusters)
        # Even identical keys need an explicit nonempty assignment for every
        # requested cluster. Never remove the last member of an existing one.
        for missing in (counts == 0).nonzero().flatten().tolist():
            eligible = (counts[labels] > 1).nonzero().flatten()
            fit = similarity[eligible, labels[eligible]]
            donor = int(eligible[fit.argmin()])
            counts[labels[donor]] -= 1
            labels[donor] = missing
            counts[missing] += 1
        masses = torch.zeros(num_clusters).index_add_(0, labels, w)
        means = torch.zeros(num_clusters, k.shape[1]).index_add_(0, labels, k * w[:, None]) / masses[:, None]
        for cluster in (means.norm(dim=-1) <= 1e-12).nonzero().flatten().tolist():
            members = (labels == cluster).nonzero().flatten()
            means[cluster] = k[members[w[members].argmax()]]
        centers = F.normalize(means, dim=-1)
        if previous is not None and torch.equal(labels, previous):
            break
        previous = labels.clone()
    merged_values = torch.zeros(num_clusters, v.shape[1]).index_add_(0, labels, v * w[:, None]) / masses[:, None]
    squared_error = (v - merged_values[labels]).square().mean(-1)
    variance = torch.zeros(num_clusters).index_add_(0, labels, squared_error * w) / masses
    return dict(keys=centers, values=merged_values, assignments=labels,
                counts=torch.bincount(labels, minlength=num_clusters),
                weight_sums=masses, within_value_variance=variance,
                split=split, seed=seed, iterations=iteration + 1, source_records=len(k),
                num_clusters=num_clusters, exact_function_preserving=False, count_bias_applied=False)


class DictionaryVeRA(InterfaceVectorVeRA):
    """Shared prototype dictionary plus input-derived writable episodic bank.

    Defaults preserve rank/key64 and top4 + top4, with fixed base weight .25
    and episodic weight 1. Smaller dimensions are supported for CPU tests.
    A remains fixed; B is fixed unless the explicit inherited train_B control
    is enabled. Base values are free parameters, initially RMS-normalized;
    their learned or merged magnitudes are not silently renormalized.

    CPU-store compatibility is deliberate: ``delta_from_value(x, value)``
    interprets value as the *episodic-only* mixture and adds the base read.
    ``delta_from_mixed_value`` instead accepts an already combined rank vector
    and applies the underlying gate exactly once. The normal forward uses the
    latter, so the base contribution cannot be added twice.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 64,
                 key_dim: int = 64, top_k: int = 4, temperature: float = .2,
                 seed: int = 42, input_epsilon: float = 1e-6, value_epsilon: float = 1e-5,
                 *, writer_mode: str = "last_token", train_B: bool = False,
                 value_mlp_hidden: int = 0, architecture_version: int = 1,
                 base_size: int = 128, base_top_k: int = 4, alpha_base: float = .25,
                 base_trainable: bool = True, routing_mode: str = "sparse",
                 dictionary_version: int = DICTIONARY_VERSION):
        _positive_int(base_size, "base_size")
        _positive_int(base_top_k, "base_top_k")
        if isinstance(alpha_base, bool) or not math.isfinite(float(alpha_base)) or alpha_base < 0:
            raise ValueError("alpha_base must be finite and nonnegative")
        if type(base_trainable) is not bool:
            raise ValueError("base_trainable must be boolean")
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {tuple(ROUTING_MODES)}")
        if type(dictionary_version) is not int or dictionary_version != DICTIONARY_VERSION:
            raise ValueError("Unsupported dictionary architecture version")
        super().__init__(in_features, out_features, rank, key_dim, top_k, temperature, seed,
                         input_epsilon, value_epsilon, writer_mode=writer_mode, train_B=train_B,
                         value_mlp_hidden=value_mlp_hidden, architecture_version=architecture_version)
        self.base_size, self.base_top_k = base_size, base_top_k
        self.alpha_base, self.base_trainable = float(alpha_base), base_trainable
        self.dictionary_version, self._routing_mode = dictionary_version, routing_mode
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2000)
            keys = F.normalize(torch.randn(base_size, key_dim), dim=-1)
            values = torch.randn(base_size, rank)
            values = values / values.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        self.base_keys = nn.Parameter(keys, requires_grad=base_trainable)
        self.base_values = nn.Parameter(values, requires_grad=base_trainable)
        self.register_buffer("dictionary_architecture", torch.tensor(self._dictionary_metadata(), dtype=torch.int64))
        self.register_buffer("dictionary_alpha", torch.tensor(self.alpha_base, dtype=torch.float64))
        self.register_buffer("dictionary_routing", torch.tensor(ROUTING_MODES[routing_mode], dtype=torch.int64))
        self.base_cpu_override = None
        self.last_base_retrieval = None

    def _dictionary_metadata(self):
        return [self.dictionary_version, self.base_size, self.base_top_k, int(self.base_trainable)]

    @property
    def routing_mode(self):
        return self._routing_mode

    @property
    def effective_routing_mode(self):
        return self._routing_mode if self.training else "sparse"

    def set_routing_mode(self, mode: str):
        if mode not in ROUTING_MODES:
            raise ValueError(f"routing mode must be one of {tuple(ROUTING_MODES)}")
        self._routing_mode = mode
        self.dictionary_routing.fill_(ROUTING_MODES[mode])
        return self

    def set_base_trainable(self, trainable: bool):
        if type(trainable) is not bool:
            raise ValueError("trainable must be boolean")
        self.base_trainable = trainable
        self.base_keys.requires_grad_(trainable)
        self.base_values.requires_grad_(trainable)
        if not trainable:
            self.base_keys.grad = self.base_values.grad = None
        self.dictionary_architecture.copy_(torch.tensor(self._dictionary_metadata(),
                                                        device=self.dictionary_architecture.device))
        return self

    def set_alpha_base(self, alpha: float):
        """Explicit base-off/active switch that stays consistent in checkpoints."""
        if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or alpha < 0:
            raise ValueError("alpha_base must be finite and nonnegative")
        self.alpha_base = float(alpha)
        self.dictionary_alpha.fill_(alpha)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        # Existing runners call module.requires_grad_(True); preserve a declared
        # frozen-prototype control rather than accidentally training it.
        if not self.base_trainable:
            self.base_keys.requires_grad_(False)
            self.base_values.requires_grad_(False)
        return self

    def _retrieve(self, query, keys, values, top_k, mode):
        if keys.ndim not in (2, 3) or keys.shape[-1] != self.key_dim:
            raise ValueError(f"Expected keys[N,{self.key_dim}] or keys[B,N,{self.key_dim}]")
        if values.shape != (*keys.shape[:-1], self.rank):
            raise ValueError("Values must match bank dimensions and rank")
        if keys.device != query.device or values.device != query.device:
            raise ValueError("Queries, keys and values must share a device")
        if keys.ndim == 3 and (query.ndim != 3 or query.shape[0] != keys.shape[0]):
            raise ValueError("Independent banks require matching query[B,S,K]")
        n = keys.shape[-2]
        shape = (*query.shape[:-1], 0)
        if n == 0:
            empty = query.new_empty(shape)
            return query.new_zeros((*query.shape[:-1], self.rank)), dict(
                indices=torch.empty(shape, dtype=torch.long, device=query.device), scores=empty,
                weights=empty, topk_indices=torch.empty(shape, dtype=torch.long, device=query.device),
                retained_dense_mass=query.new_zeros(query.shape[:-1]))
        normalized_keys = F.normalize(keys.to(query.dtype), dim=-1, eps=1e-12)
        similarity = (query @ normalized_keys.T if keys.ndim == 2 else
                      torch.bmm(query, normalized_keys.transpose(1, 2)))
        scores, indices = similarity.topk(min(top_k, n), dim=-1)
        sparse_weights = F.softmax(scores / self.temperature, dim=-1)
        float_values = values.to(query.dtype)
        if keys.ndim == 2:
            selected = float_values[indices]
        else:
            selected = float_values[torch.arange(keys.shape[0], device=query.device)[:, None, None], indices]
        sparse = (sparse_weights.unsqueeze(-1) * selected).sum(-2)
        if mode == "sparse":
            mixed, forward_indices, forward_scores, forward_weights = sparse, indices, scores, sparse_weights
            with torch.no_grad():
                dense_weights = F.softmax(similarity.detach() / self.temperature, dim=-1)
        else:
            dense_weights = F.softmax(similarity / self.temperature, dim=-1)
            dense = (dense_weights @ float_values if keys.ndim == 2 else torch.bmm(dense_weights, float_values))
            if mode == "straight_through":
                mixed = dense + (sparse - dense).detach()
                forward_indices, forward_scores, forward_weights = indices, scores, sparse_weights
            else:
                mixed = dense
                forward_indices = torch.arange(n, device=query.device).expand(*query.shape[:-1], n)
                forward_scores, forward_weights = similarity, dense_weights
        return mixed, dict(indices=forward_indices, scores=forward_scores, weights=forward_weights,
                           topk_indices=indices,
                           retained_dense_mass=dense_weights.detach().gather(-1, indices).sum(-1))

    def mix_from_banks(self, x: Tensor, keys: Tensor, values: Tensor, *, return_info=False):
        """Retrieve both banks using actual x; return their combined rank vector.

        Legacy info indices/scores/weights refer ONLY to the episodic bank.
        base_* fields use an independent index space. Dense indices contain all
        rows; topk_indices always gives the sparse diagnostic candidate set.
        """
        query = self.encode_query(x)
        mode = self.effective_routing_mode
        episodic, info = self._retrieve(query, keys, values, self.top_k, mode)
        base, base_info = self._read_base(query)
        combined = episodic + self.alpha_base * base
        if not return_info:
            return combined
        info.update({"base_" + key: value for key, value in base_info.items()})
        info["routing_mode_code"] = torch.tensor(ROUTING_MODES[mode], device=x.device)
        info["dense_backward"] = torch.tensor(mode != "sparse", device=x.device)
        return combined, info

    def delta_from_mixed_value(self, x: Tensor, mixed_value: Tensor):
        """Apply one VeRA gate to an already combined base+episodic rank vector."""
        return super().delta_from_value(x, mixed_value)

    def final_delta(self, x: Tensor, episodic_mixed: Tensor, *, base_mixed: Tensor | None = None):
        """CPU/external retrieval entry point, with explicit optional base read.

        base_mixed, when supplied, is unscaled and prevents a second base lookup.
        This lets two CPU stores share an encoded query and feed one GPU gate.
        Without it, the base dictionary is retrieved on the module's device.
        """
        if base_mixed is None:
            query = self.encode_query(x)
            base_mixed, _ = self._read_base(query)
        return self.delta_from_mixed_value(x, episodic_mixed + self.alpha_base * base_mixed)

    def _read_base(self, query):
        if self.base_cpu_override is not None:
            if self.training:
                raise RuntimeError("CPU base override is restricted to frozen evaluation")
            info = self.base_cpu_override.search(query)
            self.last_base_retrieval = info
            return info["mixed_value"].to(query.device), {
                key: value.to(query.device) if isinstance(value, Tensor) else value
                for key, value in info.items() if key != "mixed_value"}
        mixed, info = self._retrieve(query, self.base_keys, self.base_values,
                                     self.base_top_k, self.effective_routing_mode)
        self.last_base_retrieval = {key: value.detach() for key, value in info.items()}
        return mixed, info

    @contextmanager
    def use_cpu_base(self, keys: Tensor, values: Tensor, ids=None):
        """Temporarily evaluate any detached CPU base dictionary, including empty.

        This does not change Parameters, model configuration, training flags or
        episodic storage. The yielded store can be hashed before/after reads.
        Nested scopes and exceptional exits restore the previous store/trace.
        IDs are audit metadata only and cannot affect query-based retrieval.
        """
        if self.training:
            raise RuntimeError("CPU base override requires module.eval()")
        if keys.ndim != 2 or keys.shape[-1] != self.key_dim or values.shape != (len(keys), self.rank):
            raise ValueError("CPU base override requires matching encoded keys and rank values")
        if ids is None:
            ids = [f"base-{index}" for index in range(len(keys))]
        if (len(ids) != len(keys) or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("CPU base IDs must be unique nonempty strings matching the rows")
        store = PersistentVectorDB(self.key_dim, self.rank, self.temperature, self.base_top_k)
        for id, key, value in zip(ids, keys, values):
            store.write(id, key, value, 0)
        previous, previous_info = self.base_cpu_override, self.last_base_retrieval
        self.base_cpu_override, self.last_base_retrieval = store, None
        try:
            yield store
        finally:
            self.base_cpu_override, self.last_base_retrieval = previous, previous_info

    def delta_from_value(self, x: Tensor, mixed_value: Tensor):
        """Existing backend contract: supplied value is episodic-only."""
        return self.final_delta(x, mixed_value)

    def forward(self, x: Tensor, keys: Tensor, values: Tensor, return_info: bool = False):
        mixed, info = self.mix_from_banks(x, keys, values, return_info=True)
        residual = self.delta_from_mixed_value(x, mixed)
        return (residual, info) if return_info else residual

    @torch.no_grad()
    def base_cpu_store(self) -> PersistentVectorDB:
        """Detached sparse CPU snapshot; never an automatic mutable cache.

        Export only in eval mode. Later parameter changes do not update this
        snapshot; callers own its lifetime. Values are unscaled (alpha is applied
        by final_delta), and original stable prototype indices remain its IDs.
        """
        if self.training:
            raise RuntimeError("Export a frozen base snapshot after module.eval()")
        store = PersistentVectorDB(self.key_dim, self.rank, self.temperature, self.base_top_k)
        for index, (key, value) in enumerate(zip(self.base_keys, self.base_values)):
            store.write(f"base-{index}", key, value, 0)
        return store

    @torch.no_grad()
    def initialise_base(self, keys: Tensor, values: Tensor, *, split: str,
                        weights: Tensor | None = None, seed: int = 42, max_iterations: int = 40):
        """Explicit train-only prototype initialization; optimizer reset is caller-owned."""
        if keys.ndim != 2 or keys.shape[-1] != self.key_dim or values.shape != (len(keys), self.rank):
            raise ValueError("Initialization requires encoded key_dim/rank bank vectors")
        result = merge_bank(keys, values, self.base_size, split=split, weights=weights,
                            seed=seed, max_iterations=max_iterations)
        self.base_keys.copy_(result["keys"])
        self.base_values.copy_(result["values"])
        return result

    def _validate_dictionary(self, state, prefix=""):
        expected = self._dictionary_metadata()
        architecture = state.get(prefix + "dictionary_architecture")
        alpha, mode = state.get(prefix + "dictionary_alpha"), state.get(prefix + "dictionary_routing")
        if (not isinstance(architecture, Tensor) or architecture.dtype != torch.int64
                or tuple(architecture.shape) != (4,) or architecture.cpu().tolist() != expected):
            raise RuntimeError("Checkpoint dictionary architecture differs; use explicit migration for older modules")
        if (not isinstance(alpha, Tensor) or alpha.shape != () or alpha.dtype != torch.float64
                or float(alpha) != self.alpha_base):
            raise RuntimeError("Checkpoint dictionary alpha differs from configuration")
        if (not isinstance(mode, Tensor) or mode.shape != () or mode.dtype != torch.int64
                or int(mode) not in ROUTING_MODES.values()):
            raise RuntimeError("Invalid dictionary routing mode metadata")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._validate_dictionary(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        try:
            self._validate_dictionary(state_dict, prefix)
        except RuntimeError as error:
            error_msgs.append(str(error))
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._routing_mode = next(k for k, v in ROUTING_MODES.items() if v == int(self.dictionary_routing))

    def load_interface_state_dict(self, state_dict, *, strict=True):
        """Copy an InterfaceVectorVeRA checkpoint, retaining fresh base prototypes."""
        if "dictionary_architecture" in state_dict:
            raise ValueError("Dictionary checkpoint requires regular load_state_dict")
        self._validate_architecture(state_dict)
        migrated = OrderedDict(state_dict)
        for name, value in self.state_dict().items():
            if name.startswith("dictionary_") or name in ("base_keys", "base_values"):
                migrated[name] = value
        return self.load_state_dict(migrated, strict=strict)

    def load_legacy_state_dict(self, state_dict, *, strict=True):
        if "interface_architecture" in state_dict:
            return self.load_interface_state_dict(state_dict, strict=strict)
        temporary = InterfaceVectorVeRA(**super().configuration())
        temporary.load_legacy_state_dict(state_dict, strict=strict)
        return self.load_interface_state_dict(temporary.state_dict(), strict=strict)

    def configuration(self):
        config = super().configuration()
        config.update(base_size=self.base_size, base_top_k=self.base_top_k, alpha_base=self.alpha_base,
                      base_trainable=self.base_trainable, routing_mode=self.routing_mode,
                      dictionary_version=self.dictionary_version)
        return config
