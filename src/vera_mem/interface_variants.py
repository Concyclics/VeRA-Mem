"""Explicit writer/readout controls for the existing sparse VeRA interface.

The baseline uses the historical final-prompt-token feature for both key and
value. The pooled writer changes only value features: callers still pass the
historical last-token feature to ``encode_key`` and pass the masked mean of
observation-token inputs to ``encode_value``. Token masks must exclude the
Remember instruction, system/chat scaffolding, generation markers and padding;
only the feature extractor can establish those semantic boundaries.

Pooling operates on raw down-projection inputs before normalization. Its value
center is fitted separately on offline training observations and never inferred
from the historical last-token support center. Per-record key/rank dimensions
do not change. Learned B, when enabled, is one shared readout parameter.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .counterfactual_backend import BatchedStableVectorVeRA


ARCHITECTURE_VERSION = 1
WRITER_MODES = {"last_token": 0, "masked_mean": 1}


def masked_mean_support(token_features: Tensor, token_mask: Tensor) -> Tensor:
    """Mean raw observation features ``[..., tokens, D]`` over a binary mask.

    Accumulation is float32, including for BF16 features. There must be at least
    one selected token per observation. Masked positions contribute neither
    values nor gradients, even if their unused feature storage contains NaN.
    No tokenizer or boundary heuristic is hidden in this pooling operation.
    """
    if token_features.ndim < 2 or token_features.shape[-2] == 0 or token_features.shape[-1] == 0:
        raise ValueError("token_features must be nonempty [..., tokens, features]")
    if not token_features.is_floating_point():
        raise ValueError("token_features must be floating point")
    if token_mask.shape != token_features.shape[:-1] or token_mask.device != token_features.device:
        raise ValueError("token_mask must match all token dimensions and the feature device")
    if token_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("token_mask must be boolean or binary integer")
    if bool(((token_mask != 0) & (token_mask != 1)).any()):
        raise ValueError("token_mask must contain only zero and one")
    selected = token_mask.bool()
    counts = selected.sum(dim=-1)
    if bool((counts == 0).any()):
        raise ValueError("Every observation must select at least one support token")
    clean = token_features.float().masked_fill(~selected.unsqueeze(-1), 0.)
    pooled = clean.sum(dim=-2) / counts.unsqueeze(-1)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("Selected observation features must have finite means")
    return pooled


class InterfaceVectorVeRA(BatchedStableVectorVeRA):
    """Compatible last-token/pooled writer and fixed/learned-B controls.

    ``encode_key`` always retains the existing last-token writer path. In
    ``masked_mean`` mode, ``encode_value`` accepts *already pooled* [..., D]
    features, not an ambiguous [..., tokens, D] tensor. Use ``pool_support`` or
    ``encode_pooled_value`` with an explicit content mask for token inputs.

    Optional ``value_mlp_hidden > 0`` adds a two-layer GELU residual alongside
    Wv, with zero-initialized output weights. With last-token features it is an
    exact baseline at initialization; it remains a separate architecture choice.
    ``train_B`` promotes the same initialized B tensor from buffer to shared
    Parameter. A remains a frozen buffer. Online callers still freeze all shared
    parameters; constructor options describe architecture, not current grad mode.

    All architecture metadata is stored as a small integer tensor so existing
    tensor-only state digests continue to work. ``configuration`` records the
    full constructor settings. Explicit legacy migration preserves checkpoint
    parameters while initializing new pooled-value statistics as unfitted.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 64,
                 key_dim: int = 64, top_k: int = 4, temperature: float = .2,
                 seed: int = 42, input_epsilon: float = 1e-6, value_epsilon: float = 1e-5,
                 *, writer_mode: str = "last_token", train_B: bool = False,
                 value_mlp_hidden: int = 0, architecture_version: int = ARCHITECTURE_VERSION):
        if writer_mode not in WRITER_MODES:
            raise ValueError(f"writer_mode must be one of {tuple(WRITER_MODES)}")
        if not isinstance(train_B, bool):
            raise TypeError("train_B must be boolean")
        if isinstance(value_mlp_hidden, bool) or not isinstance(value_mlp_hidden, int) or value_mlp_hidden < 0:
            raise ValueError("value_mlp_hidden must be a nonnegative integer")
        if type(architecture_version) is not int or architecture_version != ARCHITECTURE_VERSION:
            raise ValueError("Unsupported interface architecture version")
        super().__init__(in_features, out_features, rank=rank, key_dim=key_dim, top_k=top_k,
                         temperature=temperature, seed=seed, input_epsilon=input_epsilon,
                         value_epsilon=value_epsilon)
        self.writer_mode = writer_mode
        self.train_B = train_B
        self.value_mlp_hidden = value_mlp_hidden
        self.architecture_version = architecture_version
        self.register_buffer("interface_architecture", torch.tensor(
            [architecture_version, WRITER_MODES[writer_mode], int(train_B), value_mlp_hidden], dtype=torch.int64))
        if train_B:
            initial = self.B.detach().clone()
            del self.B
            self.register_parameter("B", nn.Parameter(initial))
        self._value_statistics_ready = False
        if writer_mode == "masked_mean":
            self.register_buffer("value_center", torch.zeros(in_features, dtype=torch.float32))
            self.register_buffer("value_statistics_fitted", torch.tensor(False))
        self.value_mlp = None
        if value_mlp_hidden:
            # Construction, like the parent, must not consume global training
            # RNG or change CUDA generator states.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 1000)
                self.value_mlp = nn.Sequential(
                    nn.Linear(in_features, value_mlp_hidden, bias=False, dtype=torch.float32),
                    nn.GELU(),
                    nn.Linear(value_mlp_hidden, rank, bias=False, dtype=torch.float32))
                nn.init.zeros_(self.value_mlp[-1].weight)

    def pool_support(self, token_features: Tensor, token_mask: Tensor) -> Tensor:
        if self.writer_mode != "masked_mean":
            raise ValueError("pool_support requires writer_mode='masked_mean'; baseline uses the final prompt token")
        if token_features.ndim < 2 or token_features.shape[-1] != self.in_features:
            raise ValueError(f"Expected token feature width {self.in_features}")
        return masked_mean_support(token_features, token_mask)

    def encode_pooled_value(self, token_features: Tensor, token_mask: Tensor) -> Tensor:
        return self.encode_value(self.pool_support(token_features, token_mask))

    def encode_value(self, x: Tensor) -> Tensor:
        if self.writer_mode == "last_token" and self.value_mlp is None:
            # Keep the baseline computation path bit-for-bit identical.
            return super().encode_value(x)
        if self.writer_mode == "masked_mean":
            if not self._value_statistics_ready:
                raise RuntimeError("Fit pooled value statistics on offline training observations before encoding")
            center = self.value_center
        else:
            center = self.support_center
        normalized = self._domain_input(x, center)
        projected = self.Wv(normalized)
        if self.value_mlp is not None:
            projected = projected + self.value_mlp(normalized)
        return self._rms_normalize(projected, self.value_epsilon)

    @torch.no_grad()
    def fit_value_statistics(self, training_pooled_inputs: Tensor) -> "InterfaceVectorVeRA":
        """Fit a separate immutable mean of normalized pooled TRAIN features.

        Last-token query/key centers still come from inherited ``fit_statistics``
        or a legacy checkpoint. Dataset membership cannot be inferred here; the
        runner must supply training features and persist their provenance.
        """
        if self.writer_mode != "masked_mean":
            raise ValueError("Separate value statistics are only used by the masked-mean writer")
        if self._value_statistics_ready:
            raise RuntimeError("Pooled value statistics are already fitted and immutable")
        if (training_pooled_inputs.ndim != 2 or training_pooled_inputs.shape[0] == 0
                or training_pooled_inputs.shape[1] != self.in_features):
            raise ValueError(f"training_pooled_inputs must be nonempty [N, {self.in_features}]")
        inputs = training_pooled_inputs.to(device=self.Wv.weight.device, dtype=self.Wv.weight.dtype)
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("Training pooled value features must be finite")
        normalized = self._normalized_input(inputs)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("Training pooled value normalization produced nonfinite values")
        self.value_center.copy_(normalized.mean(dim=0))
        self.value_statistics_fitted.fill_(True)
        self._value_statistics_ready = True
        return self

    def _expected_architecture(self) -> list[int]:
        return [self.architecture_version, WRITER_MODES[self.writer_mode], int(self.train_B), self.value_mlp_hidden]

    def _validate_architecture(self, state_dict, prefix=""):
        key = prefix + "interface_architecture"
        if key not in state_dict:
            raise RuntimeError("Missing interface architecture metadata; use load_legacy_state_dict explicitly")
        value = state_dict[key]
        if not isinstance(value, Tensor) or value.dtype != torch.int64 or tuple(value.shape) != (4,):
            raise RuntimeError("Invalid interface architecture metadata")
        if value.detach().cpu().tolist() != self._expected_architecture():
            raise RuntimeError("Checkpoint interface architecture differs from the constructed module")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._validate_architecture(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        try:
            self._validate_architecture(state_dict, prefix)
        except RuntimeError as error:
            error_msgs.append(str(error))
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.writer_mode == "masked_mean":
            self._value_statistics_ready = bool(self.value_statistics_fitted.item())

    def load_legacy_state_dict(self, state_dict, *, strict=True):
        """Explicitly initialize a fresh variant from a StableVectorVeRA state.

        Existing weights/centers copy exactly. New pooled-value centers remain
        unfitted; optional residual MLP uses this constructor's initialization.
        Instantiate a fresh variant before migration, then fit pooled TRAIN
        statistics. This method never silently treats a variant as a legacy one.
        """
        if "interface_architecture" in state_dict:
            raise ValueError("This checkpoint already has interface architecture metadata")
        if self.writer_mode == "masked_mean" and self._value_statistics_ready:
            raise RuntimeError("Migrate legacy weights before fitting pooled value statistics")
        if self.value_mlp is not None and bool(self.value_mlp[-1].weight.detach().count_nonzero()):
            raise RuntimeError("Legacy migration requires a fresh zero-output residual MLP")
        migrated = OrderedDict(state_dict)
        if hasattr(state_dict, "_metadata"):
            migrated._metadata = state_dict._metadata
        current = self.state_dict()
        added = {"interface_architecture"}
        if self.writer_mode == "masked_mean":
            added.update(("value_center", "value_statistics_fitted"))
        if self.value_mlp is not None:
            added.update(name for name in current if name.startswith("value_mlp."))
        for name in added:
            if name in migrated:
                raise ValueError(f"Legacy checkpoint unexpectedly contains variant field {name}")
            migrated[name] = current[name]
        return self.load_state_dict(migrated, strict=strict)

    def configuration(self) -> dict[str, Any]:
        result = super().configuration()
        result.update(writer_mode=self.writer_mode, train_B=self.train_B,
                      value_mlp_hidden=self.value_mlp_hidden, architecture_version=self.architecture_version)
        return result

    def checkpoint_payload(self) -> dict[str, Any]:
        return {"module": self.state_dict(), "interface_configuration": self.configuration()}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *, legacy_configuration=None) -> "InterfaceVectorVeRA":
        """Rebuild architecture and load state; legacy hyperparameters are explicit.

        Shape metadata alone cannot recover top-k, temperature, normalization
        epsilons, or experimental writer semantics, so those are never guessed.
        ``legacy_configuration`` is needed only for old checkpoints without the
        new ``interface_configuration`` field.
        """
        if "module" not in checkpoint:
            raise ValueError("Checkpoint must contain module state")
        configuration = checkpoint.get("interface_configuration")
        if configuration is None:
            if legacy_configuration is None:
                raise ValueError("Legacy checkpoint requires explicit legacy_configuration")
            module = cls(**legacy_configuration)
            module.load_legacy_state_dict(checkpoint["module"])
        else:
            module = cls(**configuration)
            module.load_state_dict(checkpoint["module"])
        return module
