"""Independent per-query memory banks for single-record interventions.

For a batch of questions, replacing every target in one shared bank changes
several records at once. A [batch, records, features] bank instead allows each
question's counterfactual branch to replace exactly its own target record while
leaving all other records intact. The runner constructs and audits those banks;
this module guarantees that a row retrieves only within its own supplied bank.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .context_distillation import ContextDistillationBackend
from .stable_vector_vera import StableVectorVeRA


class BatchedStableVectorVeRA(StableVectorVeRA):
    """StableVectorVeRA with optional independent banks, unchanged checkpoints.

    No parameters or buffers are added: existing StableVectorVeRA state_dicts
    load strictly, including their fixed offline training statistics. Shared
    [records, features] banks use the original forward path without alteration.
    For independent banks, x must be [batch, sequence, in_features], keys must
    be [batch, records, key_dim], and values [batch, records, rank].
    """

    def forward(self, x: Tensor, keys: Tensor, values: Tensor,
                return_info: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if keys.ndim == 2:
            return super().forward(x, keys, values, return_info=return_info)
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError(f"Independent banks require x[batch, sequence, {self.in_features}]")
        if keys.ndim != 3 or keys.shape[0] != x.shape[0] or keys.shape[2] != self.key_dim:
            raise ValueError(f"Expected keys[batch, records, {self.key_dim}] matching x batch")
        batch, records, _ = keys.shape
        if values.ndim != 3 or values.shape != (batch, records, self.rank):
            raise ValueError(f"Expected values[{batch}, {records}, {self.rank}]")
        if x.device != self.b.device or keys.device != x.device or values.device != x.device:
            raise ValueError("x, keys, values, and VectorVeRA must be on the same device")
        if records == 0:
            residual = x.new_zeros((*x.shape[:-1], self.out_features))
            if not return_info:
                return residual
            shape = (*x.shape[:-1], 0)
            return residual, {
                "indices": torch.empty(shape, dtype=torch.long, device=x.device),
                "scores": torch.empty(shape, dtype=self.b.dtype, device=x.device),
                "weights": torch.empty(shape, dtype=self.b.dtype, device=x.device),
            }
        query = self.encode_query(x)
        normalized_keys = F.normalize(keys.to(dtype=query.dtype), dim=-1, eps=1e-12)
        similarities = torch.bmm(query, normalized_keys.transpose(1, 2))
        scores, indices = similarities.topk(min(self.top_k, records), dim=-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        bank_rows = torch.arange(batch, device=x.device)[:, None, None]
        selected_values = values.to(dtype=query.dtype)[bank_rows, indices]
        retrieved = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
        residual = self.delta_from_value(x, retrieved)
        if return_info:
            return residual, {"indices": indices, "scores": scores, "weights": weights}
        return residual


class CounterfactualBackend(ContextDistillationBackend):
    """Extend real student sampling to independent per-query tensor banks.

    Three-dimensional banks are sampled one row at a time through the parent's
    existing two-dimensional path. This avoids mismatching shrinking active
    batches with their banks when examples hit EOS at different times. Original
    bank tensor references are restored even on failure. Teacher scopes and
    prediction-position capture remain inherited and unchanged.

    ``last_sampling_cost`` records forward-call and input-position counts for
    the latest successful sampling call, not FLOPs or model-loading overhead.
    Independent-bank sampling is intentionally sequential; shared-bank sampling
    retains the parent's active-batch implementation and RNG behavior.
    """

    def _sampling_cost(self, questions, outputs, independent):
        prompt_lengths = [len(self.prompt_ids(question)) for question in questions]
        lengths = [len(tokens) for tokens in outputs]
        unpadded = sum(prompt * length + length * (length - 1) // 2
                       for prompt, length in zip(prompt_lengths, lengths))
        if independent:
            forward_calls, padded = sum(lengths), unpadded
        else:
            forward_calls = max(lengths)
            padded = 0
            for step in range(forward_calls):
                active = [prompt_lengths[row] + step for row, length in enumerate(lengths) if length > step]
                padded += len(active) * max(active)
        return dict(mode="independent_banks_sequential" if independent else "shared_bank_active_batch",
                    examples=len(questions), sampled_tokens=sum(lengths), forward_calls=forward_calls,
                    unpadded_input_tokens=unpadded, padded_input_tokens=padded)

    @torch.no_grad()
    def sample_student(self, questions: Sequence[str], max_new_tokens: int = 4,
                       temperature: float = 1., generator: torch.Generator | None = None
                       ) -> list[list[int]]:
        self.last_sampling_cost = None
        keys, values = self.vector_keys, self.vector_values
        if keys is None or keys.ndim != 3:
            outputs = super().sample_student(questions, max_new_tokens, temperature, generator)
            self.last_sampling_cost = self._sampling_cost(questions, outputs, independent=False)
            return outputs
        if isinstance(questions, str) or not questions:
            raise ValueError("questions must be a nonempty batch")
        if (keys.shape[0] != len(questions) or values is None or values.ndim != 3
                or values.shape[:2] != keys.shape[:2]):
            raise ValueError("Independent bank batches must match questions and record counts")
        outputs = []
        try:
            for row, question in enumerate(questions):
                self.vector_keys, self.vector_values = keys[row], values[row]
                outputs.extend(super().sample_student([question], max_new_tokens, temperature, generator))
        finally:
            self.vector_keys, self.vector_values = keys, values
        self.last_sampling_cost = self._sampling_cost(questions, outputs, independent=True)
        return outputs
