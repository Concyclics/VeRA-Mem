"""Explicit whole-block CPU injection; never routes through mixed_value."""
from __future__ import annotations

import torch

from .block_vera import BlockVeRA
from .counterfactual_backend import CounterfactualBackend


class BlockBackend(CounterfactualBackend):
    """Preserve actual-position capture, teacher scopes and route diagnostics."""

    def _inject(self, module, args, output):
        if self.mode != "vector_vera":
            return super()._inject(module, args, output)
        reader = self.vector_vera
        if not isinstance(reader, BlockVeRA):
            raise TypeError("BlockBackend requires a BlockVeRA module")
        if getattr(self, "oracle_values", None) is not None or getattr(self, "vector_override", None) is not None:
            raise ValueError("Rank-vector overrides cannot represent a full value block")
        x = args[0]
        rows = getattr(self, "_answer_capture_batch_indices", None)
        positions = getattr(self, "_answer_capture_positions", None)
        if rows is not None and positions is not None:
            self.last_answer_query_inputs = x[rows, positions]
        if self.capture_layer_input:
            self.captured_input = x[0, -1].detach().float().cpu()
        if self.vector_store is None:
            residual, info = reader(x, self.vector_keys, self.vector_values, return_info=True)
        else:
            residual, info = reader.cpu_delta(x, self.vector_store, return_info=True)
        # Large selected matrices stay in the CPU store trace, not duplicated
        # across every training forward. The backend trace needs only addresses.
        self.last_retrieval = {k: v.detach().cpu() for k, v in info.items()
                               if isinstance(v, torch.Tensor) and k not in ("selected_values", "queries")}
        if self.prefill_retrieval is None:
            self.prefill_retrieval = self.last_retrieval
        result = output + residual
        self._record_forward_diagnostics(output, result, rows, positions)
        return result
