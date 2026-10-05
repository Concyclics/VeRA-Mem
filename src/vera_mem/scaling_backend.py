"""Batched Qwen helpers for scaling experiments, preserving the original backend.

Teacher-forced inputs collected here are training features, not label-free
retrieval evaluation. Evaluation must continue to query the actual prompt and
generated prefix without exposing future answer tokens.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .backend import QwenBackend


class ScalingQwenBackend(QwenBackend):
    """Frozen Qwen with batched feature extraction and answer-only projection.

    ``oracle_values`` is an optional differentiable ``[batch, rank]`` tensor
    used only for diagnostic/curriculum training in ``vector_vera`` mode.
    It takes precedence over the normal store/tensor bank until explicitly
    cleared by the caller. Every token still computes its own query.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_values: Tensor | None = None
        self.last_answer_query_inputs: Tensor | None = None
        self.last_answer_queries: Tensor | None = None
        self.last_answer_batch_indices: Tensor | None = None
        self.last_answer_token_counts: Tensor | None = None
        self.last_answer_nll_sums: Tensor | None = None
        self._answer_capture_batch_indices: Tensor | None = None
        self._answer_capture_positions: Tensor | None = None
        self.trace_retrieval = False
        self.retrieval_trace: list[dict] = []
        self.last_residual_ratio: Tensor | None = None
        self.last_answer_residual_ratio: Tensor | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _right_pad(self, sequences: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
        if not sequences or any(len(ids) == 0 for ids in sequences):
            raise ValueError("Expected a nonempty batch of nonempty token sequences")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad or EOS token")
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, values in enumerate(sequences):
            ids[row, :len(values)] = torch.as_tensor(values, dtype=torch.long, device=self.device)
            mask[row, :len(values)] = 1
        return ids, mask

    @torch.no_grad()
    def layer_features(self, texts: Sequence[str], batch_size: int = 32) -> Tensor:
        """Return actual down-projection inputs at each final prompt token.

        Result is detached CPU float32 ``[N, in_features]`` in input order.
        Right padding is masked and each row gathers its own final position.
        The temporary pre-hook is always removed, including on failure; this
        method does not toggle or reuse the legacy ``capture_layer_input`` flag.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not texts:
            return torch.empty((0, self.target.in_features), dtype=torch.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            sequences = [self.prompt_ids(text) for text in chunk]
            ids, mask = self._right_pad(sequences)
            positions = mask.sum(dim=1) - 1
            rows = torch.arange(len(chunk), device=self.device)
            captured: list[Tensor] = []

            def capture(_module, args):
                captured.append(args[0][rows, positions].detach().float().cpu())

            handle = self.target.register_forward_pre_hook(capture)
            try:
                with self.disabled():
                    self.model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError(f"Expected one target-layer call, got {len(captured)}")
            batches.append(captured[0])
            for text, feature in zip(chunk, captured[0]):
                self.layer_feature_cache[text] = feature.clone()
        return torch.cat(batches, dim=0)

    def _inject(self, module, args, output):
        rows = getattr(self, "_answer_capture_batch_indices", None)
        positions = getattr(self, "_answer_capture_positions", None)
        if rows is not None and positions is not None:
            # Preserve this graph: an auxiliary addressing loss must be able to
            # train the query encoder on the actual prediction-position inputs.
            self.last_answer_query_inputs = args[0][rows, positions]
        oracle = getattr(self, "oracle_values", None)
        if self.mode != "vector_vera" or oracle is None:
            result = super()._inject(module, args, output)
            self._record_forward_diagnostics(output, result, rows, positions)
            return result
        if oracle.ndim != 2 or oracle.shape != (args[0].shape[0], self.vector_vera.rank):
            raise ValueError("oracle_values must have shape [batch, rank]")
        if self.capture_layer_input:
            self.captured_input = args[0][0, -1].detach().float().cpu()
        query = self.vector_vera.encode_query(args[0])
        if rows is not None and positions is not None:
            self.last_answer_queries = query[rows, positions]
        residual = self.vector_vera.delta_from_value(
            args[0], oracle.to(output.device).unsqueeze(1))
        self.last_retrieval = {
            "indices": torch.empty((*query.shape[:-1], 0), dtype=torch.long),
        }
        if self.prefill_retrieval is None:
            self.prefill_retrieval = self.last_retrieval
        result = output + residual
        self._record_forward_diagnostics(output, result, rows, positions)
        return result

    @torch.no_grad()
    def _record_forward_diagnostics(self, output, result, rows, positions):
        """Measure the realized residual and, optionally, generation addresses.

        The RMS ratio is per token, over output channels. It uses the
        post-addition result minus the original output in float32. With a BF16
        backbone it therefore measures the approximate residual *after BF16
        rounding*, not the unrounded mathematical adapter branch. The
        denominator is clamped to 1e-12 for zero backbone outputs.

        Trace entries retain only the last input-position indices per batch
        row: the prefill position predicting the first generated token, then
        each cached decode position. They contain no scores or large tensors.
        The caller enables this only during free generation, never scoring.
        """
        base = output.detach().float()
        realized = result.detach().float() - base
        self.last_residual_ratio = (
            realized.square().mean(-1).sqrt()
            / base.square().mean(-1).sqrt().clamp_min(1e-12)
        )
        if rows is not None and positions is not None:
            self.last_answer_residual_ratio = self.last_residual_ratio[rows, positions]
        if not getattr(self, "trace_retrieval", False):
            return
        if not hasattr(self, "retrieval_trace"):
            self.retrieval_trace = []
        info = self.last_retrieval if self.mode == "vector_vera" else None
        indices = info.get("indices") if info else None
        if indices is None:
            selected = torch.empty((output.shape[0], 0), dtype=torch.long)
        else:
            selected = indices[:, -1].detach().cpu().clone()
        self.retrieval_trace.append({
            "phase": "prefill" if not self.retrieval_trace else "decode",
            "indices": selected,
        })

    def batched_loss(self, questions: Sequence[str], answers: Sequence[str],
                     include_eos: bool = True) -> Tensor:
        """Mean of per-example mean answer NLLs, with strict causal shifting.

        Only hidden states that predict answer tokens are projected through the
        vocabulary head. This supports unequal prompt/answer lengths without
        projecting all padded prompt positions. ``include_eos`` also determines
        whether EOS appears in the saved counts and prediction-position inputs.

        After success, ``last_answer_query_inputs`` has shape ``[sum(counts),
        in_features]``; ``last_answer_batch_indices`` maps those rows to examples.
        Counts and NLL sums remain device tensors. NLL sums are detached logging
        statistics; query inputs retain their graph for an auxiliary loss.
        """
        if isinstance(questions, str) or isinstance(answers, str):
            raise TypeError("questions and answers must be batches, not strings")
        if not questions or len(questions) != len(answers):
            raise ValueError("questions and answers must have the same nonzero length")
        prompts = [self.prompt_ids(question) for question in questions]
        targets = [list(self.tokenizer.encode(answer, add_special_tokens=False)) for answer in answers]
        if include_eos:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("include_eos requires an EOS token")
            for target in targets:
                target.append(self.tokenizer.eos_token_id)
        if any(not prompt for prompt in prompts):
            raise ValueError("Each prompt must contain at least one token")
        if any(not target for target in targets):
            raise ValueError("Each answer must contain at least one supervised token")
        sequences = [prompt + target for prompt, target in zip(prompts, targets)]
        ids, attention_mask = self._right_pad(sequences)
        counts = torch.tensor([len(target) for target in targets], dtype=torch.long, device=self.device)
        rows = torch.repeat_interleave(torch.arange(len(questions), device=self.device), counts)
        positions = torch.cat([
            torch.arange(len(prompt) - 1, len(prompt) - 1 + len(target), device=self.device)
            for prompt, target in zip(prompts, targets)
        ])
        labels = torch.tensor([token for target in targets for token in target],
                              dtype=torch.long, device=self.device)
        self.last_answer_query_inputs = None
        self.last_answer_queries = None
        self.last_answer_residual_ratio = None
        self.last_answer_nll_sums = None
        self.last_answer_token_counts = counts
        self.last_answer_batch_indices = rows
        self._answer_capture_batch_indices = rows
        self._answer_capture_positions = positions
        try:
            hidden = self.model.model(input_ids=ids, attention_mask=attention_mask,
                                      use_cache=False).last_hidden_state
            logits = self.model.lm_head(hidden[rows, positions]).float()
            losses = F.cross_entropy(logits, labels, reduction="none")
            sums = torch.zeros(len(questions), dtype=losses.dtype, device=losses.device)
            sums = sums.index_add(0, rows, losses)
            self.last_answer_nll_sums = sums.detach()
            return (sums / counts).mean()
        finally:
            self._answer_capture_batch_indices = None
            self._answer_capture_positions = None
