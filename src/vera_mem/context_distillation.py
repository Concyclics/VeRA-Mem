"""Context-privileged teacher and context-free, real-retrieval student helpers.

Both branches score the *same token IDs* at the corresponding causal answer
positions. The teacher runs the frozen, adapter-disabled backbone with optional
context; the student never receives that context. ``hidden`` is the backbone's
final normalized hidden state, after the adapted layer and remaining layers,
not the adapted layer's input (which cannot carry this layer's memory residual).
"""
from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .scaling_backend import ScalingQwenBackend


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor,
                      direction: str = "reverse", temperature: float = 1.,
                      reduction: str = "mean") -> Tensor:
    """Full-vocabulary float32 KL, multiplied by temperature squared.

    ``reverse`` is KL(student || teacher); ``forward`` is KL(teacher || student).
    Teacher logits are always detached. ``none`` returns one loss per token so
    callers can average each sequence first rather than overweight long ones.
    """
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have equal [tokens, vocabulary] shapes")
    if student_logits.numel() == 0:
        raise ValueError("Cannot distill an empty token or vocabulary dimension")
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    if direction == "reverse":
        loss = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
    elif direction == "forward":
        loss = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    else:
        raise ValueError("direction must be 'reverse' or 'forward'")
    return _reduce(loss * temperature ** 2, reduction)


def hidden_alignment_loss(student_hidden: Tensor, teacher_hidden: Tensor,
                          reduction: str = "mean") -> Tensor:
    """Final-hidden 1 - cosine similarity, without a trainable projection."""
    if student_hidden.ndim != 2 or student_hidden.shape != teacher_hidden.shape:
        raise ValueError("Student and teacher hidden states must have equal [tokens, hidden] shapes")
    if student_hidden.numel() == 0:
        raise ValueError("Cannot align empty hidden states")
    loss = 1 - F.cosine_similarity(student_hidden.float(), teacher_hidden.detach().float(), dim=-1)
    return _reduce(loss, reduction)


class ContextDistillationBackend(ScalingQwenBackend):
    """Use one frozen backbone for teacher and adapted student without leakage."""

    _TEACHER_TRANSIENT_FIELDS = (
        "capture_layer_input", "captured_input", "trace_retrieval", "retrieval_trace",
        "last_retrieval", "prefill_retrieval", "last_residual_ratio",
        "last_answer_residual_ratio", "last_answer_query_inputs", "last_answer_queries",
        "last_answer_batch_indices", "last_answer_token_counts", "last_answer_nll_sums",
        "_answer_capture_batch_indices", "_answer_capture_positions",
    )

    @contextlib.contextmanager
    def _teacher_forward(self):
        # Hooks also maintain diagnostic state in disabled mode. Restore those
        # references so a teacher call cannot overwrite a pending student graph
        # or append to the student's retrieval trace. Banks are never changed.
        missing = object()
        saved = {name: getattr(self, name, missing) for name in self._TEACHER_TRANSIENT_FIELDS}
        self.capture_layer_input = False
        self.trace_retrieval = False
        try:
            with self.disabled(), torch.no_grad():
                yield
        finally:
            for name, value in saved.items():
                if value is missing:
                    if hasattr(self, name):
                        delattr(self, name)
                else:
                    setattr(self, name, value)

    def forward_sequences(self, questions: Sequence[str], continuations: Sequence[Sequence[int]],
                          contexts: Sequence[str] | None = None,
                          teacher: bool = False) -> dict[str, Tensor]:
        """Score exact continuation IDs, never decode/re-encode or append EOS.

        Results are flattened in sequence order. ``positions`` contains the
        branch-specific absolute positions predicting each continuation token;
        these differ for teacher and student because only teacher has context.
        ``counts`` and ``batch_indices`` allow per-sequence normalization.
        Student outputs preserve autograd through actual sparse GPU-bank reads.
        Teacher outputs are detached and all prior backend state is restored.
        """
        if isinstance(questions, str) or isinstance(continuations, str):
            raise TypeError("questions and continuations must be batches")
        if not questions or len(questions) != len(continuations):
            raise ValueError("questions and continuations must have the same nonzero length")
        if contexts is not None:
            if not teacher:
                raise ValueError("Student must not receive teacher contexts")
            if isinstance(contexts, str) or len(contexts) != len(questions):
                raise ValueError("contexts must be a batch matching questions")
            if any(not isinstance(context, str) for context in contexts):
                raise TypeError("Each context must be a string")
        targets = [list(tokens) for tokens in continuations]
        if any(not tokens for tokens in targets):
            raise ValueError("Every continuation must contain at least one token")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
               for tokens in targets for token in tokens):
            raise ValueError("Continuation tokens must be nonnegative integer IDs")
        prompts = [self.prompt_ids(question, None if contexts is None else contexts[row])
                   for row, question in enumerate(questions)]
        if any(not prompt for prompt in prompts):
            raise ValueError("Every prompt must contain at least one token")
        ids, mask = self._right_pad([prompt + tokens for prompt, tokens in zip(prompts, targets)])
        counts = torch.tensor([len(tokens) for tokens in targets], device=self.device, dtype=torch.long)
        rows = torch.repeat_interleave(torch.arange(len(questions), device=self.device), counts)
        positions = torch.cat([
            torch.arange(len(prompt) - 1, len(prompt) - 1 + len(tokens), device=self.device)
            for prompt, tokens in zip(prompts, targets)
        ])
        scope = self._teacher_forward() if teacher else contextlib.nullcontext()
        with scope:
            old_rows = getattr(self, "_answer_capture_batch_indices", None)
            old_positions = getattr(self, "_answer_capture_positions", None)
            self._answer_capture_batch_indices = rows
            self._answer_capture_positions = positions
            self.last_answer_query_inputs = None
            self.last_answer_queries = None
            self.last_answer_batch_indices = rows
            self.last_answer_token_counts = counts
            self.last_answer_nll_sums = None
            try:
                hidden = self.model.model(input_ids=ids, attention_mask=mask,
                                          use_cache=False).last_hidden_state[rows, positions]
                logits = self.model.lm_head(hidden).float()
                if self.last_answer_query_inputs is None:
                    raise RuntimeError("Target layer hook did not capture prediction-position inputs")
                result = {"logits": logits, "hidden": hidden,
                          "batch_indices": rows, "positions": positions, "counts": counts,
                          "query_inputs": self.last_answer_query_inputs}
                return {key: value.detach() for key, value in result.items()} if teacher else result
            finally:
                self._answer_capture_batch_indices = old_rows
                self._answer_capture_positions = old_positions

    @torch.no_grad()
    def sample_student(self, questions: Sequence[str], max_new_tokens: int = 4,
                       temperature: float = 1., generator: torch.Generator | None = None
                       ) -> list[list[int]]:
        """Sample real student trajectories; EOS is kept and never invented.

        Each step batches only unfinished trajectories, right-pads their actual
        prefixes, and selects each row's final non-padding hidden state. Every
        step recomputes the real VeRA retrieval path without KV caching.
        ``temperature=0`` is greedy. A supplied generator must match model device.
        Global torch RNG remains externally controllable when it is omitted.
        Compared with sequential per-example sampling, the conditional sampling
        distribution is unchanged but RNG consumption order differs: the same
        seed need not produce the same trajectories as the sequential version.
        """
        if isinstance(questions, str) or not questions:
            raise ValueError("questions must be a nonempty batch")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.mode != "vector_vera" or self.vector_store is not None:
            raise ValueError("Student sampling requires vector_vera with the GPU tensor bank")
        if self.vector_keys is None or self.vector_values is None:
            raise ValueError("Student sampling requires populated vector_keys and vector_values")
        if self.oracle_values is not None or self.vector_override is not None:
            raise ValueError("Student sampling cannot use oracle values or retrieval overrides")
        eos_ids = set()
        for eos in (self.tokenizer.eos_token_id,
                    getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)):
            if eos is not None:
                eos_ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        prompts = [self.prompt_ids(question) for question in questions]
        old_rows = getattr(self, "_answer_capture_batch_indices", None)
        old_positions = getattr(self, "_answer_capture_positions", None)
        self._answer_capture_batch_indices = None
        self._answer_capture_positions = None
        outputs: list[list[int]] = [[] for _ in questions]
        active = list(range(len(questions)))
        self.prefill_retrieval = None
        try:
            for _ in range(max_new_tokens):
                ids, mask = self._right_pad([prompts[row] + outputs[row] for row in active])
                positions = mask.sum(dim=-1) - 1
                rows = torch.arange(len(active), device=self.device)
                hidden = self.model.model(input_ids=ids, attention_mask=mask,
                                          use_cache=False).last_hidden_state[rows, positions]
                logits = self.model.lm_head(hidden).float()
                if temperature == 0:
                    tokens = logits.argmax(dim=-1).tolist()
                else:
                    tokens = torch.multinomial(F.softmax(logits / temperature, dim=-1),
                                               1, generator=generator).squeeze(-1).tolist()
                remaining = []
                for row, token in zip(active, tokens):
                    outputs[row].append(token)
                    if token not in eos_ids:
                        remaining.append(row)
                active = remaining
                if not active:
                    break
            return outputs
        finally:
            self._answer_capture_batch_indices = old_rows
            self._answer_capture_positions = old_positions
