"""Frozen Qwen backend with explicit, request-local adapter state.

Routing features are ALWAYS extracted with adaptation disabled. The answer label
is consumed only by teacher-forced scoring or supervised writes, never routing.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AnswerScore:
    nll_sum: float
    tokens: int


class QwenBackend:
    def __init__(self, model_path: str, layer: int = 20, max_input_tokens: int = 1024):
        if not torch.cuda.is_available():
            raise RuntimeError("This runner requires an explicitly selected CUDA device.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda")
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.max_input_tokens = max_input_tokens
        self.layer = layer
        self.target = self.model.model.layers[layer].mlp.down_proj
        self.mode = "none"
        self.bank = None
        self.reader = None
        self.slot = 0
        self.value = None
        self.injection_start = 0
        self.feature_cache = {}
        self.layer_feature_cache = {}
        self.capture_layer_input = False
        self.captured_input = None
        self.vector_vera = None
        self.vector_keys = None
        self.vector_values = None
        self.vector_store = None
        self.vector_override = None
        self.last_retrieval = None
        self.prefill_retrieval = None
        self.hook = self.target.register_forward_hook(self._inject)

    def _inject(self, module, args, output):
        if self.capture_layer_input:
            self.captured_input = args[0][0, -1].detach().float().cpu()
        if self.mode == "vector_vera":
            if self.vector_store is not None:
                query = self.vector_vera.encode_query(args[0]).detach().cpu()
                if self.vector_override is None:
                    info = self.vector_store.search(query)
                    mixed = info["mixed_value"].to(output.device)
                else:
                    # Diagnostic only: force known evidence or zero memory.
                    mixed = self.vector_override.to(output.device)
                    info = {"indices": torch.empty((*query.shape[:-1], 0), dtype=torch.long)}
                residual = self.vector_vera.delta_from_value(args[0], mixed)
            else:
                residual, info = self.vector_vera(args[0], self.vector_keys, self.vector_values, return_info=True)
            self.last_retrieval = {k: v.detach().cpu() for k, v in info.items() if isinstance(v, torch.Tensor)}
            if self.prefill_retrieval is None:
                self.prefill_retrieval = self.last_retrieval
            return output + residual
        if self.mode == "lora":
            return output + self.bank(args[0], self.slot)
        if self.mode == "latent" and self.value is not None:
            residual = self.reader(self.value).to(output.dtype).view(1, 1, -1)
            # Prefill injects only at the final prompt position and answer tokens;
            # cached decoding has length one and receives the same fixed memory.
            if output.shape[1] == 1:
                return output + residual
            positions = torch.arange(output.shape[1], device=output.device)
            mask = (positions >= self.injection_start).view(1, -1, 1)
            return output + mask * residual
        return output

    @contextlib.contextmanager
    def disabled(self):
        old = self.mode
        self.mode = "none"
        try:
            yield
        finally:
            self.mode = old

    def prompt_ids(self, question: str, context: str | None = None):
        content = question if context is None else (
            "Memory records (use them when relevant):\n" + context + "\n\n" + question)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the requested answer format."},
            {"role": "user", "content": content},
        ]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"Input has {len(ids)} tokens, exceeds {self.max_input_tokens}; no silent truncation")
        return ids

    def feature(self, text: str):
        """Frozen final hidden state; caller passes question OR observed support."""
        if text not in self.feature_cache:
            ids = torch.tensor([self.prompt_ids(text)], device="cuda")
            with self.disabled(), torch.no_grad():
                h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
                feature = h.float().cpu()
            self.feature_cache[text] = feature
        return self.feature_cache[text].clone()

    def layer_feature(self, text: str):
        """Pre-adapter down_proj input at final prompt token, never answer tokens.

        A single adapted layer ensures its incoming state is independent of its
        own adapter parameters. Multi-layer adaptation needs encoder versioning.
        """
        if text not in self.layer_feature_cache:
            ids = torch.tensor([self.prompt_ids(text)], device="cuda")
            self.capture_layer_input = True
            try:
                with self.disabled(), torch.no_grad():
                    self.model.model(input_ids=ids, use_cache=False)
                self.layer_feature_cache[text] = self.captured_input.clone()
            finally:
                self.capture_layer_input = False
        return self.layer_feature_cache[text].clone()

    def loss(self, question: str, answer: str, context: str | None = None,
             include_eos: bool = True):
        prompt = self.prompt_ids(question, context)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if include_eos:
            answer_ids.append(self.tokenizer.eos_token_id)
        if not answer_ids:
            raise ValueError("Empty answer token sequence")
        ids = torch.tensor([prompt + answer_ids], device="cuda")
        self.injection_start = len(prompt) - 1
        # Only answer-position logits are needed. This avoids allocating a full
        # vocabulary projection for every context token during adaptation.
        logits = self.model(input_ids=ids, use_cache=False,
                            logits_to_keep=len(answer_ids) + 1).logits[0, :-1].float()
        labels = torch.tensor(answer_ids, device="cuda")
        losses = F.cross_entropy(logits, labels, reduction="none")
        return losses.mean(), losses.sum(), len(answer_ids)

    @torch.no_grad()
    def score(self, question, answer, context=None):
        _, total, count = self.loss(question, answer, context, include_eos=False)
        return AnswerScore(float(total), count)

    @torch.no_grad()
    def generate(self, question, context=None, max_new_tokens=12):
        prompt = self.prompt_ids(question, context)
        ids = torch.tensor([prompt], device="cuda")
        self.injection_start = len(prompt) - 1
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = output[0, len(prompt):]
        return self.tokenizer.decode(tokens, skip_special_tokens=True), len(tokens), elapsed

    @torch.no_grad()
    def choose(self, question, context=None):
        """Raw summed conditional log likelihood of each letter, without EOS."""
        scores = [self.score(question, x, context).nll_sum for x in "ABCD"]
        return "ABCD"[min(range(4), key=scores.__getitem__)], scores
