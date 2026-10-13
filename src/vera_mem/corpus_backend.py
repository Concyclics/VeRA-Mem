"""Frozen-prefix reuse without changing the actual down-projection injection.

The standard Transformers forward constructs masks/positions and runs through
layer 20. A private exception stops it there. Each adapter branch reconstructs
residual + (down_output + delta), preserving BF16 addition order, then runs the
remaining frozen layers. Prefix states are shared only for identical inputs.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .corpus_reader import CorpusReader


class _PrefixFinished(Exception):
    pass


class CorpusBackbone:
    def __init__(self, model_path, layer=20):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
        self.layer = layer
        types = getattr(self.model.config, "layer_types", ["full_attention"] * len(self.model.model.layers))
        if set(types) != {"full_attention"}:
            raise ValueError("Prefix reuse is qualified only for uniform full-attention layers")
        self.target = self.model.model.layers[layer].mlp.down_proj

    @torch.no_grad()
    def prefix(self, ids, attention_mask, past=None):
        captured = {}
        layer = self.model.model.layers[self.layer]
        def residual(module, inputs):
            captured["residual"] = inputs[0].detach()
        def down(module, inputs, output):
            captured["x"], captured["down"] = inputs[0].detach(), output.detach()
        def stop(module, inputs, kwargs, output):
            captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()
            captured["kwargs"] = kwargs
            raise _PrefixFinished()
        handles = [layer.post_attention_layernorm.register_forward_pre_hook(residual),
                   self.target.register_forward_hook(down),
                   layer.register_forward_hook(stop, with_kwargs=True)]
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        try:
            self.model.model(input_ids=ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past, use_cache=past is not None)
        except _PrefixFinished:
            pass
        finally:
            for handle in handles:
                handle.remove()
        if "hidden" not in captured:
            raise RuntimeError("Frozen prefix did not reach the declared injection layer")
        return captured

    def suffix(self, prefix, delta=None):
        hidden = prefix["hidden"] if delta is None else prefix["residual"] + (prefix["down"] + delta)
        for layer in self.model.model.layers[self.layer + 1:]:
            hidden = layer(hidden, **prefix["kwargs"])
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return self.model.model.norm(hidden)

    @torch.no_grad()
    def qualify_prefix(self):
        ids = torch.tensor([[151643, 100, 200, 300, 400, 500],
                            [151643, 600, 700, 800, 151643, 151643]], device="cuda")
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], device="cuda")
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        expected = self.model.model(input_ids=ids, attention_mask=mask,
                                    position_ids=positions, use_cache=False).last_hidden_state
        prefix = self.prefix(ids, mask)
        actual = self.suffix(prefix)
        zero = self.suffix(prefix, torch.zeros_like(prefix["down"]))
        error = float((actual - expected).abs().max())
        zero_error = float((zero - expected).abs().max())
        if error != 0 or zero_error != 0:
            raise RuntimeError(f"Prefix decomposition changed frozen outputs: {error}, {zero_error}")
        return {"max_hidden_error": error, "zero_injection_error": zero_error,
                "backbone_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}


class CorpusLoRA(nn.Module):
    def __init__(self, in_features=9728, out_features=2560, rank=96, seed=20261013):
        super().__init__()
        self.configuration = dict(in_features=in_features, out_features=out_features, rank=rank, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down = nn.Linear(in_features, rank, bias=False)
            self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.float())).to(x.dtype)


def adapter_delta(reader, prefix, bank, index, *, articles=None, positive=None,
                  force_positive=False, supervision=False, empty=False, shuffle_values=False,
                  oracle=False, query_chunk=512):
    """Index search once per position; no answer tokens enter this query path."""
    x = prefix["x"]
    if reader is None or empty:
        return None, {}
    if isinstance(reader, CorpusLoRA):
        return reader(x), {}
    flat = x.reshape(-1, x.shape[-1])
    features = reader.query_features(flat)
    route = reader.route_queries(features)
    exclusions = None if articles is None else np.repeat(np.asarray(articles), x.shape[1])
    positives = None if positive is None else np.repeat(np.asarray(positive), x.shape[1])
    began = time.monotonic()
    if oracle:
        if positives is None:
            raise ValueError("An oracle read needs declared observation IDs")
        candidate_ids = np.full((len(flat), 32), -1, dtype=np.int64)
        candidate_ids[:, 0] = positives
    else:
        candidate_ids = index.search(route.detach().float().cpu().numpy(), 32, exclusions,
                                     forced=positives if force_positive else None)
    candidates = torch.from_numpy(candidate_ids).to(x.device)
    auxiliary = {"index_seconds": time.monotonic() - began, "candidate_ids": candidate_ids}
    results, pieces = [], []
    for start in range(0, len(flat), query_chunk):
        end = min(len(flat), start + query_chunk)
        delta, info = reader.read(flat[start:end], features[start:end], candidates[start:end], bank,
                                  shuffle_values=shuffle_values)
        results.append(delta)
        if supervision:
            positive_tensor = (torch.as_tensor(positives[start:end], device=x.device)
                               if positives is not None and force_positive else None)
            pieces.append(reader.retrieval_supervision(features[start:end], info, bank, positive=positive_tensor))
        auxiliary.setdefault("selected", []).append(info["selected"].detach())
    if pieces:
        auxiliary.update({key: torch.cat([part[key] for part in pieces]) for key in pieces[0]})
    auxiliary["selected"] = torch.cat(auxiliary["selected"]).reshape(*x.shape[:2], -1)
    return torch.cat(results).reshape(*x.shape[:2], -1), auxiliary


def coarse_teacher_loss(student_logits, teacher_ids, teacher_probs):
    """Exact forward cross-entropy after grouping the teacher tail into 'other'."""
    logp = F.log_softmax(student_logits.float(), -1)
    chosen = logp.gather(-1, teacher_ids)
    tail_mass = (1 - teacher_probs.sum(-1)).clamp_min(0)
    student_tail_logp = torch.log1p(-chosen.exp().sum(-1).clamp(max=1 - 1e-7))
    return -(chosen * teacher_probs).sum(-1) - tail_mass * student_tail_logp


def answer_batch(tokenizer, episodes, *, style=0, heldout=False, context=False, device="cuda"):
    """Teacher-forced answer labels, excluding prompt and padding positions."""
    from .corpus_episodes import prompt_ids
    rows = []
    for episode in episodes:
        prompt = prompt_ids(tokenizer, episode, style=style, heldout=heldout,
                            context=episode["support"] if context else None)
        answer = tokenizer.encode(episode["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        rows.append((prompt, answer))
    size = max(len(p) + len(a) - 1 for p, a in rows)
    ids = np.full((len(rows), size), tokenizer.pad_token_id, dtype=np.int64)
    labels = np.full_like(ids, -100)
    mask = np.zeros_like(ids)
    for i, (p, a) in enumerate(rows):
        full = p + a
        ids[i, :len(full) - 1] = full[:-1]
        labels[i, len(p) - 1:len(full) - 1] = a
        mask[i, :len(full) - 1] = 1
    return tuple(torch.from_numpy(t).to(device) for t in (ids, labels, mask))
