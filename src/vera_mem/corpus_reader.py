"""Corpus interface v1: learned Q/K/V, indexed blocks, and matched operators.

This intentionally does not modify the historical three-slot BlockVeRA.
The candidate index has fixed coordinates; its learned query projection and
the fine Q/K/V maps are distinct. Selected soft weights train fine addresses.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .corpus_features import project_input, projection

MODES = ("diagonal", "factor_mean_outer", "block_outer")


class FeatureBank:
    """Frozen feature payloads; token labels are exposed only to training losses."""
    def __init__(self, directory, device="cuda", reserve=8192):
        self.encoded = False
        directory = Path(directory)
        if not json.loads((directory / "progress.json").read_text())["complete"]:
            raise ValueError("A training bank requires a completed feature cache")
        k = np.load(directory / "keys.npy", mmap_mode="r")
        v = np.load(directory / "values.npy", mmap_mode="r")
        self.size = len(k)
        shape = (self.size + reserve, *k.shape[1:])
        self.keys = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.values = torch.empty_like(self.keys)
        self.masks = torch.zeros(shape[:2], dtype=torch.bool, device=device)
        self.labels = torch.zeros(shape[:2], dtype=torch.long, device=device)
        for start in range(0, self.size, 4096):
            end = min(start + 4096, self.size)
            self.keys[start:end] = torch.from_numpy(k[start:end].copy()).to(device)
            self.values[start:end] = torch.from_numpy(v[start:end].copy()).to(device)
        self.masks[:self.size] = torch.from_numpy(np.load(directory / "masks.npy")).to(device)
        self.labels[:self.size] = torch.from_numpy(np.load(directory / "labels.npy").astype(np.int64)).to(device)
        stats = np.load(directory / "statistics.npz")
        self.key_mean = torch.tensor(stats["key_mean"], device=device)
        self.value_mean = torch.tensor(stats["value_mean"], device=device)

    def append(self, keys, values, masks, labels=None):
        n = len(keys)
        if self.size + n > len(self.keys):
            raise ValueError("Reserved online capacity exhausted; create a larger bank explicitly")
        begin, end = self.size, self.size + n
        for destination, source in ((self.keys, keys), (self.values, values), (self.masks, masks)):
            destination[begin:end] = torch.as_tensor(source, device=destination.device)
        # Online writing never requires next-token labels or gold QA answers.
        if labels is not None:
            self.labels[begin:end] = torch.as_tensor(labels, device=self.labels.device)
        self.size = end
        return np.arange(begin, end, dtype=np.int64)


class FrozenBank:
    """Inference VDB: only low-rank K/V rows and masks, never gold token IDs.

    The separately stored ANN coarse vector is also computed from observed
    features. New payloads are written by the frozen shared K/V encoder.
    """
    def __init__(self, reader, features, reserve=8192):
        self.encoded = True
        self.size = 0
        self.reader = reader
        if any(p.requires_grad for p in reader.parameters()):
            raise ValueError("Freeze shared weights before constructing an inference bank")
        shape = (features.size + reserve, 8, reader.rank)
        self.keys = torch.empty(shape, dtype=torch.bfloat16, device=features.keys.device)
        self.values = torch.empty_like(self.keys)
        self.masks = torch.zeros(shape[:2], dtype=torch.bool, device=self.keys.device)
        for start in range(0, features.size, 2048):
            end = min(start + 2048, features.size)
            self.append(features.keys[start:end], features.values[start:end], features.masks[start:end])

    @torch.no_grad()
    def append(self, keys, values, masks):
        if any(p.requires_grad for p in self.reader.parameters()):
            raise ValueError("Online writer weights must remain frozen")
        n = len(keys)
        begin, end = self.size, self.size + n
        if end > len(self.keys):
            raise ValueError("Reserved online capacity exhausted")
        # Match the offline FP16 cache -> BF16 training feature contract.
        keys = torch.as_tensor(keys, device=self.keys.device).to(torch.float16).to(torch.bfloat16).float()
        values = torch.as_tensor(values, device=self.values.device).to(torch.float16).to(torch.bfloat16).float()
        self.keys[begin:end] = F.normalize(self.reader.K(keys - self.reader.key_mean), dim=-1)
        self.values[begin:end] = torch.tanh(self.reader.V(values - self.reader.value_mean))
        self.masks[begin:end] = torch.as_tensor(masks, device=self.masks.device)
        self.size = end
        return np.arange(begin, end, dtype=np.int64)


class CorpusReader(nn.Module):
    def __init__(self, in_features=9728, out_features=2560, dimension=512, rank=64,
                 coarse_dimension=128, mode="block_outer", seed=20261013, temperature=.1):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.rank, self.temperature = mode, rank, temperature
        self.configuration = dict(in_features=in_features, out_features=out_features,
                                  dimension=dimension, rank=rank, coarse_dimension=coarse_dimension,
                                  mode=mode, seed=seed, temperature=temperature)
        self.register_buffer("feature_projection", projection(in_features, dimension).to(torch.bfloat16))
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("A", torch.randn(rank, in_features, generator=generator) / math.sqrt(in_features))
        self.register_buffer("key_mean", torch.zeros(dimension))
        self.register_buffer("value_mean", torch.zeros(dimension))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.route = nn.Linear(dimension, coarse_dimension, bias=False)
            self.Q = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, rank))
            self.K = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, rank))
            self.V = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, rank))
            self.B = nn.Linear(rank, out_features, bias=False)
            self.u = nn.Linear(rank, rank)
            self.w = nn.Linear(rank, rank, bias=False)
        self.gain = nn.Parameter(torch.zeros(out_features))
        with torch.no_grad():
            self.route.weight.zero_()
            self.route.weight[:, :coarse_dimension].copy_(torch.eye(coarse_dimension))
            self.u.weight.copy_(torch.eye(rank))
            self.w.weight.copy_(torch.eye(rank))
            # Match all shared parameters, independent of the selected operator.
            self.K.load_state_dict(self.Q.state_dict())
        if mode == "diagonal":
            self.u.requires_grad_(False)
            self.w.requires_grad_(False)

    def set_statistics(self, bank):
        self.key_mean.copy_(bank.key_mean)
        self.value_mean.copy_(bank.value_mean)

    def query_features(self, x):
        return project_input(x, self.feature_projection)

    def route_queries(self, features):
        return F.normalize(self.route(features - self.key_mean), dim=-1)

    def latent(self, x, values, weights):
        """Identical row weights and nonlinear factors isolate the outer rank."""
        z = F.linear(x.float(), self.A)
        mean = (values * weights[..., None]).sum(-2)
        diagonal = z * mean
        if self.mode == "diagonal":
            return diagonal
        u = F.normalize(self.u(values), dim=-1, eps=1e-6)
        w = self.w(values)
        if self.mode == "factor_mean_outer":
            mean_u = (u * weights[..., None]).sum(-2)
            mean_w = (w * weights[..., None]).sum(-2)
            outer = mean_w * (mean_u * z).sum(-1, keepdim=True)
        else:
            outer = (w * (u * z[..., None, :]).sum(-1, keepdim=True) * weights[..., None]).sum(-2)
        return diagonal + outer

    def read(self, x, features, candidates, bank, *, top_blocks=4, shuffle_values=False):
        """Return delta and sparse training diagnostics; candidates are [N,C]."""
        n, count = candidates.shape
        safe = candidates.clamp_min(0)
        valid = (candidates >= 0)[..., None] & bank.masks[safe]
        query = F.normalize(self.Q(features - self.key_mean), dim=-1)
        # Encode each distinct candidate once, keeping gradients to current K.
        unique, inverse = torch.unique(safe, sorted=False, return_inverse=True)
        unique_keys = (bank.keys[unique].float() if getattr(bank, "encoded", False)
                       else F.normalize(self.K(bank.keys[unique].float() - self.key_mean), dim=-1))
        keys = unique_keys[inverse].reshape(n, count, 8, self.rank)
        scores = (query[:, None, None] * keys).sum(-1) / self.temperature
        scores = scores.masked_fill(~valid, -1e4)
        group_scores = torch.logsumexp(scores, -1)
        selection = group_scores.topk(min(top_blocks, count), dim=-1).indices
        row = torch.arange(n, device=x.device)[:, None]
        selected = safe[row, selection]
        selected_scores = scores[row, selection].reshape(n, -1)
        selected_valid = valid[row, selection].reshape(n, -1)
        weights = F.softmax(selected_scores, dim=-1) * selected_valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        if shuffle_values:
            # Keep keys, IDs, and masks; permute complete value blocks.
            selected_values = (selected + 7919) % bank.size
        else:
            selected_values = selected
        vu, vi = torch.unique(selected_values, sorted=False, return_inverse=True)
        values = (bank.values[vu].float() if getattr(bank, "encoded", False)
                  else torch.tanh(self.V(bank.values[vu].float() - self.value_mean)))
        values = values[vi].reshape(n, -1, self.rank)
        latent = self.latent(x, values, weights)
        delta = (self.B(latent) * self.gain).to(x.dtype)
        return delta, {"scores": scores, "valid": valid, "safe_candidates": safe,
                       "selected": selected, "weights": weights, "group_scores": group_scores}

    def retrieval_supervision(self, features, info, bank, *, positive=None):
        """Sparse causal teacher and address losses; never used for inference."""
        if getattr(bank, "encoded", False):
            raise ValueError("Inference stores contain no teacher labels or raw feature targets")
        safe, valid = info["safe_candidates"], info["valid"]
        with torch.no_grad():
            teacher_query = F.normalize(features - self.key_mean, dim=-1)
            teacher_key = F.normalize(bank.keys[safe].float() - self.key_mean, dim=-1)
            teacher_scores = (teacher_query[:, None, None] * teacher_key).sum(-1) / .1
            teacher_scores.masked_fill_(~valid, -1e4)
            teacher_probability = F.softmax(teacher_scores.flatten(1), -1) * valid.flatten(1)
            teacher_probability /= teacher_probability.sum(-1, keepdim=True).clamp_min(1e-12)
            probability, slots = teacher_probability.topk(min(16, teacher_probability.shape[-1]), dim=-1)
            probability /= probability.sum(-1, keepdim=True).clamp_min(1e-12)
            token_ids = bank.labels[safe].flatten(1).gather(1, slots)
        student_logp = F.log_softmax(info["scores"].flatten(1), -1)
        address = -(teacher_probability * student_logp).sum(-1)
        # Route learning is explicit; gradients cannot pass through CPU ANN.
        coarse = (bank.keys[safe, :, :128].float() - self.key_mean[:128])
        coarse = F.normalize((coarse * valid[..., None]).sum(2) / valid.sum(2)[..., None].clamp_min(1), dim=-1)
        route_score = (self.route_queries(features)[:, None] * coarse).sum(-1) / .1
        route_score.masked_fill_(~valid.any(-1), -1e4)
        route_target = teacher_probability.reshape_as(valid).sum(-1)
        route_loss = -(route_target * F.log_softmax(route_score, -1)).sum(-1)
        if positive is not None:
            is_positive = (safe == positive[:, None]) & valid.any(-1)
            if not is_positive.any(-1).all():
                raise ValueError("Positive address supervision requires an explicit candidate")
            positive_index = is_positive.float().argmax(-1)
            address = address + F.cross_entropy(info["group_scores"], positive_index, reduction="none")
            route_loss = route_loss + F.cross_entropy(route_score, positive_index, reduction="none")
        return {"token_ids": token_ids, "probabilities": probability,
                "address": address, "route": route_loss}


def sparse_cross_entropy(logits, token_ids, probabilities):
    """CE to sparse vocabulary targets, aggregating duplicate IDs implicitly.

    This is KL plus the teacher's constant entropy. It has identical student
    gradients; logs call it sparse CE, never the numerical value of KL.
    """
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return torch.logsumexp(logits, -1) * probabilities.sum(-1) - (
        logits.gather(-1, token_ids) * probabilities).sum(-1)
