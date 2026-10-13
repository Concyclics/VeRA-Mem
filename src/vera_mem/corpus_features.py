"""Frozen, support-only feature cache for the version-1 corpus interface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .corpus_data import Corpus, atomic_json, sha256


def projection(width=9728, dimension=512, seed=20261013):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dimension, width, generator=generator) / math.sqrt(width)


def project_input(x, matrix):
    # Absolute activation magnitude is retained in the independent VeRA A path.
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return F.linear(normalized.to(matrix.dtype), matrix).float()


def pool_features(features, lengths, tokens, slots=8, span=16):
    """Observed V spans and causal K states predicting their final observed token.

    Position zero is BOS. The key for a span ending at observed token e is
    state e-1; the value includes e. No query, gold answer, or future span is
    accepted by this writer API.
    """
    batch, _, dim = features.shape
    lengths = torch.as_tensor(lengths, device=features.device)[:, None]
    starts = torch.arange(slots, device=features.device)[None, :] * span + 1
    ends = torch.minimum(starts + span, lengths + 1)
    starts = torch.minimum(starts, lengths + 1)
    mask = ends > starts
    rows = torch.arange(batch, device=features.device)[:, None]
    keys = features[rows, (ends - 2).clamp_min(0)] * mask[..., None]
    cumulative = F.pad(features.cumsum(1), (0, 0, 1, 0))
    values = (cumulative[rows, ends] - cumulative[rows, starts]) / (ends - starts).clamp_min(1)[..., None]
    labels = tokens[rows, (ends - 1).clamp_min(0)] * mask
    return keys, values, mask, labels


class SupportEncoder:
    """Only the frozen prefix is required to capture the actual layer input."""
    def __init__(self, model_path, layer=20, dimension=512, seed=20261013):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                              dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model.requires_grad_(False).eval()
        self.model.layers = nn.ModuleList(list(self.model.layers)[:layer + 1])
        self.model.to("cuda")
        self.target = self.model.layers[layer].mlp.down_proj
        self.matrix = projection(self.target.in_features, dimension, seed).to("cuda", torch.bfloat16)
        self.captured = None
        def capture(module, inputs, output):
            self.captured = project_input(inputs[0], self.matrix)
        self.handle = self.target.register_forward_hook(capture)

    @torch.inference_mode()
    def encode(self, observations, boundary, pad):
        if any(not 0 < len(row) <= 128 for row in observations):
            raise ValueError("Writer accepts nonempty observed chunks of at most 128 tokens")
        length = max(len(row) for row in observations) + 1
        cpu_ids = np.full((len(observations), length), pad, dtype=np.int64)
        cpu_mask = np.zeros_like(cpu_ids)
        for i, row in enumerate(observations):
            cpu_ids[i, 0] = boundary
            cpu_ids[i, 1:len(row) + 1] = row
            cpu_mask[i, :len(row) + 1] = 1
        ids = torch.from_numpy(cpu_ids).to("cuda")
        mask = torch.from_numpy(cpu_mask).to("cuda")
        self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        result = pool_features(self.captured, [len(row) for row in observations], ids)
        self.captured = None
        return tuple(t.detach().cpu().numpy() for t in result)


def prepare_features(data, output, model, split="train", batch_size=96, dimension=512):
    corpus = Corpus(data, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "progress.json"
    provenance = {"schema": "vera-corpus-features-v1", "split": split,
                  "data_manifest_sha256": sha256(Path(data) / "manifest.json"),
                  "model_config_sha256": sha256(Path(model) / "config.json"),
                  "dimension": dimension, "layer": 20, "projection_seed": 20261013,
                  "source_sha256": sha256(__file__), "batch_size": batch_size,
                  "chunks": len(corpus.chunks)}
    done = 0
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old["provenance"] != provenance:
            raise ValueError("Cannot resume a feature cache with changed provenance")
        done = old["chunks_done"]
        if old.get("complete"):
            print("Feature cache is already complete", flush=True)
            return
    count = len(corpus.chunks)
    def array(name, dtype, shape):
        path = output / (name + ".npy")
        return np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                        dtype=dtype, shape=shape)
    keys = array("keys", np.float16, (count, 8, dimension))
    values = array("values", np.float16, keys.shape)
    masks = array("masks", np.bool_, (count, 8))
    labels = array("labels", np.int32, (count, 8))
    encoder = SupportEncoder(model, dimension=dimension)
    began = time.monotonic()
    start_done = done
    for start in range(done, count, batch_size):
        end = min(count, start + batch_size)
        k, v, m, y = encoder.encode([corpus.chunk(i) for i in range(start, end)],
                                    corpus.manifest["boundary_token_id"], corpus.manifest["pad_token_id"])
        keys[start:end], values[start:end], masks[start:end], labels[start:end] = k, v, m, y
        if (start // batch_size) % 40 == 0 or end == count:
            for item in (keys, values, masks, labels):
                item.flush()
            state = {"provenance": provenance, "chunks_done": end, "complete": False,
                     "current_session_seconds": time.monotonic() - began,
                     "current_session_chunks": end - start_done,
                     "chunks_per_second": (end - start_done) / (time.monotonic() - began),
                     "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
            atomic_json(state_path, state)
            print(json.dumps(state), flush=True)
    # Fit only on the training split; held-out caches reuse these statistics.
    if split == "train":
        ksum = np.zeros(dimension, dtype=np.float64)
        vsum = np.zeros(dimension, dtype=np.float64)
        n = 0
        for start in range(0, count, 2048):
            valid = masks[start:start + 2048]
            ksum += keys[start:start + 2048][valid].astype(np.float64).sum(0)
            vsum += values[start:start + 2048][valid].astype(np.float64).sum(0)
            n += int(valid.sum())
        np.savez(output / "statistics.npz", key_mean=(ksum / n).astype(np.float32),
                 value_mean=(vsum / n).astype(np.float32), fitted_slots=n)
    state.update(complete=True, files={p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                      for p in sorted(output.iterdir()) if p.suffix in (".npy", ".npz")})
    atomic_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    prepare_features(args.data, args.output, args.model, args.split, args.batch_size)
