"""Reuse frozen chunk features after a data-boundary correction, with exact input checks.

The cache encoder is a pure function of observed token IDs and its pinned
projection/backbone. Article names and QA labels never affect the feature.
Unmatched chunks are encoded afresh; statistics are refitted on the corrected
training split. The old cache remains intact for the preprocessing audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .corpus_data import Corpus, atomic_json, sha256
from .corpus_features import SupportEncoder


def exact_chunk_map(previous, current):
    table = {}
    for i in range(len(previous.chunks)):
        table.setdefault(hashlib.sha256(previous.chunk(i).tobytes()).digest(), i)
    mapping = np.full(len(current.chunks), -1, dtype=np.int64)
    for i in range(len(current.chunks)):
        tokens = current.chunk(i)
        old = table.get(hashlib.sha256(tokens.tobytes()).digest())
        if old is not None:
            if not np.array_equal(tokens, previous.chunk(old)):
                raise RuntimeError("Hash match did not preserve exact input tokens")
            mapping[i] = old
    return mapping


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    previous, current = Corpus(args.old_data, "train"), Corpus(args.data, "train")
    old_root, output = Path(args.old_features), Path(args.output)
    old_manifest = json.loads((old_root / "progress.json").read_text())
    if not old_manifest["complete"]:
        raise ValueError("The source cache has not finished")
    if old_manifest["provenance"]["data_manifest_sha256"] != sha256(Path(args.old_data) / "manifest.json"):
        raise ValueError("Source cache/input identity mismatch")
    if old_manifest["provenance"]["model_config_sha256"] != sha256(Path(args.model) / "config.json"):
        raise ValueError("Backbone config changed")
    if old_manifest["provenance"]["dimension"] != 512 or old_manifest["provenance"]["projection_seed"] != 20261013:
        raise ValueError("Source cache feature coordinates differ")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    if progress_path.exists() and json.loads(progress_path.read_text()).get("complete"):
        return
    t0 = time.monotonic()
    mapping = exact_chunk_map(previous, current)
    np.save(output / "reuse_mapping.npy", mapping)
    print(json.dumps({"chunks": len(mapping), "exact_reused": int((mapping >= 0).sum()),
                      "newly_encoded": int((mapping < 0).sum())}), flush=True)
    sources, arrays = {}, {}
    for name in ("keys", "values", "masks", "labels"):
        sources[name] = np.load(old_root / f"{name}.npy", mmap_mode="r")
        arrays[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=sources[name].dtype,
                                                shape=(len(mapping), *sources[name].shape[1:]))
    provenance = {"schema": "vera-corpus-features-v1", "split": "train", "dimension": 512,
                  "layer": 20, "projection_seed": 20261013, "chunks": len(mapping),
                  "data_manifest_sha256": sha256(Path(args.data) / "manifest.json"),
                  "model_config_sha256": old_manifest["provenance"]["model_config_sha256"],
                  "repair_source_sha256": sha256(__file__),
                  "parent_feature_manifest_sha256": sha256(old_root / "progress.json"),
                  "exact_reused_chunks": int((mapping >= 0).sum()), "newly_encoded_chunks": int((mapping < 0).sum()),
                  "verification": "SHA256 token lookup followed by exact token-array equality for every reused row"}
    for start in range(0, len(mapping), 2048):
        stop = min(start + 2048, len(mapping))
        positions = np.arange(start, stop)[mapping[start:stop] >= 0]
        for name, target in arrays.items():
            target[positions] = sources[name][mapping[positions]]
    encoder = SupportEncoder(args.model)
    missing = np.flatnonzero(mapping < 0)
    for start in range(0, len(missing), 96):
        positions = missing[start:start + 96]
        result = encoder.encode([current.chunk(i) for i in positions],
                                 current.manifest["boundary_token_id"], current.manifest["pad_token_id"])
        for name, value in zip(("keys", "values", "masks", "labels"), result):
            arrays[name][positions] = value
        if start % (96 * 40) == 0:
            record = {"provenance": provenance, "complete": False,
                      "new_chunks_done": min(start + 96, len(missing)), "seconds": time.monotonic() - t0}
            atomic_json(progress_path, record)
            print(json.dumps(record), flush=True)
    for target in arrays.values():
        target.flush()
    means = {}
    for name in ("keys", "values"):
        total = np.zeros(512, dtype=np.float64)
        slots = 0
        for start in range(0, len(mapping), 2048):
            valid = arrays["masks"][start:start + 2048]
            total += arrays[name][start:start + 2048][valid].astype(np.float64).sum(0)
            slots += int(valid.sum())
        means["key_mean" if name == "keys" else "value_mean"] = (total / slots).astype(np.float32)
    np.savez(output / "statistics.npz", **means, fitted_slots=slots)
    atomic_json(progress_path, {"provenance": provenance, "complete": True, "chunks_done": len(mapping),
                               "seconds": time.monotonic() - t0, "files": {
                                   p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                   for p in sorted(output.iterdir()) if p.suffix in (".npy", ".npz")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old-data", "data", "old-features", "output", "model"):
        parser.add_argument("--" + key, required=True)
    main(parser.parse_args())
