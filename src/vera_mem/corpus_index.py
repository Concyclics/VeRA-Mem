"""Fixed-coordinate ANN candidates with trainable, current-weight reranking.

Coarse coordinates do not depend on trainable K/V weights. Updating those
weights cannot stale the index: candidate K/V are recomputed on every read.
Physical versions are append-only; logical upserts immediately mask old rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .corpus_data import Corpus, atomic_json, sha256


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def coarse_keys(features, masks, mean, dimension=128):
    centered = np.asarray(features[..., :dimension], dtype=np.float32) - mean[:dimension]
    return normalize((centered * masks[..., None]).sum(1) / masks.sum(1)[:, None].clip(1))


class CandidateIndex:
    def __init__(self, vectors, articles, *, index=None, neighbors=32, ef_search=128, threads=8,
                 backend="hnsw", nlist=4096):
        import faiss
        faiss.omp_set_num_threads(threads)
        self.vectors = normalize(vectors)
        self.articles = np.asarray(articles, dtype=np.int64)
        if len(self.vectors) != len(self.articles):
            raise ValueError("Each physical block needs an article ID")
        self.active = np.ones(len(vectors), dtype=bool)
        self.logical = {str(i): i for i in range(len(vectors))}
        self.backend = backend
        self.resources = None
        if backend == "hnsw":
            self.index = index or faiss.IndexHNSWFlat(self.vectors.shape[1], neighbors, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 128
            self.index.hnsw.efSearch = ef_search
            if self.index.ntotal == 0 and len(self.vectors):
                self.index.add(self.vectors)
            self.search_index = self.index
        elif backend == "ivf_gpu":
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("GPU IVF requires a GPU-enabled Faiss installation; no silent CPU fallback")
            self.resources = faiss.StandardGpuResources()
            self.resources.setTempMemory(256 << 20)
            self.resources.setDefaultNullStreamAllDevices()
            if index is None:
                if len(self.vectors) < nlist * 39:
                    raise ValueError("Too few training blocks for the declared IVF centroids")
                sample = np.random.default_rng(20261013).choice(len(self.vectors), min(len(self.vectors), nlist * 64), replace=False)
                kmeans = faiss.Kmeans(self.vectors.shape[1], nlist, niter=25, nredo=1,
                                      spherical=True, seed=20261013, gpu=True, verbose=False)
                kmeans.train(self.vectors[sample])
                quantizer = faiss.IndexFlatIP(self.vectors.shape[1])
                quantizer.add(kmeans.centroids)
                index = faiss.IndexIVFFlat(quantizer, self.vectors.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
                index.is_trained = True
                index.nprobe = 128
            options = faiss.GpuClonerOptions()
            options.useFloat16 = False
            self.search_index = faiss.index_cpu_to_gpu(self.resources, 0, index, options)
            if index.ntotal == 0 and len(self.vectors):
                self.search_index.add(self.vectors)
                index = faiss.index_gpu_to_cpu(self.search_index)
            self.index = index
        else:
            raise ValueError(backend)
        if self.index.ntotal != len(vectors):
            raise ValueError("Index and payload counts differ")

    def search(self, queries, count=32, exclude_articles=None, *, forced=None):
        """No exact-scan fallback. Missing candidates have ID -1.

        A forced block is an explicitly marked training/oracle intervention;
        ordinary evaluation calls do not pass it. Same-article exclusion is
        applied after ANN, with bounded oversampling.
        """
        queries = normalize(queries)
        if queries.ndim != 2:
            raise ValueError("Require [queries, coarse dimension]")
        result = np.full((len(queries), count), -1, dtype=np.int64)
        if not self.index.ntotal:
            return result
        if exclude_articles is not None:
            exclude_articles = np.broadcast_to(np.asarray(exclude_articles), (len(queries),))
        unresolved = np.arange(len(queries))
        for budget in (max(count * 2, 64), max(count * 4, 128), max(count * 16, 512)):
            if not len(unresolved):
                break
            _, found = self.search_index.search(queries[unresolved], min(budget, self.index.ntotal))
            # Filtering is vectorized, preserving score order from ANN.
            safe = np.maximum(found, 0)
            valid = (found >= 0) & self.active[safe]
            if exclude_articles is not None:
                valid &= self.articles[safe] != exclude_articles[unresolved, None]
            positions = np.cumsum(valid, axis=1) - 1
            rr, cc = np.where(valid & (positions < count))
            result[unresolved[rr], positions[rr, cc]] = found[rr, cc]
            unresolved = unresolved[(result[unresolved] < 0).any(1)]
        if forced is not None:
            forced = np.broadcast_to(np.asarray(forced, dtype=np.int64), (len(queries),))
            if np.any((forced < 0) | (forced >= len(self.active))) or not self.active[forced].all():
                raise ValueError("Forced blocks must be active")
            missing = ~(result == forced[:, None]).any(1)
            result[missing, -1] = forced[missing]
        return result

    def upsert(self, logical_id, vector, article=-1):
        """The caller atomically stages the matching support-only payload first."""
        return int(self.upsert_many([logical_id], np.asarray(vector)[None], [article])[0])

    def upsert_many(self, logical_ids, vectors, articles):
        """Commit an ordered observation minibatch without copying the index per record."""
        vectors = normalize(vectors)
        if vectors.shape[1] != self.vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("Invalid new index vector")
        if len(logical_ids) != len(vectors) or len(articles) != len(vectors):
            raise ValueError("Index upsert batch lengths differ")
        if len(set(map(str, logical_ids))) != len(logical_ids):
            raise ValueError("Each upsert batch must have unique logical IDs")
        position = self.index.ntotal
        self.index.add(vectors)
        if self.search_index is not self.index:
            self.search_index.add(vectors)
        self.vectors = np.concatenate((self.vectors, vectors))
        self.articles = np.concatenate((self.articles, np.asarray(articles, dtype=np.int64)))
        self.active = np.concatenate((self.active, np.ones(len(vectors), dtype=bool)))
        for offset, logical_id in enumerate(logical_ids):
            old = self.logical.get(str(logical_id))
            if old is not None:
                self.active[old] = False
            self.logical[str(logical_id)] = position + offset
        return np.arange(position, position + len(vectors), dtype=np.int64)

    def set_search_budget(self, value):
        if self.backend == "hnsw":
            self.index.hnsw.efSearch = int(value)
        else:
            self.index.nprobe = int(value)
            self.search_index.nprobe = int(value)

    def clone(self):
        import faiss
        instance = CandidateIndex(self.vectors.copy(), self.articles.copy(), index=faiss.clone_index(self.index),
                                  backend=self.backend,
                                  ef_search=int(self.index.hnsw.efSearch) if self.backend == "hnsw" else 128)
        instance.active = self.active.copy()
        instance.logical = dict(self.logical)
        return instance

    def save(self, directory):
        import faiss
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        faiss.write_index(self.index, str(directory / "ann.faiss"))
        np.savez(directory / "metadata.npz", vectors=self.vectors, articles=self.articles, active=self.active)
        atomic_json(directory / "logical.json", self.logical)
        atomic_json(directory / "settings.json", {"backend": self.backend, "faiss_version": faiss.__version__})
        atomic_json(directory / "complete.json", {p.name: sha256(p) for p in directory.iterdir()})

    @classmethod
    def load(cls, directory, threads=8):
        import faiss
        directory = Path(directory)
        manifest = json.loads((directory / "complete.json").read_text())
        if any(sha256(directory / name) != expected for name, expected in manifest.items()):
            raise ValueError("Index snapshot checksum mismatch")
        metadata = np.load(directory / "metadata.npz")
        stored_path = directory / "ann.faiss"
        if not stored_path.exists():
            stored_path = directory / "hnsw.faiss"
        stored = faiss.read_index(str(stored_path))
        backend = (json.loads((directory / "settings.json").read_text())["backend"]
                   if (directory / "settings.json").exists() else "hnsw")
        instance = cls(metadata["vectors"], metadata["articles"],
                       index=stored, ef_search=int(stored.hnsw.efSearch) if backend == "hnsw" else 128,
                       threads=threads, backend=backend)
        instance.active = metadata["active"].copy()
        instance.logical = json.loads((directory / "logical.json").read_text())
        return instance


def build(features, data, output, backend="ivf_gpu"):
    features, output = Path(features), Path(output)
    progress = json.loads((features / "progress.json").read_text())
    if not progress["complete"] or progress["provenance"]["split"] != "train":
        raise ValueError("Only a completed training feature cache can initialize this index")
    keys = np.load(features / "keys.npy", mmap_mode="r")
    masks = np.load(features / "masks.npy", mmap_mode="r")
    mean = np.load(features / "statistics.npz")["key_mean"]
    coarse = np.empty((len(keys), 128), dtype=np.float32)
    for start in range(0, len(keys), 4096):
        coarse[start:start + 4096] = coarse_keys(keys[start:start + 4096], masks[start:start + 4096], mean)
    began = time.monotonic()
    corpus = Corpus(data, "train")
    index = CandidateIndex(coarse, corpus.article_groups[corpus.chunks[:, 0]], backend=backend)
    index.save(output)
    atomic_json(output / "build.json", {"seconds": time.monotonic() - began, "blocks": len(keys),
                                       "feature_manifest_sha256": sha256(features / "progress.json"),
                                       "coarse_dimension": 128, "backend": backend,
                                       "ivf_centroids": 4096 if backend == "ivf_gpu" else None,
                                       "coordinate_rule": "Frozen projected features, train-only mean",
                                       "trainable_key_index_staleness": "Not applicable: fine keys recomputed per read"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("ivf_gpu", "hnsw"), default="ivf_gpu")
    args = parser.parse_args()
    build(args.features, args.data, args.output, args.backend)
