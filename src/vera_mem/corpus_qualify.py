"""Engineering qualification on training-only activations before formal exposure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .corpus_backend import CorpusBackbone
from .corpus_data import Corpus, atomic_json
from .corpus_index import CandidateIndex, normalize
from .corpus_reader import CorpusReader


def main(args):
    import faiss
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(args.data, "train")
    backbone = CorpusBackbone(args.model)
    qualification = backbone.qualify_prefix()
    reader = CorpusReader().to("cuda").eval().requires_grad_(False)
    stats = np.load(Path(args.features) / "statistics.npz")
    reader.key_mean.copy_(torch.tensor(stats["key_mean"], device="cuda"))
    table = corpus.window_table()
    selected = np.random.default_rng(20261013).choice(len(table), 8, replace=False)
    queries, article_ids = [], []
    with torch.no_grad():
        for record in table[selected]:
            ids, labels, mask = [torch.from_numpy(t).to("cuda") for t in corpus.batch([record], corpus.manifest["pad_token_id"])]
            prefix = backbone.prefix(ids, mask)
            # Actual query activations, not self-queries sampled from index vectors.
            x = prefix["x"][0]
            positions = torch.linspace(0, len(x) - 1, 32, device="cuda").long()
            q = reader.route_queries(reader.query_features(x[positions]))
            queries.append(q.cpu().numpy())
            article_ids.extend([int(corpus.article_groups[record[0]])] * len(positions))
    queries = normalize(np.concatenate(queries))
    article_ids = np.asarray(article_ids)
    index = CandidateIndex.load(args.index, threads=8)
    exact = faiss.IndexFlatIP(index.vectors.shape[1])
    exact.add(index.vectors)
    started = time.monotonic()
    _, found = exact.search(queries, 512)
    exact_seconds = time.monotonic() - started
    valid = index.articles[found] != article_ids[:, None]
    truth = [set(row[keep][:32].tolist()) for row, keep in zip(found, valid)]
    if any(len(row) != 32 for row in truth):
        raise RuntimeError("Exact qualification search did not provide 32 eligible neighbors")
    probes = []
    chosen = None
    budgets = (64, 128, 256, 512) if index.backend == "hnsw" else (8, 16, 32, 64, 128, 256, 512, 1024)
    budget_name = "ef_search" if index.backend == "hnsw" else "nprobe"
    for ef in budgets:
        index.set_search_budget(ef)
        latencies = []
        for repeat in range(3):
            started = time.monotonic()
            actual = index.search(queries, 32, article_ids)
            latencies.append(time.monotonic() - started)
        recall = float(np.mean([len(set(row.tolist()) & target) / 32 for row, target in zip(actual, truth)]))
        probes.append({budget_name: ef, "recall_at_32": recall, "batch_seconds": latencies,
                       "queries_per_second": len(queries) / np.median(latencies)})
        if chosen is None and recall >= .95:
            chosen = ef
    # All settings are engineering diagnostics, not test-quality tuning.
    chosen = chosen or budgets[-1]
    index.set_search_budget(chosen)
    qualified_index = output / "index"
    if not qualified_index.exists():
        index.save(qualified_index)
    result = {"prefix": qualification, "queries": len(queries), "training_articles": len(set(article_ids)),
              "blocks": index.index.ntotal, "exact_batch_seconds": exact_seconds, "probes": probes,
              "chosen_search_budget": chosen, "budget_name": budget_name, "backend": index.backend,
              "selection_rule": "Smallest tested search budget with >=95% same-article-excluded recall@32; otherwise maximum and report shortfall",
              "index_coordinate_contract": "Fixed projected training features; fine K/V recomputed during training",
              "parameters": sum(p.numel() for p in reader.parameters()),
              "query_source": "Actual layer-20 down-projection inputs from eight training windows"}
    atomic_json(output / "qualification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "data", "features", "index", "output"):
        parser.add_argument("--" + key, required=True)
    main(parser.parse_args())
