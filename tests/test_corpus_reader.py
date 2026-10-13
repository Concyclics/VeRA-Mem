"""Independent operator, gradient, indexing, and sparse-target checks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from vera_mem.corpus_index import CandidateIndex, normalize
from vera_mem.corpus_reader import CorpusReader, FrozenBank, sparse_cross_entropy


def make_bank(blocks=9):
    return SimpleNamespace(keys=torch.randn(blocks, 8, 512), values=torch.randn(blocks, 8, 512),
                           masks=torch.ones(blocks, 8, dtype=torch.bool), labels=torch.randint(0, 31, (blocks, 8)),
                           size=blocks)


def test_block_operator_and_rank_one_match_their_explicit_matrices():
    for mode in ("diagonal", "factor_mean_outer", "block_outer"):
        reader = CorpusReader(in_features=13, out_features=7, rank=5, mode=mode)
        x, values = torch.randn(2, 13), torch.randn(2, 8, 5)
        weights = torch.rand(2, 8).softmax(-1)
        z = F.linear(x, reader.A)
        matrix = torch.diag_embed((values * weights[..., None]).sum(1))
        if mode != "diagonal":
            u, w = F.normalize(reader.u(values), dim=-1), reader.w(values)
            if mode == "factor_mean_outer":
                matrix += torch.einsum("bi,bj->bij", (w * weights[..., None]).sum(1), (u * weights[..., None]).sum(1))
            else:
                matrix += torch.einsum("bs,bsi,bsj->bij", weights, w, u)
        expected = torch.einsum("bij,bj->bi", matrix, z)
        assert torch.allclose(reader.latent(x, values, weights), expected, atol=1e-6)
        permutation = torch.randperm(8)
        assert torch.allclose(reader.latent(x, values[:, permutation], weights[:, permutation]), expected, atol=1e-6)


def test_matched_outer_arms_share_parameters_but_keep_distinct_covariance():
    mean = CorpusReader(in_features=13, out_features=7, rank=5, mode="factor_mean_outer")
    block = CorpusReader(in_features=13, out_features=7, rank=5, mode="block_outer")
    for a, b in zip(mean.parameters(), block.parameters()):
        assert torch.equal(a, b)
    x, v, w = torch.randn(3, 13), torch.randn(3, 8, 5), torch.full((3, 8), 1 / 8)
    assert not torch.allclose(mean.latent(x, v, w), block.latent(x, v, w))


def test_selected_qkv_receive_gradients_and_inference_never_reads_labels():
    reader, bank = CorpusReader(in_features=13, out_features=7, rank=5), make_bank()
    reader.gain.data.fill_(.1)
    x = torch.randn(4, 13)
    f = reader.query_features(x)
    candidates = torch.arange(8)[None].expand(4, -1)
    before, info = reader.read(x, f, candidates, bank)
    before.square().sum().backward()
    for component in (reader.Q, reader.K, reader.V):
        assert sum(float(p.grad.abs().sum()) for p in component.parameters()) > 0
    bank.labels.fill_(9999999)
    after, _ = reader.read(x, f, candidates, bank)
    assert torch.equal(before, after)


def test_missing_candidates_cannot_inject_values():
    reader, bank = CorpusReader(in_features=13, out_features=7, rank=5), make_bank()
    reader.gain.data.fill_(1)
    x = torch.randn(2, 13)
    delta, _ = reader.read(x, reader.query_features(x), torch.full((2, 4), -1), bank)
    assert delta.count_nonzero() == 0


def test_sparse_teacher_matches_dense_ce_and_gradient_with_duplicate_ids():
    # Verify algebra in float64: two equivalent float32 reduction orders can
    # differ by one ulp near zero, exceeding allclose's default absolute bound.
    logits = torch.randn(2, 11, dtype=torch.float64, requires_grad=True,
                         generator=torch.Generator().manual_seed(17))
    ids = torch.tensor([[3, 3, 7], [2, 8, 2]])
    weights = torch.tensor([[.2, .3, .5], [.1, .8, .1]], dtype=torch.float64)
    dense = torch.zeros_like(logits).scatter_add_(1, ids, weights)
    expected = -(dense * F.log_softmax(logits, -1)).sum(-1)
    actual = sparse_cross_entropy(logits, ids, weights)
    assert torch.allclose(actual, expected)
    assert torch.allclose(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                          torch.autograd.grad(expected.sum(), logits)[0])


def test_ann_excludes_same_article_stale_versions_and_restores_snapshot(tmp_path):
    pytest.importorskip("faiss")
    vectors = normalize(np.random.default_rng(17).normal(size=(40, 16)))
    index = CandidateIndex(vectors, np.repeat(np.arange(10), 4), threads=1)
    result = index.search(vectors[:4], count=8, exclude_articles=np.zeros(4, dtype=int))
    assert np.all(result >= 4)
    replacement = index.upsert("0", vectors[0], article=10)
    result = index.search(vectors[:1], count=8)
    assert 0 not in result and replacement in result
    second = index.upsert("0", vectors[1], article=10)
    assert replacement not in index.search(vectors[:2], count=16)
    path = tmp_path / "snapshot"
    index.index.hnsw.efSearch = 256
    index.save(path)
    restored = CandidateIndex.load(path, threads=1)
    assert restored.logical["0"] == second
    assert restored.index.hnsw.efSearch == 256
    assert np.array_equal(index.search(vectors, 8), restored.search(vectors, 8))
    with (path / "logical.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        CandidateIndex.load(path)


def test_frozen_bank_keeps_only_low_rank_rows_and_never_changes_writer_weights():
    reader, features = CorpusReader(in_features=13, out_features=7, rank=5), make_bank()
    reader.gain.data.fill_(.1)
    reader.requires_grad_(False)
    features.keys = features.keys.to(torch.float16).to(torch.bfloat16)
    features.values = features.values.to(torch.float16).to(torch.bfloat16)
    before = {name: value.clone() for name, value in reader.state_dict().items()}
    bank = FrozenBank(reader, features, reserve=2)
    assert bank.keys.shape[-1] == bank.values.shape[-1] == 5
    assert not hasattr(bank, "labels")
    positions = bank.append(features.keys[:1], features.values[:1], features.masks[:1])
    assert positions.tolist() == [features.size]
    assert all(torch.equal(value, reader.state_dict()[name]) for name, value in before.items())
    with pytest.raises(TypeError):
        bank.append(features.keys[:1], features.values[:1], features.masks[:1], labels=[1])
    x = torch.randn(3, 13)
    candidates = torch.arange(8)[None].expand(3, -1)
    original, _ = reader.read(x, reader.query_features(x), candidates, features)
    frozen, _ = reader.read(x, reader.query_features(x), candidates, bank)
    assert torch.allclose(original, frozen, atol=2e-3, rtol=.03)
