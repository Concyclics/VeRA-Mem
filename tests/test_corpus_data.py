"""Leakage, complete exposure, and causal writer tests for the corpus track."""
import numpy as np
import torch

from vera_mem.corpus_data import articles, windows
from vera_mem.corpus_features import pool_features


def test_subheadings_and_shard_boundary_do_not_split_articles():
    rows = ["", " = A = \n", "", "text\n", " = = Sub = = \n", "more\n", "", " = B = \n", "", "last\n"]
    result = list(articles(iter(rows)))
    assert len(result) == 2
    assert result[0][0] == "A"
    assert "Sub" in result[0][1] and "more" in result[0][1]


def test_sports_legend_equal_signs_are_not_article_boundaries():
    rows = ["", " = Team season = \n", "", " Note : Pos \n", " = Position ; GP = \n",
            " Games played in ; G \n", " = Goals ; A = \n", " Assists ; Pts \n", ""]
    result = list(articles(rows))
    assert len(result) == 1 and "Goals" in result[0][1]


def test_an_equals_sign_inside_an_album_title_is_preserved():
    result = list(articles(["", " = E = MC2 ( album ) = \n", "", "An album.\n"]))
    assert result[0][0] == "E = MC2 ( album )"


def test_every_document_target_scored_exactly_once_with_bounded_context():
    for length in (2, 17, 513, 1025, 1026, 2051, 5000):
        for stride in (512, 1024):
            counts = np.zeros(length, dtype=int)
            for start, end, first in windows(length, size=1024, stride=stride):
                assert 0 <= start < end <= length and end - start <= 1025
                counts[start + first:end] += 1
            assert counts[0] == 0 and np.all(counts[1:] == 1)


def test_writer_keys_precede_labels_and_values_contain_only_observed_spans():
    features = torch.arange(2 * 129 * 3, dtype=torch.float).reshape(2, 129, 3)
    tokens = torch.arange(258).reshape(2, 129)
    keys, values, mask, labels = pool_features(features, [128, 19], tokens)
    assert torch.equal(keys[0, 0], features[0, 15])
    assert torch.equal(values[0, 0], features[0, 1:17].mean(0))
    assert labels[0, 0] == tokens[0, 16]
    assert torch.equal(keys[1, 1], features[1, 18])
    assert torch.equal(values[1, 1], features[1, 17:20].mean(0))
    assert labels[1, 1] == tokens[1, 19]
    assert mask[1].tolist() == [True, True, False, False, False, False, False, False]
    assert values[1, 2:].count_nonzero() == keys[1, 2:].count_nonzero() == 0


def test_unobserved_padding_cannot_change_a_written_value():
    features = torch.randn(1, 129, 8)
    tokens = torch.arange(129)[None]
    before = pool_features(features, [19], tokens)
    features[:, 20:] = float("nan")
    after = pool_features(features, [19], tokens)
    for old, new in zip(before, after):
        assert torch.equal(old, new)
