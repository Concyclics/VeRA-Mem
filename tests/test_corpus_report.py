"""Check clustered estimands and strict pairing, including unequal group sizes."""
import numpy as np
import pytest

from vera_mem.corpus_report import cluster_interval, joint_success, paired, ppl_ratio_interval


def test_bootstrap_keeps_unequal_articles_and_paraphrases_together():
    result = cluster_interval([1., 1., 0., 0., 0., 0.], [1, 1, 2, 2, 2, 2])
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["ci95"] == [0., 1.]
    assert result["articles"] == 2
    assert result["queries"] == 6


def test_pairing_rejects_missing_duplicates_or_changed_articles():
    rows = [{"id": "a", "style": 0, "article": 1, "em": 1.},
            {"id": "a", "style": 1, "article": 1, "em": 0.}]
    difference, articles = paired(rows, rows[::-1])
    assert np.array_equal(difference, [0., 0.])
    assert np.array_equal(articles, [1, 1])
    for invalid in (rows[:1], rows + rows[:1], [dict(rows[0], article=2), rows[1]]):
        with pytest.raises(ValueError):
            paired(rows, invalid)
    changed = [dict(row, em=1.) for row in rows]
    assert joint_success(rows, changed, changed)["mean"] == .5


def test_ppl_ratio_uses_token_counts_and_excludes_contamination():
    left = [[10., 10.], [200., 100.], [900., 10.]]
    right = [[0., 10.], [100., 100.], [0., 10.]]
    result = ppl_ratio_interval(left, right, excluded=[2])
    assert result["ratio"] == pytest.approx(np.e)
    assert result["ci95"] == pytest.approx([np.e, np.e])
    assert result["target_tokens"] == 110
    with pytest.raises(ValueError):
        ppl_ratio_interval(left, [[0., 11.], [100., 100.], [0., 10.]])
