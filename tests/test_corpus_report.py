"""Check clustered estimands and strict pairing, including unequal group sizes."""
import numpy as np
import pytest
import json
import math

from vera_mem.corpus_report import analyze_evaluation, cluster_interval, joint_success, paired, ppl_ratio_interval


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


def test_completed_evaluation_with_positive_oracle_gain_serializes(tmp_path):
    """Exercise the failed export path using actual condition-file joins."""
    (tmp_path / "complete.json").write_text("{}")
    (tmp_path / "evaluation.json").write_text(json.dumps({"development": True, "split": "validation"}))
    folder = tmp_path / "continual_validation"
    folder.mkdir()
    (folder / "complete.json").write_text(json.dumps({
        "parameter_digests_before": {"reader": "fixed"},
        "parameter_digests_after": {"reader": "fixed"}, "groups": {}}))

    def condition(name, phase, score):
        rows = [{"id": str(i), "article": i, "style": style, "em": score, "f1": score}
                for i in range(2) for style in range(2)]
        (folder / f"{name}_{phase}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    condition("base", "empty", 0.)
    condition("lora", "prewrite", 0.)
    modes = ("diagonal", "factor_mean_outer", "block_outer")
    for name in modes:
        for phase in ("write_a_0000", "final_a", "oracle_a", "write_b_0000", "restore_a_0000"):
            condition(name, phase, 1.)
        for phase in ("prewrite", "shuffled_a", "stale_b_0000"):
            condition(name, phase, 0.)
    methods = {name: {"ppl": math.e, "article_nll_tokens": [[10., 10.], [10., 10.]]}
               for name in ("base", "lora", *modes)}
    (tmp_path / "static_validation.json").write_text(json.dumps({
        "excluded_primary_articles": [], "methods": methods}))
    (tmp_path / "shuffled_validation.json").write_text(json.dumps({"methods": methods}))
    result = json.loads(json.dumps(analyze_evaluation(tmp_path)))
    assert result["methods"]["block_outer"]["diagnostic_gates"]["at_least_half_positive_oracle_gain"] is True
    assert result["methods"]["block_outer"]["fraction_of_positive_oracle_gain_recovered"] == 1.
