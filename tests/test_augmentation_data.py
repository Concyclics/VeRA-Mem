"""Protocol checks for isolated template and entity generalization."""

from collections import Counter
from dataclasses import replace
import random
import re

import pytest

from vera_mem.augmentation_data import (
    datasets, get_templates, protocol_fingerprint, protocol_manifest,
    render_question, render_support,
)
from vera_mem.data import MEMORY_WORDS, synthetic_dataset


def test_training_facts_and_canonical_rendering_match_scaling_study():
    data = datasets()
    old = synthetic_dataset(1042, 4096, 0)["stream"]
    groups = {word: [row for row in old if row.answer == word] for word in MEMORY_WORDS}
    expected = [groups[word][i] for i in range(256) for word in MEMORY_WORDS]
    assert [row.id for row in data["train"]] == [row.id for row in expected]
    for row, original in zip(data["train"], expected):
        assert (row.answer, row.support, row.question) == (
            original.answer, original.support, original.question)
        assert render_question(row, "train_query_00") == row.question
        assert render_question(row, "train_query_01") == row.paraphrase
        assert render_support(row, "train_support_00") == row.support


def test_fact_splits_are_disjoint_fresh_deterministic_and_balanced():
    random.seed(901)
    before = random.getstate()
    data = datasets()
    assert random.getstate() == before
    assert data == datasets()
    ids = {name: {row.id for row in rows} for name, rows in data.items()}
    assert {name: len(rows) for name, rows in data.items()} == {
        "train": 4096, "dev": 64, "test": 128, "control": 32}
    old_online = synthetic_dataset(7042, 128, 64)
    old_ids = {row.id for rows in old_online.values() for row in rows}
    for name, rows in data.items():
        assert len(ids[name]) == len(rows)
        for other in data:
            if other != name:
                assert ids[name].isdisjoint(ids[other])
        assert Counter(row.answer for row in rows) == {
            word: len(rows) // len(MEMORY_WORDS) for word in MEMORY_WORDS}
        if name != "train":
            assert ids[name].isdisjoint(old_ids)
        assert all(row.metadata["split"] == name for row in rows)
        assert all(row.metadata["never_written"] == (name == "control") for row in rows)
    assert ids["dev"].isdisjoint(
        row.id for row in synthetic_dataset(2042, 64, 0)["stream"])
    for train_size, test_size in ((16, 16), (33, 17), (128, 64)):
        smaller = datasets(train_size, test_size)
        assert smaller["train"] == data["train"][:train_size]
        assert smaller["test"] == data["test"][:test_size]
        assert smaller["control"] == data["control"]
        assert smaller["dev"] == data["dev"]
        for split in ("train", "test"):
            counts = [Counter(row.answer for row in smaller[split])[word]
                      for word in MEMORY_WORDS]
            assert max(counts) - min(counts) <= 1


def test_templates_have_separate_expression_families_and_no_hidden_query_labels():
    expected_counts = {"train": (8, 4), "dev": (2, 2), "test": (3, 3)}
    all_templates = []
    rows = datasets(16, 16)["train"]
    for split, counts in expected_counts.items():
        for kind, count in zip(("query", "support"), counts):
            templates = get_templates(split, kind)
            assert isinstance(templates, tuple) and len(templates) == count
            all_templates.extend(templates)
            for template in templates:
                for row in rows:
                    renderer = render_question if kind == "query" else render_support
                    text = renderer(row, template.id)
                    assert renderer(row, template) == text
                    assert text.count(row.metadata["entity"]) == 1
                    vocabulary_in_text = set(re.findall(r"\b[a-z]+\b", text.lower())) & set(MEMORY_WORDS)
                    if kind == "query":
                        assert not vocabulary_in_text
                        poisoned = replace(row, answer="DO_NOT_RENDER", support="DO_NOT_RENDER", question="DO_NOT_RENDER")
                        assert render_question(poisoned, template) == text
                    else:
                        assert vocabulary_in_text == {row.answer}
                        assert len(re.findall(rf"\b{row.answer}\b", text)) == 1
    assert len({template.id for template in all_templates}) == len(all_templates)
    assert len({template.family for template in all_templates}) == len(all_templates)
    # Confirm actual representation families, rather than only trusting ID names.
    exposed_text = "\n".join(template.text for template in all_templates if template.split != "test")
    for held_out_marker in ("<request>", "<memory>", "entity,memory_word", "User:", "Assistant:"):
        assert held_out_marker not in exposed_text


def test_manifest_identifies_old_paraphrase_as_seen_and_hashes_fact_sizes():
    manifest = protocol_manifest()
    assert manifest["historical_paraphrase_is_training_template"] == "train_query_01"
    assert len(protocol_fingerprint()) == 64
    assert protocol_fingerprint() == protocol_fingerprint()
    assert protocol_fingerprint(32, 16) != protocol_fingerprint()
    smaller = protocol_manifest(32, 16)
    assert smaller["fact_sha256"]["train"] != manifest["fact_sha256"]["train"]
    assert smaller["fact_sha256"]["test"] != manifest["fact_sha256"]["test"]
    assert smaller["fact_sha256"]["dev"] == manifest["fact_sha256"]["dev"]
    assert smaller["fact_sha256"]["control"] == manifest["fact_sha256"]["control"]


@pytest.mark.parametrize("train_size,eval_size", [(0, 16), (4097, 16), (32, 0), (32, 129), (True, 16), (32, 1.5)])
def test_invalid_counts_rejected(train_size, eval_size):
    with pytest.raises(ValueError):
        datasets(train_size, eval_size)


def test_unknown_or_mutated_templates_and_missing_entity_are_rejected():
    row = datasets(16, 16)["test"][0]
    for split, kind in (("bogus", "query"), ("test", "question")):
        with pytest.raises(ValueError):
            get_templates(split, kind)
    for template_id in ("absent", "train_support_00"):
        with pytest.raises(ValueError):
            render_question(row, template_id)
    with pytest.raises(ValueError, match="does not match"):
        render_question(row, replace(get_templates("train", "query")[0], text="leaked"))
    with pytest.raises(ValueError, match="no entity"):
        render_support(replace(row, metadata={}), "train_support_00")
