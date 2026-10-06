"""Equal-length counterfactuals, single-row interventions and fresh style splits."""
from collections import Counter
from dataclasses import asdict, replace
import random
import re

import pytest

from vera_mem import augmentation_data as historical
from vera_mem import counterfactual_data as data
from vera_mem.data import MEMORY_WORDS, synthetic_dataset


@pytest.fixture(scope="module")
def packet():
    return data.datasets()


def test_original_training_development_facts_and_cache_order_are_exactly_reused(packet):
    old = historical.datasets()
    for split in ("train", "dev"):
        assert [pair.original for pair in packet[split]] == old[split]
    assert len(packet["train"]) == 4096 and len(packet["dev"]) == 64 and len(packet["confirmation"]) == 64
    assert packet["template_ids"]["train"] == {
        "q": [f"train_query_{index:02d}" for index in range(8)],
        "s": [f"train_support_{index:02d}" for index in range(4)],
    }
    assert packet["template_ids"]["dev"]["q"] == ["train_query_00", "dev_query_00", "dev_query_01"]
    assert packet["template_ids"]["dev"]["s"] == ["train_support_00", "dev_support_00", "dev_support_01"]


def test_A_B_have_equal_token_lengths_balanced_marginals_and_multiple_alternatives(packet):
    for split in ("train", "dev", "confirmation"):
        pairs = packet[split]
        for world in ("a", "b"):
            assert Counter(getattr(pair, world).answer for pair in pairs) == {
                word: len(pairs)//16 for word in MEMORY_WORDS}
        for pair in pairs:
            assert pair.a.answer != pair.b.answer
            assert data.ANSWER_TOKEN_LENGTHS[pair.a.answer] == data.ANSWER_TOKEN_LENGTHS[pair.b.answer]
            assert pair.a.id == pair.b.id == pair.id
            assert pair.a.metadata == pair.b.metadata
            assert pair.a.metadata["entity"] == pair.entity
            assert pair.a.question == pair.b.question
            assert pair.a.paraphrase == pair.b.paraphrase
            assert data.render_support(pair, "train_support_00", world="B") == pair.b.support
        alternatives = {word: {pair.b.answer for pair in pairs if pair.a.answer == word} for word in MEMORY_WORDS}
        # Fixed B per entity does not mean one fixed A-label -> B-label mapping.
        assert all(len(values) > 1 for values in alternatives.values())
    # Within-length alternatives cover every allowed other class in training.
    for word in MEMORY_WORDS:
        expected = set(data.TOKEN_LENGTH_GROUPS[data.ANSWER_TOKEN_LENGTHS[word]])-{word}
        assert {pair.b.answer for pair in packet["train"] if pair.a.answer == word} == expected


def test_fresh_confirmation_is_disjoint_from_all_declared_historical_sets(packet):
    old = historical.datasets()
    forbidden = {row.metadata["entity"] for rows in old.values() for row in rows}
    for seed, count, control in ((2042, 64, 0), (7042, 128, 64), (8042, 32, 16), (27042, 64, 0)):
        extra = synthetic_dataset(seed, count, control)
        forbidden.update(row.metadata["entity"] for rows in extra.values() for row in rows)
    entities = {pair.entity for pair in packet["confirmation"]}
    assert len(entities) == 64 and entities.isdisjoint(forbidden)
    assert all(pair.a.metadata["seed"] == 37042 for pair in packet["confirmation"])
    assert all(pair.a.metadata["reserved_for_confirmation"] is True for pair in packet["confirmation"])
    ids = {split: {pair.id for pair in packet[split]} for split in ("train", "dev", "confirmation")}
    for split in ids:
        for other in ids:
            if split != other:
                assert ids[split].isdisjoint(ids[other])


def test_nested_training_prefixes_are_balanced_and_global_rng_is_untouched(packet):
    random.seed(204)
    before = random.getstate()
    smaller = data.datasets(32)
    assert before == random.getstate()
    assert smaller["train"] == packet["train"][:32]
    assert smaller["dev"] == packet["dev"]
    assert smaller["confirmation"] == packet["confirmation"]
    assert smaller["confirmation_view_assignments"] == packet["confirmation_view_assignments"]
    for world in ("a", "b"):
        assert Counter(getattr(pair, world).answer for pair in smaller["train"]) == {word: 2 for word in MEMORY_WORDS}


def test_four_style_cells_are_balanced_for_every_answer_in_both_worlds(packet):
    views = packet["confirmation_view_assignments"]
    assert len(views) == 64
    expected = Counter({(q, s): 1 for q in range(2) for s in range(2)})
    for world in ("a", "b"):
        for answer in MEMORY_WORDS:
            actual = Counter((view["query_view"], view["support_view"])
                             for pair, view in zip(packet["confirmation"], views) if getattr(pair, world).answer == answer)
            assert actual == expected
    ids = packet["template_ids"]["confirmation"]
    for pair, view in zip(packet["confirmation"], views):
        assert view["id"] == pair.id
        assert view["query_template_id"] == ids["q"][view["query_view"]]
        assert view["support_template_id"] == ids["s"][view["support_view"]]
        assert view["paraphrase_template_id"] == ids["s"][1-view["support_view"]]
    assert data.confirmation_view_assignments(packet["confirmation"]) == views
    changed = data.confirmation_view_assignments(packet["confirmation"], seed=34043)
    assert changed != views
    # Independent randomization is neither alternating rows nor label-modulo assignment.
    assert [view["query_view"] for view in views] != [index % 2 for index in range(64)]


def test_reserved_actual_structure_families_and_query_label_separation(packet):
    train_text = "\n".join(template.text for split in ("train", "dev") for kind in ("query", "support")
                           for template in data.get_templates(split, kind))
    assert '"requested_field"' not in train_text
    assert "| --- | --- |" not in train_text
    fresh_q = data.get_templates("confirmation", "query")
    fresh_s = data.get_templates("confirmation", "support")
    assert len(fresh_q) == len(fresh_s) == 2
    assert "JSON" in fresh_q[0].text and "Markdown" in fresh_q[1].text
    for split in ("train", "dev", "confirmation"):
        pair = packet[split][0]
        for template in data.get_templates(split, "query"):
            assert data.render_question(pair, template, world="A") == data.render_question(pair, template.id, world="B")
            poisoned = replace(pair.a, question="HIDDEN_ANSWER", support="HIDDEN_ANSWER", answer="HIDDEN_ANSWER")
            text = data.render_question(poisoned, template)
            assert "HIDDEN_ANSWER" not in text and text.count(pair.entity) == 1
            assert not (set(re.findall(r"\b[a-z]+\b", text.lower())) & set(MEMORY_WORDS))
        for template in data.get_templates(split, "support"):
            for world in ("A", "B"):
                example = pair.a if world == "A" else pair.b
                text = data.render_support(pair, template, world=world)
                assert text.count(pair.entity) == 1
                assert len(re.findall(rf"\b{example.answer}\b", text)) == 1
    # Rendering remains compatible with old template IDs, but they are never
    # exposed as new reserved confirmation families.
    assert data.render_question(packet["dev"][0], "test_query_00") == historical.render_question(packet["dev"][0].a, "test_query_00")


def test_three_banks_change_only_the_target_and_keep_A_B_questions_identical(packet):
    pairs = packet["confirmation"]
    target, indices = 7, [4, 7, 20, 31, 41]
    view = packet["confirmation_view_assignments"][target]
    result = data.paired_bank(pairs, target, indices, query_template_id=view["query_template_id"],
                             support_template_id=view["support_template_id"])
    assert result["target_position"] == 1 and result["target_id"] == pairs[target].id
    assert result["answers"] == {"A": pairs[target].a.answer, "B": pairs[target].b.answer, "P": pairs[target].a.answer}
    assert result["paraphrase_template_id"] == view["paraphrase_template_id"]
    for index in range(len(indices)):
        a, b, p = [result[world][index] for world in ("A", "B", "P")]
        assert a.id == b.id == p.id
        assert a.question == b.question == p.question
        if index == result["target_position"]:
            assert a.question == result["query"]
            assert a.answer != b.answer and a.answer == p.answer
            changed_B = {name for name, value in asdict(a).items() if value != asdict(b)[name]}
            changed_P = {name for name, value in asdict(a).items() if value != asdict(p)[name]}
            assert changed_B == {"answer", "support"}
            assert changed_P == {"support"}
        else:
            assert a == b == p
        assert a is not b and a.metadata is not b.metadata
    # Copying/intervening on a bank must not mutate its source or another world.
    source_answer = pairs[indices[0]].a.answer
    result["A"][0].answer = "MUTATED"
    result["A"][0].metadata["entity"] = "MUTATED"
    assert result["B"][0].answer == source_answer
    assert pairs[indices[0]].a.answer == source_answer
    assert pairs[indices[0]].entity != "MUTATED"
    canonical = data.paired_bank(packet["train"], 0, [0, 1, 16])
    assert canonical["paraphrase_template_id"] == "train_support_01"


def test_same_answer_wrong_entities_remain_explicit_negative_addresses(packet):
    pairs, target = packet["confirmation"], 0
    for world in ("A", "B", "P"):
        candidates = data.hard_negative_indices(pairs, target, world=world)
        expected_answer = pairs[target].b.answer if world == "B" else pairs[target].a.answer
        assert len(candidates) == (4 if world == "B" else 3)
        assert all(pairs[index].entity != pairs[target].entity for index in candidates)
        assert all(pairs[index].a.answer == expected_answer for index in candidates)
        subset = [target]+candidates[:2]
        assert data.hard_negative_indices(pairs, target, world=world, indices=subset) == candidates[:2]


def test_manifest_preserves_seeds_balanced_tables_freshness_and_no_raw_fact_ids(packet):
    manifest = data.protocol_manifest(32)
    assert manifest["seeds"]["confirmation_entities"] == 37042
    assert manifest["counts"] == {"train": 32, "dev": 64, "confirmation": 64}
    assert manifest["reserved_confirmation_families"] == ["JSON object", "Markdown table"]
    for world, labels in manifest["confirmation_answer_style_crosstab"].items():
        assert world in {"A", "B"}
        for cells in labels.values():
            assert len(cells) == 4 and set(cells.values()) == {1}
    assert len(manifest["historical_entity_exclusion_sha256"]) == 64
    assert all(pair.id not in str(manifest) for pair in packet["confirmation"])
    assert len(data.protocol_fingerprint(32)) == 64
    assert data.protocol_fingerprint(32) == data.protocol_fingerprint(32)
    assert data.protocol_fingerprint(16) != data.protocol_fingerprint(32)


def test_tokenizer_preflight_rechecks_lengths_and_distinct_first_tokens():
    tokens = {word: [100+index]+([200+index] if data.ANSWER_TOKEN_LENGTHS[word] == 2 else [])
              for index, word in enumerate(MEMORY_WORDS)}
    data.validate_tokenization(tokens)
    with pytest.raises(ValueError, match="all 16"):
        data.validate_tokenization({"apple": [100]})
    with pytest.raises(ValueError, match="length changed"):
        data.validate_tokenization({**tokens, "apple": [100, 101]})
    with pytest.raises(ValueError, match="first-token"):
        data.validate_tokenization({**tokens, "apple": tokens["river"]})


@pytest.mark.parametrize("size", [0, 15, 17, 4097, True, 32.0])
def test_invalid_training_counts_are_rejected(size):
    with pytest.raises(ValueError, match="multiple of 16"):
        data.datasets(size)


def test_invalid_banks_templates_and_counterfactual_worlds_are_rejected(packet):
    pairs, pair = packet["train"], packet["train"][0]
    for indices in ([], [1, 2], [0, 0], [-1, 0], [0, 5000]):
        with pytest.raises(ValueError, match="indices"):
            data.paired_bank(pairs, 0, indices)
    with pytest.raises(ValueError, match="target_index"):
        data.paired_bank(pairs, True)
    for bad in ("train_support_00", "cf_confirmation_support_json"):
        with pytest.raises(ValueError, match="different support style"):
            data.paired_bank(pairs, 0, [0, 1], paraphrase_template_id=bad)
    with pytest.raises(ValueError, match="Expected query"):
        data.render_question(pair, "train_support_00")
    with pytest.raises(ValueError, match="World"):
        data.render_support(pair, "train_support_00", world="unknown")
    with pytest.raises(ValueError, match="share ID"):
        data.CounterfactualPair(pair.a, replace(pair.b, id="different"))
    with pytest.raises(ValueError, match="different vocabulary"):
        data.CounterfactualPair(pair.a, replace(pair.b, answer=pair.a.answer))
    with pytest.raises(ValueError, match="same token length"):
        other = next(word for word in MEMORY_WORDS if data.ANSWER_TOKEN_LENGTHS[word] != data.ANSWER_TOKEN_LENGTHS[pair.a.answer])
        data.CounterfactualPair(pair.a, replace(pair.b, answer=other))
    with pytest.raises(ValueError, match="exactly 64"):
        data.confirmation_view_assignments(packet["confirmation"][:32])
