"""Frozen data, answer separation and answer-free rendering for interface probes."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
import random

import pytest

from vera_mem import interface_data as data


def test_entities_relations_ids_and_complete_answers_are_disjoint_across_splits():
    splits = data.datasets(train_size=128, dev_size=32, confirm_size=32)
    data.validate_splits(splits)
    seen_entities, seen_answers, seen_ids = set(), set(), set()
    for split, rows in splits.items():
        entities = {f.entity for f in rows}
        answers = {f.answer(world) for f in rows for world in ("A", "B")}
        ids = {f.id for f in rows}
        assert not entities & seen_entities
        assert not answers & seen_answers
        assert not ids & seen_ids
        assert len(ids) == len(rows)
        by_entity = defaultdict(set)
        for fact in rows:
            by_entity[fact.entity].add(fact.relation)
            assert fact.split == split and fact.stable_id == fact.id
            assert fact.id == data.stable_id(fact.entity, fact.relation)
            assert len(fact.answer_a.split()) == len(fact.answer_b.split()) == 3
            assert fact.answer_a.split()[0] != fact.answer_b.split()[0]
        assert all(relations == set(data.RELATIONS) for relations in by_entity.values())
        assert len(entities) == len(rows) // 4
        seen_entities |= entities
        seen_answers |= answers
        seen_ids |= ids


def test_nested_data_budgets_keep_same_labels_and_do_not_consume_global_rng():
    random.seed(88)
    before = random.getstate()
    small = data.datasets(32, 16, 16)
    assert random.getstate() == before
    larger = data.datasets(128, 32, 32)
    assert small["train"] == larger["train"][:32]
    assert small["dev"] == larger["dev"][:16]
    assert small["confirm"] == larger["confirm"][:16]
    assert data.datasets(32, 16, 16) == small
    for split, rows in small.items():
        assert set(f.answer_a.split()[0] for f in rows) <= set(data.FIRST_WORDS)


def test_every_fact_has_both_answer_hard_negatives_and_same_entity_relation_negatives():
    for facts in data.datasets(128, 32, 32).values():
        by_id = {f.id: f for f in facts}
        for world in ("A", "B"):
            negatives = data.hard_negative_ids(facts, world)
            for fact in facts:
                assert len(negatives[fact.id]) == 1
                other = by_id[negatives[fact.id][0]]
                assert other.entity != fact.entity and other.relation == fact.relation
                assert other.answer(world) == fact.answer(world)
            actual = data.original_bank_hard_negative_ids(facts, world)
            for fact in facts:
                # The actual B bank changes just this target; every other
                # entry retains its original A value, not its B alternative.
                assert len(actual[fact.id]) == (1 if world == "A" else 2)
                actual_bank = {f.id: f.answer_a for f in facts}
                actual_bank[fact.id] = fact.answer(world)
                for identifier in actual[fact.id]:
                    other = by_id[identifier]
                    assert other.entity != fact.entity and other.relation == fact.relation
                    assert actual_bank[other.id] == actual_bank[fact.id]
        relation_negatives = data.relation_negative_ids(facts)
        for fact in facts:
            assert len(relation_negatives[fact.id]) == 3
            for identifier in relation_negatives[fact.id]:
                other = by_id[identifier]
                assert other.entity == fact.entity and other.relation != fact.relation
                assert other.answer_a != fact.answer_a and other.answer_b != fact.answer_b


def test_questions_do_not_change_when_answers_or_worlds_change_and_hide_record_id():
    fact = data.datasets(16, 16, 16)["train"][0]
    changed = replace(fact, answer_a=fact.answer_b, answer_b=fact.answer_a)
    for split in ("train", "dev", "confirm"):
        for template in data.get_templates(split, "query"):
            questions = [data.render_question(f, template, world=world)
                         for f in (fact, changed) for world in ("A", "B", "P")]
            assert len(set(questions)) == 1
            text = questions[0]
            assert fact.entity in text and fact.relation in text
            assert fact.id not in text
            assert fact.answer_a not in text and fact.answer_b not in text
            assert text.endswith(data.ANSWER_INSTRUCTION)
    poison = fact.a
    poison.answer, poison.question, poison.support, poison.id = "POISON_LABEL", "POISON_Q", "POISON_SUPPORT", "POISON_ID"
    assert data.render_question(poison, "if_train_query_00") == data.render_question(fact, "if_train_query_00")


def test_support_changes_only_selected_world_and_example_adapter_preserves_identity():
    fact = data.datasets(16, 16, 16)["train"][0]
    a, b = fact.a, fact.b
    assert a.id == b.id == fact.id and a.metadata == b.metadata
    assert a.question == b.question and a.paraphrase == b.paraphrase
    assert a.answer == fact.answer_a and b.answer == fact.answer_b
    assert fact.original == a and fact.alternative == b
    for split in ("train", "dev", "confirm"):
        for template in data.get_templates(split, "support"):
            original = data.render_support(fact, template, world="A")
            alternative = data.render_support(fact, template, world="B")
            assert data.render_support(fact, template, world="P") == original
            assert fact.answer_a in original and fact.answer_b not in original
            assert fact.answer_b in alternative and fact.answer_a not in alternative
            assert alternative == original.replace(fact.answer_a, fact.answer_b)
            assert fact.id not in original and fact.id not in alternative
            assert data.writer_text(fact, template) == data.WRITER_PREFIX + original
            assert data.render_support(b, template) == alternative


def test_confirmation_has_four_reserved_structures_and_valid_json():
    facts = data.datasets(16, 16, 16)["confirm"]
    for kind in ("query", "support"):
        assert {t.family for t in data.get_templates("confirm", kind)} == {"json", "markdown", "yaml", "ini"}
        assert data.get_templates("confirmation", kind) == data.get_templates("confirm", kind)
        ordinary = data.get_templates("train", kind) + data.get_templates("dev", kind)
        assert not {t.family for t in ordinary} & {"json", "markdown", "yaml", "ini"}
    record = json.loads(data.render_support(facts[0], "if_confirm_support_json", world="B"))
    assert record == dict(entity=facts[0].entity, relation=facts[0].relation, value=facts[0].answer_b)
    assert len(data.template_ids()["train"]["q"]) == 8
    assert len(data.template_ids()["train"]["s"]) == 4
    assert data.MAX_NEW_TOKENS == 32


def test_counts_and_split_integrity_reject_dropped_hard_negatives_and_overlap():
    for invalid in (0, 1, 7, 8, 9, 24, True, 4097):
        with pytest.raises(ValueError, match="multiple"):
            data.datasets(train_size=invalid)
    original = data.datasets(16, 16, 16)
    broken = dict(original, train=original["train"][:4])
    with pytest.raises(ValueError, match="hard negative"):
        data.validate_splits(broken)
    # Reproduce the former defect: each world alone has negatives, but B
    # values have no other original A-bank entry. Reject this arrangement.
    by_relation = {}
    for fact in original["train"]:
        by_relation.setdefault(fact.relation, (fact.answer_a, fact.answer_b))
    formerly_valid = [replace(f, answer_a=by_relation[f.relation][0],
                             answer_b=by_relation[f.relation][1]) for f in original["train"]]
    assert all(data.hard_negative_ids(formerly_valid, "B").values())
    with pytest.raises(ValueError, match="original-bank"):
        data.validate_splits(dict(original, train=formerly_valid))
    broken = dict(original, dev=[replace(f, split="dev") for f in original["train"][:8]])
    with pytest.raises(ValueError, match="identity|overlap"):
        data.validate_splits(broken)
    with pytest.raises(ValueError, match="ID"):
        replace(original["train"][0], id="answer-dependent-id")


def test_tokenizer_contract_measures_actual_tokens_and_reserves_eos():
    facts = data.datasets(16, 16, 16)["train"]
    vocabulary = {word: i for i, word in enumerate(data.FIRST_WORDS + data.TAIL_WORDS)}
    encode = lambda text: [vocabulary[word] for word in text.split()]
    result = data.validate_tokenization(facts, encode)
    assert result["min_answer_tokens"] == result["max_answer_tokens"] == 3
    assert result["unique_answers"] == 8 and result["unequal_length_pairs"] == 0
    assert result["eos_budget_reserved"] and result["max_new_tokens"] == 32
    with pytest.raises(ValueError, match="multi-token"):
        data.validate_tokenization(facts, lambda _text: [1])
    with pytest.raises(ValueError, match="EOS-budget"):
        data.validate_tokenization(facts, lambda _text: [1] * 32)
    with pytest.raises(ValueError, match="first-token"):
        data.validate_tokenization(facts, lambda text: [1, *encode(text)])


def test_manifest_records_seeds_novel_answer_contract_and_entity_uncertainty_unit():
    manifest = data.protocol_manifest(32, 16, 16)
    assert manifest["counts"] == dict(train=32, dev=16, confirm=16)
    assert manifest["entity_counts"] == dict(train=8, dev=4, confirm=4)
    assert manifest["seeds"] == data.SEEDS
    assert manifest["complete_answers_disjoint"] and manifest["component_vocabulary_shared"]
    assert manifest["uncertainty_unit"].startswith("entity")
    assert all(len(value) == 64 for value in manifest["fact_sha256"].values())
    assert manifest == data.protocol_manifest(32, 16, 16)
