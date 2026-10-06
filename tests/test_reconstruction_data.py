"""Controls against memorized bindings, leaked probe text and bad payload spans."""
from collections import Counter
import copy
import json
import random
import re

import pytest

from vera_mem import reconstruction_data as data


@pytest.fixture
def splits():
    return data.datasets()


def test_default_split_identity_and_complete_answer_separation(splits):
    assert {k: len(v) for k, v in splits.items()} == data.DEFAULT_COUNTS
    entities, identities, answers = set(), set(), set()
    for split, rows in splits.items():
        current_entities = {r["entity"] for r in rows}
        current_ids = {r["id"] for r in rows}
        current_answers = {r[w] for r in rows for w in ("a", "b", "c", "d")}
        assert len(current_entities) == len(current_ids) == len(rows)
        assert len(current_answers) == 4 * len(rows)
        assert not entities & current_entities
        assert not identities & current_ids
        assert not answers & current_answers
        assert all(r["split"] == split and r["id"] == data.stable_id(r["entity"]) for r in rows)
        entities.update(current_entities); identities.update(current_ids); answers.update(current_answers)
    assert data.validate_records(splits)["component_vocabulary_seen_in_training"]


def test_all_probe_words_are_seen_in_train_and_each_train_world_is_balanced(splits):
    vocabularies = (data.FIRST_WORDS, data.SECOND_WORDS, data.THIRD_WORDS)
    for field in ("a", "b"):
        for position, vocabulary in enumerate(vocabularies):
            counts = Counter(r[field].split()[position] for r in splits["train"])
            assert counts == Counter({word: 1 for word in vocabulary})
    for rows in splits.values():
        for r in rows:
            for field in ("a", "b", "c", "d", "swap_a"):
                assert all(word in vocabulary for word, vocabulary in zip(r[field].split(), vocabularies))


def test_fresh_c_and_sealed_d_change_only_third_word_and_are_never_ab_labels(splits):
    ab = {r[w] for rows in splits.values() for r in rows for w in ("a", "b")}
    seen_c, seen_d = set(), set()
    for rows in splits.values():
        for row in rows:
            a, b, c, d = (row[w].split() for w in ("a", "b", "c", "d"))
            assert a[:2] == c[:2] == d[:2]
            assert c[2] not in {a[2], b[2], d[2]}
            assert d[2] not in {a[2], b[2]}
            assert row["c"] not in ab and row["d"] not in ab
            seen_c.add(row["c"]); seen_d.add(row["d"])
    assert seen_c.isdisjoint(seen_d)


def test_swap_is_single_target_intervention_with_original_bank_donor(splits):
    for rows in splits.values():
        by_id = {r["id"]: r for r in rows}
        original_bank = {r["id"]: r["a"] for r in rows}
        for row in rows:
            donor = by_id[row["provenance"]["swap_donor_id"]]
            assert donor["id"] != row["id"]
            assert donor["provenance"]["swap_donor_id"] == row["id"]
            assert row["swap_a"] == donor["a"]
            updated = dict(original_bank, **{row["id"]: row["swap_a"]})
            assert {k for k in updated if updated[k] != original_bank[k]} == {row["id"]}
            assert updated[donor["id"]] == updated[row["id"]]
            pairs = data.evaluation_pairs([row], "swap_a")[0]
            assert pairs["questions"] == row["questions"]
            assert pairs["entity"] == row["entity"]
            assert donor["entity"] not in " ".join(pairs["questions"])
            assert all(row["entity"] in s and donor["entity"] not in s for s in pairs["supports"][1])


def test_all_questions_hide_all_target_words_ids_and_worlds(splits):
    all_words = set(data.FIRST_WORDS + data.SECOND_WORDS + data.THIRD_WORDS)
    for rows in splits.values():
        for row in rows:
            for question in row["questions"]:
                assert row["entity"] in question and row["id"] not in question
                assert question.endswith(data.INSTRUCTION)
                assert set(re.findall(r"[a-z]+", question.casefold())).isdisjoint(all_words)
                assert all(row[w] not in question for w in ("a", "b", "c", "d", "swap_a"))
            assert row["questions"][0] != row["questions"][1]


def test_training_projection_has_no_probe_labels_or_heldout_text_and_is_detached(splits):
    raw = splits["train"]
    projected = data.training_records(raw)
    wire = json.dumps(projected)
    for row, train in zip(raw, projected):
        assert set(train) == {"id", "entity", "relation", "split", "a", "b", "worlds", "questions", "supports", "provenance"}
        assert train["worlds"] == ["A", "B"]
        assert train["questions"] == [row["questions"][0]]
        assert train["supports"] == [[row["supports"][0][0]], [row["supports"][1][0]]]
        assert train["provenance"]["projection"] == "canonical_AB_only"
        assert row["c"] not in wire and row["d"] not in wire
        assert row["questions"][1] not in wire
        assert "swap_donor" not in wire and "source_worlds" not in wire
    projected[0]["supports"][0][0] = "changed"
    assert raw[0]["supports"][0][0] != "changed"
    with pytest.raises(ValueError, match="training records"):
        data.training_records(splits["dev"])


@pytest.mark.parametrize("condition,world,field", [("known_ab", "B", "b"), ("fresh_c", "C", "c"),
    ("fresh_d", "D", "d"), ("swap_a", "SWAP", "swap_a")])
def test_two_world_adapter_keeps_background_a_and_target_identity(splits, condition, world, field):
    rows = splits["dev"]
    pairs = data.evaluation_pairs(rows, condition)
    for row, pair in zip(rows, pairs):
        assert pair["id"] == row["id"] and pair["entity"] == row["entity"]
        assert pair["a"] == row["a"] and pair["b"] == row[field]
        assert pair["worlds"] == pair["provenance"]["source_worlds"] == ["A", world]
        assert pair["supports"][0] == row["supports"][0]
        assert pair["supports"][1] == row["supports"][data.WORLDS.index(world)]
        assert pair["questions"] == row["questions"]
    pairs[0]["questions"][0] = "mutated"
    assert rows[0]["questions"][0] != "mutated"


def test_payload_span_is_exact_three_observed_words_for_every_world_and_format(splits):
    fields = ("a", "b", "c", "d", "swap_a")
    for rows in splits.values():
        for row in rows:
            for field, views in zip(fields, row["supports"]):
                for text in views:
                    start, end = data.payload_span(text, row[field])
                    assert text[start:end] == row[field]
                    assert row["entity"] not in text[start:end]
                    assert len(text[start:end].split()) == 3
                    assert start == text.rfind(row[field])
    answer = splits["train"][0]["a"]
    for broken in ("absent", answer + "; " + answer, "prefix" + answer, answer + "suffix"):
        with pytest.raises(ValueError, match="exactly one"):
            data.payload_span(broken, answer)


def test_repeatability_rng_isolation_and_manifest_sealing(splits):
    random.seed(909)
    state = random.getstate()
    assert data.datasets() == splits
    assert state == random.getstate()
    altered = data.datasets(seed=data.SEED + 1)
    assert altered != splits
    manifest = data.protocol_manifest(splits)
    assert manifest == data.protocol_manifest()
    assert manifest["counts"] == data.DEFAULT_COUNTS
    assert manifest["worlds"] == list(data.WORLDS)
    assert manifest["training_projection_sha256"] == data.digest(data.training_records(splits["train"]))
    assert all(len(x) == 64 for x in manifest["record_sha256"].values())
    assert "reciprocal SWAP" in manifest["uncertainty_unit"]
    assert "sealed D" in manifest["evaluation_scope"]


@pytest.mark.parametrize("size", [0, 1, 8, 15, 17, 129, True, 16.0])
def test_invalid_budgets_are_not_silently_rounded(size):
    with pytest.raises(ValueError, match="multiples of 16"):
        data.datasets(train_size=size)


def test_large_budget_is_a_new_valid_dataset_not_a_nested_claim():
    rows = data.datasets(32, 32, 64)
    assert len(rows["train"]) == 32
    assert data.validate_records(rows)["entity_ids_disjoint"]


def test_maximum_declared_budget_reserves_separate_c_and_d_combinations():
    rows = data.datasets(128, 128, 128)
    assert all(len(split) == 128 for split in rows.values())
    assert data.validate_records(rows)["complete_answers_disjoint"]


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x["train"][0].update(c=x["train"][0]["a"]), "overlap"),
    (lambda x: x["train"][0].update(d=x["train"][0]["c"]), "overlap"),
    (lambda x: x["train"][0]["questions"].__setitem__(0, "answer " + x["train"][0]["a"]), "Question"),
    (lambda x: x["train"][0]["supports"][2].__setitem__(0, x["train"][0]["supports"][0][0]), "Support"),
    (lambda x: x["train"][0]["provenance"].update(swap_donor_id=x["train"][0]["id"]), "SWAP"),
    (lambda x: x["train"][0].update(swap_a=x["train"][0]["b"]), "SWAP"),
    (lambda x: x["train"][0].update(id="answer-dependent"), "identity"),
    (lambda x: x["dev"].__setitem__(0, copy.deepcopy(x["train"][0])), "overlap"),
])
def test_semantic_corruption_is_rejected(splits, mutation, match):
    broken = copy.deepcopy(splits)
    mutation(broken)
    with pytest.raises(ValueError, match=match):
        data.validate_records(broken)


def test_tokenizer_preflight_sees_only_ab_and_never_repairs_labels(splits):
    train = data.training_records(splits["train"])
    vocab = {word: index for index, word in enumerate(data.FIRST_WORDS + data.SECOND_WORDS + data.THIRD_WORDS)}
    seen = []
    def encode(text):
        seen.append(text)
        return [vocab[w] for w in text.split()]
    proof = data.validate_tokenization(train, encode)
    assert seen == [row[w] for row in train for w in ("a", "b")]
    assert proof["min_answer_tokens"] == proof["max_answer_tokens"] == 3
    assert proof["eos_reserved"] and proof["unequal_length_pairs"] == 0
    with pytest.raises(ValueError, match="training projection"):
        data.validate_tokenization(splits["train"], encode)
    with pytest.raises(ValueError, match="training projection"):
        data.validate_tokenization(data.evaluation_pairs(splits["train"], "fresh_c"), encode)
    with pytest.raises(ValueError, match="EOS"):
        data.validate_tokenization(train, lambda _x: [1] * 32)
    with pytest.raises(ValueError, match="multi-token"):
        data.validate_tokenization(train, lambda _x: [1])


def test_strict_training_validator_accepts_projection_and_rejects_raw_eval(splits):
    projected = data.training_records(splits["train"])
    proof = data.validate_training_records(projected)
    assert proof["records"] == 16 and proof["worlds"] == ["A", "B"] and proof["views"] == 1
    assert proof["records_sha256"] == data.digest(projected)
    assert proof["canonical_only"] and proof["probe_fields_absent"]
    with pytest.raises(ValueError, match="forbidden"):
        data.validate_training_records(splits["train"])


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(c="window ivory badger"),
    lambda r: r.update(d="window ivory lizard"),
    lambda r: r.update(swap_a=r["a"]),
    lambda r: r.update(a_tokens=[1, 2, 3]),
    lambda r: r["provenance"].update(heldout_question="hidden prompt"),
    lambda r: r["provenance"].update(source_record_sha256="not-a-hash"),
    lambda r: r["provenance"].update(source_split="dev"),
    lambda r: r["questions"].append(r["questions"][0]),
    lambda r: r["supports"][0].append(r["supports"][0][0]),
    lambda r: r.update(worlds=["A", "C"]),
    lambda r: r.update(split="dev"),
    lambda r: r["questions"].__setitem__(0, "Answer: " + r["a"]),
    lambda r: r["supports"][1].__setitem__(0, r["supports"][0][0]),
])
def test_strict_training_validator_rejects_leakage_and_metadata_smuggling(splits, mutation):
    projected = data.training_records(splits["train"])
    mutation(projected[0])
    with pytest.raises(ValueError):
        data.validate_training_records(projected)
