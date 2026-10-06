"""Real episode invariants, split novelty, and causal single-target bank writes."""
from collections import Counter
import copy
import json
import random
import re

import pytest

from vera_mem import qkv_data as data
from vera_mem import reconstruction_data as old


@pytest.fixture(scope="module")
def raw():
    return data.dataset()


def test_dataset_counts_component_balance_and_historical_exclusion(raw):
    proof = data.validate_dataset(raw)
    assert proof["entities"] == 64 and proof["payloads"] == 128
    assert proof["novel_complete_payloads"] == 32
    assert proof["edit_position_counts"] == {0: 6, 1: 5, 2: 5}
    history = old.datasets(seed=101042)
    old_answers = {r[w] for rows in history.values() for r in rows for w in ("a", "b", "c", "d")}
    old_entities = {r["entity"] for rows in history.values() for r in rows}
    assert len(old_answers) == 448
    new_answers = set(raw["train"]["payloads"]) | {r[w] for r in raw["known"] for w in ("c", "d")}
    assert len(new_answers) == 160 and new_answers.isdisjoint(old_answers)
    entities = set(raw["train"]["entities"]) | {r["entity"] for s in ("dev", "confirm") for r in raw[s]}
    assert len(entities) == 96 and entities.isdisjoint(old_entities)
    for index, vocab in enumerate(data.VOCABULARIES):
        assert Counter(p.split()[index] for p in raw["train"]["payloads"]) == Counter({w: 8 for w in vocab})


def test_factorial_pairing_separates_entity_transfer_from_new_content(raw):
    train = raw["train"]
    assert [r["entity"] for r in raw["known"]] == train["entities"][:16]
    known_entities = {r["entity"] for r in raw["known"]}
    dev_entities = {r["entity"] for r in raw["dev"]}
    confirm_entities = {r["entity"] for r in raw["confirm"]}
    assert not known_entities & dev_entities and not known_entities & confirm_entities and not dev_entities & confirm_entities
    for i in range(16):
        for split in ("known", "dev", "confirm"):
            row = raw[split][i]
            assert row["a"] == train["payloads"][2*i]
            assert row["b"] == train["payloads"][2*i+1]
            assert [row[f] for f in data.FIELDS] == [raw["known"][i][f] for f in data.FIELDS]
            assert row["c"] != row["d"]
            for field in ("c", "d"):
                assert [j for j, (a, b) in enumerate(zip(row["a"].split(), row[field].split())) if a != b] == [i % 3]
                assert row[field] not in train["payloads"]
    assert data.validate_dataset(raw)["eval_payloads_intentionally_paired"]


def test_questions_hide_targets_and_writer_only_observes_its_bound_entity(raw):
    words = set(sum(data.VOCABULARIES, ()))
    for split in ("known", "dev", "confirm"):
        rows = raw[split]; ids = {r["id"]: r for r in rows}
        for row in rows:
            assert row["worlds"] == list(data.WORLDS)
            for question in row["questions"]:
                assert row["entity"] in question and row["id"] not in question
                assert words.isdisjoint(re.findall(r"[a-z]+", question.casefold()))
                assert all(row[f] not in question for f in data.FIELDS)
            for field, supports in zip(data.FIELDS, row["supports"]):
                for support in supports:
                    begin, end = data.payload_span(support, row[field])
                    assert support[begin:end] == row[field]
                    assert row["entity"] in support and row["id"] not in support
            donor = ids[row["provenance"]["swap_donor_id"]]
            assert donor["id"] != row["id"] and donor["provenance"]["swap_donor_id"] == row["id"]
            assert row["swap_a"] == donor["a"]
            assert all(donor["entity"] not in s for s in row["supports"][4])


def test_training_projection_contains_no_probe_entities_labels_or_templates(raw):
    train = data.training_data(raw)
    assert data.validate_training_data(train)["probe_fields_absent"]
    wire = json.dumps(train)
    for split in ("known", "dev", "confirm"):
        for row in raw[split]:
            assert row["c"] not in wire and row["d"] not in wire
            assert row["questions"][1] not in wire
            if split != "known":
                assert row["entity"] not in wire
    for row in train["rows"]:
        assert row["worlds"] == ["A", "B"] and len(row["questions"]) == 1
        assert [len(v) for v in row["supports"]] == [1, 1]
    train["payloads"][0] = "mutated"
    assert raw["train"]["payloads"][0] != "mutated"


def test_determinism_global_rng_isolation_and_new_seed(raw):
    random.seed(65); state = random.getstate()
    assert data.dataset() == raw and data.episode(17, "rebind") == data.episode(17, "rebind")
    assert random.getstate() == state
    assert data.dataset(seed=121043) != raw
    assert data.episode(17, "rebind", seed=81043) != data.episode(17, "rebind", seed=81042)
    mutated = data.dataset(); mutated["provenance"]["historical_counts"]["train"] = 999
    assert old.DEFAULT_COUNTS["train"] == 16


def test_static_and_rebind_share_every_target_background_and_array_position():
    for step in range(32):
        static, rebind = [data.episode(step, regime) for regime in data.REGIMES]
        for key in ("step", "epoch", "batch_index", "targets", "background", "bank_entities", "local_targets"):
            assert static[key] == rebind[key]
        for e in (static, rebind):
            assert len(e["targets"]) == len(set(e["targets"])) == 8
            assert len(e["bank_entities"]) == len(set(e["bank_entities"])) == 16
            assert set(e["targets"]).isdisjoint(e["background"])
            assert [e["bank_entities"][i] for i in e["local_targets"]] == e["targets"]
            assert data.validate_episode(e)["valid"]
        assert static["mapping"] == [[2*i, 2*i+1] for i in range(64)]
        assert rebind["mapping"] != static["mapping"]


def test_mapping_is_epoch_constant_and_changes_only_at_epoch_boundary():
    for epoch in range(4):
        packets = [data.episode(8*epoch+i, "rebind") for i in range(8)]
        assert all(p["mapping"] == packets[0]["mapping"] for p in packets)
        assert packets[0]["mapping"] != data.episode(8*(epoch+1), "rebind")["mapping"]
        assert sorted(t for p in packets for t in p["targets"]) == list(range(64))
        assert sorted(v for pair in packets[0]["mapping"] for v in pair) == list(range(128))


def test_full_2048_step_schedule_has_matched_entity_payload_and_gold_token_exposures(raw):
    # Use unequal artificial token lengths; equality follows the schedule, not
    # an assumption that three whitespace words always tokenize to three tokens.
    lengths = [sum(1 + len(w) % 3 for w in p.split()) + 1 for p in raw["train"]["payloads"]]
    totals = []
    for regime in data.REGIMES:
        entities, payloads, tokens = Counter(), Counter(), 0
        for ep in data.episodes(2048, regime):
            entities.update(ep["targets"])
            used = [p for entity in ep["targets"] for p in ep["mapping"][entity]]
            payloads.update(used); tokens += sum(lengths[p] for p in used)
        assert entities == Counter({i: 256 for i in range(64)})
        assert payloads == Counter({i: 256 for i in range(128)})
        totals.append((entities, payloads, tokens))
    assert totals[0] == totals[1]


@pytest.mark.parametrize("regime", data.REGIMES)
def test_each_b_branch_changes_only_its_target_and_preserves_entity_identity(raw, regime):
    train = raw["train"]
    for step in (0, 7, 8, 13):
        ep = data.episode(step, regime)
        before = dict(zip(ep["bank_entities"], ep["a_payload_indices"]))
        for target, local in zip(ep["targets"], ep["local_targets"]):
            after = list(ep["a_payload_indices"])
            after[local] = ep["b_payload_indices"][local]
            mapped = dict(zip(ep["bank_entities"], after))
            assert {e for e in mapped if mapped[e] != before[e]} == {target}
            assert mapped[target] == ep["mapping"][target][1]
            # Correct observation re-renders the current entity with its current
            # assigned payload; the cache may not reuse another entity's text.
            content = train["payloads"][mapped[target]]
            support = data.render_support(train["entities"][target], content)
            assert train["entities"][target] in support and content in support


@pytest.mark.parametrize("mutation", [
    lambda v: v["train"].update(c="hidden probe"),
    lambda v: v["train"]["rows"][0].update(c=v["known"][0]["c"]),
    lambda v: v["train"]["rows"][0]["questions"].append(v["known"][0]["questions"][1]),
    lambda v: v["train"]["rows"][0]["provenance"].update(eval_id=v["dev"][0]["id"]),
    lambda v: v["train"]["payloads"].__setitem__(0, v["train"]["payloads"][1]),
    lambda v: v["train"]["payloads"].__setitem__(0, []),
    lambda v: v["train"]["entities"].__setitem__(0, v["dev"][0]["entity"]),
    lambda v: v["known"][0].update(c=v["known"][0]["a"]),
    lambda v: v["known"][0].update(d=v["known"][0]["c"]),
    lambda v: v["dev"][0].update(c=v["dev"][1]["c"]),
    lambda v: v["confirm"][0].update(a=v["confirm"][1]["a"]),
    lambda v: v["known"][0]["questions"].__setitem__(0, "answer " + v["known"][0]["c"]),
    lambda v: v["known"][0]["supports"][2].__setitem__(0, v["known"][0]["supports"][0][0]),
    lambda v: v["known"][0]["provenance"].update(swap_donor_id=v["known"][0]["id"]),
    lambda v: v["provenance"].update(excluded_answers_sha256="0" * 64),
])
def test_dataset_semantic_corruption_is_rejected(raw, mutation):
    broken = copy.deepcopy(raw); mutation(broken)
    with pytest.raises(ValueError):
        data.validate_dataset(broken)


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(regime="other"),
    lambda e: e.update(step=-1),
    lambda e: e.update(seed=True),
    lambda e: e.update(undocumented_target="gold"),
    lambda e: e["targets"].__setitem__(0, e["targets"][1]),
    lambda e: e["background"].__setitem__(0, e["targets"][0]),
    lambda e: e["local_targets"].__setitem__(0, 16),
    lambda e: e["mapping"][0].__setitem__(0, e["mapping"][0][1]),
    lambda e: e["a_payload_indices"].reverse(),
    lambda e: e["b_payload_indices"].__setitem__(0, True),
    lambda e: e.update(batch_index=99),
])
def test_episode_corruption_cannot_smuggle_a_different_bank_or_target(mutation):
    episode = data.episode(0, "rebind"); mutation(episode)
    with pytest.raises(ValueError):
        data.validate_episode(episode)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "121042", 101042])
def test_bad_or_historical_data_seed_rejected(seed):
    with pytest.raises(ValueError):
        data.dataset(seed)


@pytest.mark.parametrize("step,regime,seed", [(-1, "static", 1), (True, "static", 1),
    (0, "other", 1), (0, "rebind", -1), (0, "rebind", True)])
def test_bad_schedule_identity_rejected(step, regime, seed):
    with pytest.raises(ValueError):
        data.episode(step, regime, seed)
