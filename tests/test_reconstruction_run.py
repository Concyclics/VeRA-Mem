"""Training-cache isolation and tokenizer-derived payload features, CPU only."""
from contextlib import nullcontext
import copy
import re
from types import SimpleNamespace

import pytest
import torch

from vera_mem import reconstruction_data as data
from vera_mem import reconstruction_run as run
from vera_mem.reconstruction_eval import changed_group, shuffled, phases, worlds
from vera_mem.reconstruction_vera import GroupedVectorDB, ReconstructionVeRA


class OffsetTokenizer:
    """Small offset-preserving tokenizer with leading spaces and split words."""
    def __init__(self, merge_answer=None):
        self.merge_answer = merge_answer
        self.last_text = None
        self.last_offsets = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"

    def __call__(self, text, **_kwargs):
        offsets = []
        for match in re.finditer(r"\s*\w+|[^\w\s]", text):
            start, stop = match.span()
            if len(match.group().strip()) >= 6:
                middle = stop - 2
                offsets.extend(((start, middle), (middle, stop)))
            else:
                offsets.append((start, stop))
        if self.merge_answer:
            start = text.index(self.merge_answer)
            stop = start + len(" ".join(self.merge_answer.split()[:2]))
            indexes = [i for i, (a, b) in enumerate(offsets) if b > start and a < stop]
            offsets[indexes[0]:indexes[-1] + 1] = [(offsets[indexes[0]][0], stop)]
        self.last_text, self.last_offsets = text, offsets
        return dict(input_ids=list(range(1, len(offsets) + 1)), offset_mapping=offsets)


def test_payload_offsets_select_only_observed_words_and_exclude_entity_prefix_and_chat():
    row = data.datasets()["train"][0]
    tokenizer = OffsetTokenizer()
    ids, masks = run.payload_encoding(tokenizer, row["supports"][0][0], row["a"])
    assert len(masks) == 3 and all(len(m) == len(ids) for m in masks)
    assert all(sum(x) <= 1 for x in zip(*masks))
    for word, mask in zip(row["a"].split(), masks):
        pieces = [tokenizer.last_text[a:b] for (a, b), selected in zip(tokenizer.last_offsets, mask) if selected]
        assert "".join(pieces).strip() == word
        assert row["entity"] not in "".join(pieces)
        assert "<assistant>" not in "".join(pieces)
    assert any(sum(mask) > 1 for mask in masks)


def test_payload_offsets_reject_tokens_crossing_word_boundaries_and_missing_payload():
    row = data.datasets()["train"][0]
    with pytest.raises(ValueError, match="crosses payload word boundary"):
        run.payload_encoding(OffsetTokenizer(row["a"]), row["supports"][0][0], row["a"])
    with pytest.raises(ValueError, match="exactly one"):
        run.payload_encoding(OffsetTokenizer(), row["supports"][0][0], row["b"])


class FeatureBackend:
    device = torch.device("cpu")
    def __init__(self):
        self.tokenizer = OffsetTokenizer()
        self.target = torch.nn.Identity()
        self.model = SimpleNamespace(model=self.forward)
        self.calls = 0

    def disabled(self):
        return nullcontext()

    def prompt_ids(self, text):
        messages = [dict(role="system", content="You are a helpful assistant. Follow the requested answer format."),
                    dict(role="user", content=text)]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(rendered)["input_ids"]

    def _right_pad(self, sequences):
        ids = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, seq in enumerate(sequences):
            ids[index, :len(seq)] = torch.tensor(seq)
            mask[index, :len(seq)] = 1
        return ids, mask

    def forward(self, input_ids, attention_mask, use_cache):
        assert not use_cache
        self.calls += 1
        features = torch.stack((input_ids.float(), input_ids.float().square()), -1)
        self.target(features)


def test_payload_features_pool_one_token_weighted_slot_and_three_word_slots():
    row = data.datasets()["train"][0]
    supports = [row["supports"][0][0], row["supports"][1][1]]
    answers = [row["a"], row["b"]]
    backend = FeatureBackend()
    one, three, counts = run.payload_features(backend, supports, answers)
    assert one.shape == (2, 1, 2) and three.shape == (2, 3, 2)
    assert backend.calls == 1 and not backend.target._forward_pre_hooks
    for index, (support, answer) in enumerate(zip(supports, answers)):
        ids, masks = run.payload_encoding(backend.tokenizer, support, answer)
        x = torch.stack((torch.tensor(ids).float(), torch.tensor(ids).float().square()), -1)
        masks = torch.tensor(masks, dtype=torch.bool)
        assert counts[index] == masks.sum(-1).tolist()
        assert torch.equal(one[index, 0], x[masks.any(0)].mean(0))
        assert torch.equal(three[index], torch.stack([x[m].mean(0) for m in masks]))
        weights = masks.sum(-1).float()
        assert torch.allclose(one[index, 0], (three[index] * weights[:, None]).sum(0) / weights.sum())


@pytest.mark.parametrize("slots", [1, 3])
def test_independent_training_banks_change_only_their_own_target_group(slots):
    features = torch.arange(8 * 2 * slots * 4).reshape(8, 2, 1, slots, 4).float()
    original = features.clone()
    targets = [1, 6, 3]
    a = run.independent_features(features, targets, 0)
    b = run.independent_features(features, targets, 1)
    assert a.shape == b.shape == (3, 8, slots, 4)
    for batch, target in enumerate(targets):
        assert torch.equal(a[batch], features[:, 0, 0])
        assert torch.equal(b[batch, target], features[target, 1, 0])
        changed = (a[batch] != b[batch]).any(-1).any(-1).nonzero().flatten().tolist()
        assert changed == [target]
    b[0, 0] = -100
    assert torch.equal(features, original)


@pytest.fixture
def training_packet():
    rows = data.training_records(data.datasets()["train"])
    return dict(protocol=run.PROTOCOL, model_revision=run.REVISION, writer_prefix=data.WRITER_PREFIX,
                split="train", worlds=["A", "B"], views=1, rows=rows, rows_sha256=data.digest(rows),
                q=torch.arange(16 * 9728).reshape(16, 1, 9728).float(),
                slots1=torch.arange(16 * 2 * 9728).reshape(16, 2, 1, 1, 9728).float(),
                slots3=torch.arange(16 * 2 * 3 * 9728).reshape(16, 2, 1, 3, 9728).float(),
                payload_token_counts=[[2, 1, 1] for _ in range(32)])


class StatisticsOnlyModule:
    def __init__(self, *args, **kwargs):
        self.slots = kwargs["slots"]
    def to(self, device):
        self.device = torch.device(device)
        return self
    def fit_statistics(self, q, support):
        self.query_statistics_input = q.clone()
        self.support_statistics_input = support.clone()
    def fit_value_statistics(self, support):
        self.value_statistics_input = support.clone()


@pytest.mark.parametrize("slots", [1, 3])
def test_initialize_fits_exactly_canonical_ab_tensor_axes(training_packet, monkeypatch, slots):
    monkeypatch.setattr(run, "ReconstructionVeRA", StatisticsOnlyModule)
    model = run.initialize(training_packet, SimpleNamespace(seed=71042, slots=slots, learned_b=False), "cpu")
    assert torch.equal(model.query_statistics_input, training_packet["q"][:, 0])
    expected = training_packet["slots" + str(slots)].flatten(0, 3)
    assert torch.equal(model.support_statistics_input, expected)
    assert torch.equal(model.value_statistics_input, expected)
    assert expected.shape == (32 * slots, 9728)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(split="known"),
    lambda p: p.update(worlds=["A", "B", "C", "D", "SWAP"]),
    lambda p: p.update(views=2),
    lambda p: p.update(protocol="other"),
    lambda p: p.update(model_revision="unsealed-model"),
    lambda p: p.update(writer_prefix="insert gold here"),
    lambda p: p.update(rows_sha256="0" * 64),
    lambda p: p.update(q=p["q"][:, :, :-1]),
    lambda p: p.update(q=p["q"].long()),
    lambda p: p.update(slots1=p["slots1"][:, :, :, 0]),
    lambda p: p.update(slots3=p["slots3"][:, :, :, :2]),
    lambda p: p["q"].__setitem__((0, 0, 0), float("nan")),
    lambda p: p["slots3"].__setitem__((0, 0, 0, 0, 0), float("inf")),
    lambda p: p["slots1"].requires_grad_(True),
    lambda p: p.update(payload_token_counts=p["payload_token_counts"][:-1]),
    lambda p: p["payload_token_counts"][0].__setitem__(0, 0),
    lambda p: p["payload_token_counts"][0].__setitem__(0, True),
    lambda p: p["rows"][0].update(c="window ivory badger"),
])
def test_initialize_rejects_eval_or_malformed_cache_before_any_statistics_fit(training_packet, monkeypatch, mutation):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Must reject before constructing a trainable module")
    monkeypatch.setattr(run, "ReconstructionVeRA", forbidden)
    mutation(training_packet)
    with pytest.raises(ValueError):
        run.initialize(training_packet, SimpleNamespace(seed=71042, slots=1, learned_b=False), "cpu")


def make_store(slots=3):
    db = GroupedVectorDB(4, 2, slots=slots)
    for i in range(4):
        keys = torch.arange(slots * 4).reshape(slots, 4).float() + i + 1
        values = torch.arange(slots * 2).reshape(slots, 2).float() + 10 * i
        db.write_group(f"fact-{i}", keys, values, 0)
    return db


def test_group_update_and_restore_preserve_other_facts_but_advance_target_timestamp():
    db = make_store(); base = db.snapshot(); original_hash = db.hash()
    db.write_group("fact-1", torch.ones(3, 4), torch.full((3, 2), 99.), 1)
    changed_group(base, db.snapshot(), 1, 3)
    keys = base["store"]["keys"][3:6]
    values = base["store"]["values"][3:6]
    db.write_group("fact-1", keys, values, 2)
    changed_group(base, db.snapshot(), 1, 3)
    assert torch.equal(db.values, base["store"]["values"])
    assert db.timestamps[3:6] == (2, 2, 2)
    assert db.hash() != original_hash  # A restored payload has a later timestamp.


def test_group_assertion_rejects_a_changed_other_payload_or_timestamp():
    db = make_store(); base = db.snapshot()
    db.write_group("fact-1", torch.ones(3, 4), torch.ones(3, 2), 1)
    wrong = db.snapshot(); wrong["store"]["values"][0, 0] += 1
    with pytest.raises(AssertionError, match="Non-target"):
        changed_group(base, wrong, 1, 3)


@pytest.mark.parametrize("mutation", [
    lambda p: p["store"]["timestamps"].__setitem__(3, 0),
    lambda p: p["store"]["timestamps"].__setitem__(slice(3, 6), [0, 0, 0]),
    lambda p: p.update(slots=1),
    lambda p: p["store"]["config"].update(top_k=1),
    lambda p: p["store"].update(values=p["store"]["values"].double()),
    lambda p: p["store"].update(keys=p["store"]["keys"][:-1]),
    lambda p: p["store"]["values"].__setitem__((0, 0), -0.0),
])
def test_group_assertion_rejects_partial_stale_or_non_bitwise_atomic_update(mutation):
    db = make_store(); base = db.snapshot()
    assert base["store"]["values"][0, 0] == 0
    db.write_group("fact-1", torch.ones(3, 4), torch.ones(3, 2), 1)
    candidate = db.snapshot()
    mutation(candidate)
    with pytest.raises(AssertionError):
        changed_group(base, candidate, 1, 3)
    wrong = db.snapshot(); wrong["store"]["timestamps"][0] = 3
    with pytest.raises(AssertionError, match="Non-target"):
        changed_group(base, wrong, 1, 3)


def test_shuffle_deranges_whole_values_preserving_keys_slots_ids_and_timestamps():
    db = make_store(); before = db.snapshot()
    wrong, permutation = shuffled(db, 111042)
    assert sorted(permutation) == list(range(4))
    assert all(i != j for i, j in enumerate(permutation))
    assert torch.equal(wrong.keys, db.keys)
    assert wrong.ids == db.ids and wrong.fact_ids == db.fact_ids and wrong.timestamps == db.timestamps
    grouped = db.values.reshape(4, 3, 2)
    assert torch.equal(wrong.values.reshape(4, 3, 2), grouped[permutation])
    assert torch.equal(db.values, before["store"]["values"])


def test_development_and_confirmation_keep_sealed_d_out_of_development_generations():
    known = dict(split="known")
    assert phases(known, "confirmation") == ["CC"]
    assert worlds(known, "confirmation", "CC") == ["D"]
    for phase in phases(known, "development"):
        assert "D" not in worlds(known, "development", phase)
    assert phases(dict(split="dev"), "development") == ["CC"]
    with pytest.raises(ValueError):
        phases(dict(split="confirm"), "development")


class TinySequenceBackend:
    """Real sparse adapter, frozen toy token inputs, unequal target lengths."""
    device = torch.device("cpu")
    def __init__(self):
        targets = {"a0": [0], "a1": [1, 2], "a2": [2, 1, 0], "a3": [1],
                   "b0": [3, 4, 5], "b1": [4], "b2": [5, 3], "b3": [3, 5, 4]}
        self.tokenizer = SimpleNamespace(eos_token_id=6,
            encode=lambda text, **_kwargs: list(targets[text]))
        self.calls = 0

    def prompt_ids(self, question):
        return [0] * (int(question[1:]) + 2)

    def forward_sequences(self, questions, continuations):
        self.calls += 1
        counts = torch.tensor([len(t) for t in continuations])
        positions = torch.arange(int(counts.max())).float()
        base = torch.tensor([.7, -.2, .4, 1.1])
        x = torch.stack([base + int(q[1:]) * torch.tensor([.11, -.13, .17, -.07])
                         + positions[:, None] * torch.tensor([.04, .03, -.02, .01]) for q in questions])
        all_logits = self.vector_vera(x, self.vector_keys, self.vector_values)
        all_logits = all_logits + torch.linspace(-.15, .15, 7)
        rows = torch.repeat_interleave(torch.arange(len(questions)), counts)
        pos = torch.cat([torch.arange(n) for n in counts.tolist()])
        return dict(logits=all_logits[rows, pos], query_inputs=x[rows, pos],
                    batch_indices=rows, counts=counts)


@pytest.mark.parametrize("slots,learned_b", [(1, False), (3, False), (1, True), (3, True)])
def test_combined_ab_training_matches_two_independent_forward_losses_and_gradients(tmp_path, monkeypatch, slots, learned_b):
    rng = torch.Generator().manual_seed(821)
    features = torch.randn(4, 2, 1, slots, 4, generator=rng)
    model = ReconstructionVeRA(4, 7, rank=3, key_dim=3, top_k=2, temperature=.6,
                               seed=827, slots=slots, train_B=learned_b)
    model.fit_statistics(torch.randn(4, 4, generator=rng), features.flatten(0, 3))
    model.fit_value_statistics(features.flatten(0, 3))
    with torch.no_grad(): model.b.fill_(.3)  # Exercise nonzero writer/readout gradients.
    reference = copy.deepcopy(model)
    rows = [dict(id=str(i), questions=[f"q{i}"], a=f"a{i}", b=f"b{i}") for i in range(4)]
    packet = dict(rows=rows, **{"slots" + str(slots): features})
    args = SimpleNamespace(seed=71042, updates=1, batch_size=2, slots=slots, learned_b=learned_b,
                           run_dir=tmp_path, cache=tmp_path / "cache.pt")
    args.cache.write_bytes(b"sealed toy cache")
    monkeypatch.setattr(run, "initialize", lambda *_args: model)
    backend = TinySequenceBackend()
    result = run.train(backend, packet, args)
    log = json_read(tmp_path / "training.jsonl")
    targets = log["targets"]
    reference_metrics = []
    reference_backend = TinySequenceBackend()
    for world, field in enumerate(("a", "b")):
        # Independent construction, without reusing the runner's bank helper.
        bank = features[:, 0, 0].unsqueeze(0).repeat(len(targets), 1, 1, 1)
        if world:
            for batch, target in enumerate(targets): bank[batch, target] = features[target, 1, 0]
        keys, values = reference.encode_bank(bank)
        reference_backend.vector_vera = reference
        reference_backend.vector_keys, reference_backend.vector_values = keys, values
        tokens = run.gold_tokens(reference_backend, [rows[i][field] for i in targets])
        branch = reference_backend.forward_sequences([f"q{i}" for i in targets], tokens)
        ce, _ = run.supervised_loss(branch, tokens)
        tt = torch.tensor(targets)[branch["batch_indices"]]
        losses = reference.group_address_loss(branch["query_inputs"], tt, keys,
                                               batch_indices=branch["batch_indices"], reduction="none")
        address = run.sequence_mean(losses, branch["batch_indices"], branch["counts"])
        ((ce + .2 * address) / 2).backward()
        reference_metrics.append((float(ce.detach()), float(address.detach())))
    groups = [list(reference.Wq.parameters()) + list(reference.Wk.parameters()),
              list(reference.Wv.parameters()), [reference.b]]
    if learned_b: groups.append([reference.B])
    for params in groups: torch.nn.utils.clip_grad_norm_(params, 1.)
    assert backend.calls == result["totals"]["backbone_calls"] == 1
    assert reference_backend.calls == 2
    assert result["totals"]["target_exposures"] == len(targets)
    for logged, (ce, address) in zip(log["metrics"], reference_metrics):
        assert logged["ce"] == pytest.approx(ce, abs=1e-6)
        assert logged["address"] == pytest.approx(address, abs=1e-6)
    for (name, parameter), (ref_name, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == ref_name and parameter.grad is not None and expected.grad is not None
        assert torch.allclose(parameter.grad, expected.grad, atol=2e-6, rtol=2e-5), name


def json_read(path):
    import json
    return json.loads(path.read_text())
