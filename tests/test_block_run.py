"""CPU contracts for contextual rebindings, joint training, and online writes."""
from contextlib import contextmanager
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from vera_mem import block_data as data
from vera_mem import block_run as run
from vera_mem import block_eval as evaluation
from vera_mem.block_vera import BlockVeRA, BlockVectorDB
from vera_mem.reconstruction_eval import changed_group
from vera_mem.run import tensor_digest


def toy_module(mode="diagonal"):
    rng = torch.Generator().manual_seed(902)
    m = BlockVeRA(4, 7, rank=3, key_dim=3, temperature=.6, seed=910, block_mode=mode)
    support = torch.randn(384, 4, generator=rng)
    m.fit_statistics(torch.randn(64, 4, generator=rng), support)
    m.fit_value_statistics(support)
    with torch.no_grad(): m.b.fill_(.3)
    return m


@pytest.mark.parametrize("regime", data.REGIMES)
def test_contextual_cross_cache_and_target_labels_stay_bound_to_current_entity(regime):
    # Values explicitly identify entity, payload, and slot independently. Moving
    # a donor entity's contextual hidden state cannot accidentally pass.
    matrix = torch.empty(64, 128, 3, 3)
    matrix[..., 0] = torch.arange(64)[:, None, None]
    matrix[..., 1] = torch.arange(128)[None, :, None]
    matrix[..., 2] = torch.arange(3)[None, None, :]
    original = matrix.clone(); packet = dict(payloads=[f"p{i}" for i in range(128)])
    for step in (0, 7, 8):
        ep = data.episode(step, regime)
        banks = run.independent_banks(matrix, ep)
        assert banks.shape == (16, 16, 3, 3)
        answers = run.target_answers(packet, ep)
        for batch, (entity, local) in enumerate(zip(ep["targets"], ep["local_targets"])):
            for row, other_entity in enumerate(ep["bank_entities"]):
                expected_a = matrix[other_entity, ep["mapping"][other_entity][0]]
                expected_b = matrix[other_entity, ep["mapping"][other_entity][int(row == local)]]
                assert torch.equal(banks[batch, row], expected_a)
                assert torch.equal(banks[batch+8, row], expected_b)
            assert answers[batch] == f"p{ep['mapping'][entity][0]}"
            assert answers[batch+8] == f"p{ep['mapping'][entity][1]}"
            changed = (banks[batch] != banks[batch+8]).any(-1).any(-1).nonzero().flatten().tolist()
            assert changed == [local]
        banks[0, 0] = -1
        assert torch.equal(matrix, original)


class TinySequenceBackend:
    """Frozen causal-position inputs and real sparse trainable QKV reads."""
    device = torch.device("cpu")
    def __init__(self):
        self.calls = 0
        self.tokenizer = SimpleNamespace(eos_token_id=6, encode=self.encode)

    @staticmethod
    def encode(text, **_kwargs):
        value = int(text[1:])
        return [(value+j) % 6 for j in range(1 + value % 3)]

    def prompt_ids(self, question):
        return [0] * (2 + int(question[1:]) % 5)

    def forward_sequences(self, questions, continuations):
        self.calls += 1
        counts = torch.tensor([len(c) for c in continuations])
        positions = torch.arange(int(counts.max())).float()
        base = torch.tensor([.7, -.2, .4, 1.1])
        x = torch.stack([base + int(q[1:]) * torch.tensor([.011, -.013, .017, -.007])
                         + positions[:, None] * torch.tensor([.04, .03, -.02, .01]) for q in questions])
        logits = self.vector_vera(x, self.vector_keys, self.vector_values) + torch.linspace(-.15, .15, 7)
        rows = torch.repeat_interleave(torch.arange(len(questions)), counts)
        pos = torch.cat([torch.arange(n) for n in counts.tolist()])
        return dict(logits=logits[rows, pos], query_inputs=x[rows, pos], batch_indices=rows, counts=counts)


@pytest.mark.parametrize("regime", data.REGIMES)
@pytest.mark.parametrize("mode", ("diagonal", "pooled_outer", "block_outer"))
def test_training_matches_independent_ab_losses_parameter_gradients_and_accounting(tmp_path, monkeypatch, regime, mode):
    matrix = torch.randn(64, 128, 3, 4, generator=torch.Generator().manual_seed(943))
    packet = dict(matrix=matrix, rows=[dict(questions=[f"q{i}"]) for i in range(64)],
                  payloads=[f"p{i}" for i in range(128)])
    model = toy_module(mode); reference = copy.deepcopy(model)
    args = SimpleNamespace(seed=91042, block_mode=mode, regime=regime, updates=1, run_dir=tmp_path, cache=tmp_path/"cache.pt")
    args.cache.write_bytes(b"immutable toy feature source")
    monkeypatch.setattr(run, "initialize", lambda *_args: model)
    backend = TinySequenceBackend(); result = run.train(backend, packet, args)
    logged = json.loads((tmp_path/"training.jsonl").read_text()); ep = logged["episode"]
    data.validate_episode(ep)
    expected_backend = TinySequenceBackend(); metrics = []
    for world in (0, 1):
        # Independently construct every question's full observation bank.
        banks = []
        for target in ep["targets"]:
            banks.append(torch.stack([matrix[entity, ep["mapping"][entity][int(world == 1 and entity == target)]]
                                      for entity in ep["bank_entities"]]))
        keys, values = reference.encode_bank(torch.stack(banks))
        expected_backend.vector_vera = reference
        expected_backend.vector_keys, expected_backend.vector_values = keys, values
        questions = [f"q{i}" for i in ep["targets"]]
        answers = [f"p{ep['mapping'][i][world]}" for i in ep["targets"]]
        tokens = run.gold_tokens(expected_backend, answers)
        branch = expected_backend.forward_sequences(questions, tokens)
        ce, _ = run.supervised_loss(branch, tokens)
        targets = torch.tensor([ep["bank_entities"].index(i) for i in ep["targets"]])[branch["batch_indices"]]
        losses = reference.group_address_loss(branch["query_inputs"], targets, keys,
            batch_indices=branch["batch_indices"], reduction="none")
        address = run.sequence_mean(losses, branch["batch_indices"], branch["counts"])
        ((ce + .2*address) / 2).backward(); metrics.append((float(ce.detach()), float(address.detach())))
    groups = [[*reference.Wq.parameters(), *reference.Wk.parameters(), reference.slot_position],
              list(reference.Wv.parameters()), [reference.b], [reference.B]]
    if mode != "diagonal": groups.append([*reference.P_in.parameters(),*reference.P_out.parameters()])
    for params in groups: torch.nn.utils.clip_grad_norm_(params, 1.)
    expected_ce = sum(x[0] for x in metrics)/2; expected_address = sum(x[1] for x in metrics)/2
    assert logged["metrics"]["ce"] == pytest.approx(expected_ce, abs=2e-6)
    assert logged["metrics"]["address"] == pytest.approx(expected_address, abs=2e-6)
    assert logged["metrics"]["loss"] == pytest.approx(expected_ce+.2*expected_address, abs=2e-6)
    assert backend.calls == result["totals"]["backbone_calls"] == 1 and expected_backend.calls == 2
    assert result["totals"]["target_exposures"] == 8
    answers = [f"p{ep['mapping'][i][w]}" for w in (0, 1) for i in ep["targets"]]
    tokens = run.gold_tokens(backend, answers)
    assert result["totals"]["gold_tokens"] == sum(map(len, tokens))
    questions = [f"q{i}" for i in ep["targets"]] * 2
    lengths = [len(backend.prompt_ids(q)) + len(t) for q,t in zip(questions,tokens)]
    assert logged["gold_token_lengths"] == list(map(len,tokens))
    assert logged["input_lengths"] == lengths
    assert result["totals"]["input_positions"] == sum(lengths)
    assert result["totals"]["padded_input_positions"] == 16 * max(lengths)
    for name, parameter in model.named_parameters():
        expected = dict(reference.named_parameters())[name]
        assert parameter.grad is not None and expected.grad is not None
        torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=3e-5, msg=name)
        assert logged["parameter_gradient_norms"][name] is not None
    saved = torch.load(tmp_path/"last.pt", weights_only=True)
    assert saved["regime"] == regime and saved["step"] == 1 and saved["protocol"] == run.PROTOCOL


def test_initialize_statistics_use_static_128_observations_not_all_cross_product(monkeypatch):
    raw = data.dataset(); train = raw["train"]
    # Broadcast identical coordinates to avoid allocating a 0.96GB test cache.
    # isfinite may inspect just the representative identical coordinate here.
    matrix = torch.arange(64*128*3).reshape(64,128,3,1).float().expand(-1,-1,-1,9728)
    q = torch.arange(64).float()[:, None].expand(-1, 9728)
    packet = dict(protocol=run.PROTOCOL, split="train", worlds=["A", "B"], views=1,
        model_revision=run.REVISION, data_seed=data.SEED, writer_prefix=data.WRITER_PREFIX,
        entities=train["entities"], payloads=train["payloads"], rows=train["rows"], rows_sha256=data.digest(train["rows"]),
        matrix=matrix, q=q, payload_token_counts=[[1,1,1] for _ in range(64*128)])
    original_finite = torch.isfinite
    monkeypatch.setattr(torch, "isfinite", lambda value: original_finite(value[..., 0])
                        if value is matrix or value is q else original_finite(value))
    class Spy:
        def __init__(self, *args, **kwargs): self.configuration = kwargs
        def to(self, device): return self
        def fit_statistics(self, queries, support): self.queries, self.support = queries, support
        def fit_value_statistics(self, support): self.values = support
    monkeypatch.setattr(run, "BlockVeRA", Spy)
    m = run.initialize(packet, SimpleNamespace(seed=91042, block_mode="block_outer"), "cpu")
    expected = torch.cat([matrix[i, 2*i:2*i+2] for i in range(64)]).reshape(384,9728)
    assert torch.equal(m.support, expected) and torch.equal(m.values, expected) and torch.equal(m.queries, q)
    assert m.configuration["train_B"] is True


def validation_packet():
    train = data.training_data(data.dataset())
    return dict(protocol=run.PROTOCOL, split="train", worlds=["A","B"], views=1,
        model_revision=run.REVISION, data_seed=data.SEED, writer_prefix=data.WRITER_PREFIX,
        entities=train["entities"], payloads=train["payloads"], rows=train["rows"], rows_sha256=data.digest(train["rows"]),
        matrix=torch.ones(64,128,3,1).expand(-1,-1,-1,9728),
        q=torch.ones(64,1).expand(-1,9728), payload_token_counts=[[1,1,1] for _ in range(8192)])


def inject_probe_with_matching_digest(packet):
    packet["rows"][0]["c"] = "apple amber badger"
    packet["rows_sha256"] = data.digest(packet["rows"])


@pytest.mark.parametrize("mutation", [
    inject_probe_with_matching_digest,
    lambda p: p.update(split="known"),
    lambda p: p.update(writer_prefix="changed writer prompt"),
    lambda p: p.update(rows_sha256="0"*64),
    lambda p: p["entities"].reverse(),
    lambda p: p.update(q=p["q"].long()),
    lambda p: p["matrix"].requires_grad_(True),
    lambda p: p["q"].__setitem__((0,0),float("nan")),
    lambda p: p["payload_token_counts"][0].__setitem__(0,True),
    lambda p: p["payload_token_counts"].pop(),
])
def test_initialize_rejects_untrusted_training_schema_before_constructing_module(monkeypatch,mutation):
    packet = validation_packet(); mutation(packet)
    original_finite = torch.isfinite
    # The test tensors repeat exactly one stored coordinate along feature width;
    # inspecting that coordinate is equivalent and avoids a huge boolean copy.
    monkeypatch.setattr(torch,"isfinite",lambda x:original_finite(x[...,0])
                        if x.ndim and x.stride(-1)==0 else original_finite(x))
    def forbidden(*_a,**_kw): raise AssertionError("Cache must fail before fitting trainable module")
    monkeypatch.setattr(run,"BlockVeRA",forbidden)
    with pytest.raises(ValueError):
        run.initialize(packet,SimpleNamespace(seed=81042,read_mode="diagonal",readout="vera"),"cpu")


def test_evaluation_axes_preserve_all_predeclared_factorial_cells():
    assert evaluation.axes("known", "development") == {"CC": ["B", "C", "SWAP"]}
    assert evaluation.axes("dev", "development") == {"CC": ["B", "C"]}
    assert evaluation.axes("known", "confirmation") == {"CC": ["D"]}
    assert evaluation.axes("confirm", "confirmation") == {p: ["B", "D"] for p in evaluation.PHASES}
    for split in ("known", "dev"):
        assert all("D" not in worlds for worlds in evaluation.axes(split,"development").values())
    with pytest.raises(ValueError): evaluation.axes("confirm", "development")
    assert evaluation.axes("train","preflight") == {"CC":["B"]}
    assert evaluation.axes("known","smoke") == {"CC":["B"]}
    for split,part in (("known","preflight"),("dev","preflight"),("confirm","preflight"),
                       ("train","smoke"),("dev","smoke"),("confirm","smoke")):
        with pytest.raises(ValueError): evaluation.axes(split,part)
