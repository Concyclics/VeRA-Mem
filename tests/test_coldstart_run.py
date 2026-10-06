"""Cold-start initialization, real donor episodes and scheduled optimizer forks."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from vera_mem import coldstart_run as run
from vera_mem import coldstart_data
from vera_mem.dictionary_vera import DictionaryVeRA


def simple_rows(size=32):
    rows = []
    for i in range(size):
        a, b = f"word{i} alpha beta", f"word{(i+1)%size} alpha beta"
        support = [f"Note E{i}. {a}", f"BEGIN HELDOUT E{i}\n{a}\nEND"]
        rows.append(dict(id=f"id-{i}", entity=f"E{i}", relation="continuation", split="train",
                         a=a, b=b, questions=[f"Query E{i}", f"HELDOUT question E{i}"],
                         supports=[support, [s.replace(a, b) for s in support]], provenance={}))
    return rows


def features(size=32):
    rng = torch.Generator().manual_seed(91)
    q = torch.randn(size, 1, 5, generator=rng).half()
    last = torch.randn(size, 2, 1, 5, generator=rng).half()
    pool = (last.float() * torch.tensor([.2, .7, 2., 3., .4]) + .8).half()
    rows = simple_rows(size)
    for r in rows:
        r["questions"] = r["questions"][:1]
        r["supports"] = [w[:1] for w in r["supports"]]
    return dict(protocol=run.PROTOCOL, model_revision=run.REVISION, split="train", task="wikipedia",
                rows=rows, q=q, last=last, pool=pool, writer_prefix="Remember this information: ")


def tiny_module():
    m = DictionaryVeRA(5, 6, rank=3, key_dim=4, top_k=2, base_top_k=2, base_size=7,
                       temperature=.6, seed=13, writer_mode="masked_mean", alpha_base=0., base_trainable=False)
    p = features()
    m.fit_statistics(p["q"].float().flatten(0, 1), p["last"].float().flatten(0, 2))
    m.fit_value_statistics(p["pool"].float().flatten(0, 2))
    with torch.no_grad():
        m.b.fill_(.3)
    return m


def test_deterministic_exhaustive_sampler_keeps_true_other_entity_b_donor():
    rows = simple_rows(128)
    one, two = run.BankSampler(rows, 43), run.BankSampler(rows, 43)
    for epoch in range(2):
        visited = []
        for _ in range(len(rows) // 8):
            sample = one.next()
            assert sample == two.next()
            visited.extend(sample["targets"])
            assert len(sample["episode"]) == len(set(sample["episode"])) == 72
            assert [sample["episode"][column] for column in sample["columns"]] == sample["targets"]
            assert sample["qviews"] == [0] * 8 and sample["sviews"] == [0] * 72
            for target in sample["targets"]:
                assert any(rows[j]["a"] == rows[target]["b"] and rows[j]["entity"] != rows[target]["entity"]
                           for j in sample["episode"])
        assert sorted(visited) == list(range(len(rows))), epoch


def test_sampler_on_actual_fresh_synthetic_data_includes_a_b_donors_and_other_relations():
    rows = coldstart_data.synthetic_records(dict(train=128, dev=16, confirm=16))[0]["train"]
    sampler = run.BankSampler(rows, 63042)
    for _ in range(16):
        sample = sampler.next()
        episode = set(sample["episode"])
        for target in sample["targets"]:
            row = rows[target]
            for world in ("a", "b"):
                assert any(rows[j]["a"] == row[world] and rows[j]["entity"] != row["entity"] for j in episode)
            assert {i for i, r in enumerate(rows) if r["entity"] == row["entity"]} <= episode


def test_single_training_view_does_not_fabricate_paraphrase_variation():
    p = features()
    sample = run.BankSampler(p["rows"], 42).next()
    batch = run.make_batch(p, sample, p["pool"], .05)
    # This stage intentionally trains canonical observations only. Its P branch
    # repeats A; callers must not claim that its zero JS establishes invariance.
    assert batch["contexts"][0] == batch["contexts"][2]
    torch.testing.assert_close(batch["key_banks"][0], batch["key_banks"][2], rtol=0, atol=0)
    torch.testing.assert_close(batch["value_banks"][0], batch["value_banks"][2], rtol=0, atol=0)
    for row, column in enumerate(sample["columns"]):
        changed = (batch["key_banks"][1][row] != batch["key_banks"][0][row]).any(-1).nonzero().flatten()
        assert changed.tolist() == [column]


def test_fresh_init_uses_only_first_8192_train_rows_and_separate_pool_statistics(tmp_path, monkeypatch):
    p = features(8194)
    called = []
    real_class = run.DictionaryVeRA
    def small_constructor(*args, **kwargs):
        called.append((args, deepcopy(kwargs)))
        kwargs.update(rank=3, key_dim=4, top_k=2, base_top_k=2, base_size=7)
        return real_class(5, 6, **kwargs)
    monkeypatch.setattr(run, "DictionaryVeRA", small_constructor)
    args = SimpleNamespace(checkpoint=None, cluster_init=False, seed=63042, run_dir=tmp_path)
    result = run.initialize(p, args)
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    state = checkpoint["module"]
    assert called[0][0] == (9728, 2560)
    assert called[0][1]["rank"] == called[0][1]["key_dim"] == 64
    assert called[0][1]["base_size"] == 1024
    assert checkpoint["step"] == 0 and "optimizer" not in checkpoint
    assert checkpoint["architecture"]["alpha_base"] == 0.
    assert checkpoint["architecture"]["base_trainable"] is False
    for field, feature, lastdim in [("query_center", "q", 1), ("support_center", "last", 2),
                                    ("value_center", "pool", 2)]:
        chosen = p[feature][:8192].float().flatten(0, lastdim)
        expected = (F.normalize(chosen, dim=-1) * 5 ** .5).mean(0)
        torch.testing.assert_close(state[field], expected, rtol=0, atol=0)
    assert not torch.equal(state["support_center"], state["value_center"])
    assert state["statistics_fitted"] and state["value_statistics_fitted"]
    assert state["b"].count_nonzero() == 0
    assert torch.equal(state["Wq.weight"], state["Wk.weight"])
    assert result["train_records"] == 8194


def test_cluster_initialization_uses_warm_encoders_and_world_a_only_without_refitting(tmp_path, monkeypatch):
    m, p = tiny_module(), features()
    source = tmp_path / "warm.pt"
    run.save_module(m, source, step=1024)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    observed = {}
    original = DictionaryVeRA.initialise_base
    def capture(self, keys, values, **options):
        observed.update(keys=keys.clone(), values=values.clone(), options=options)
        return original(self, keys, values, **options)
    monkeypatch.setattr(DictionaryVeRA, "initialise_base", capture)
    output = tmp_path / "initialized"
    output.mkdir()
    run.initialize(p, SimpleNamespace(checkpoint=source, cluster_init=True, seed=63042, run_dir=output))
    torch.testing.assert_close(observed["keys"], m.encode_key(p["last"][:, 0, 0].float()))
    torch.testing.assert_close(observed["values"], m.encode_value(p["pool"][:, 0, 0].float()))
    assert observed["options"]["split"] == "train"
    result, checkpoint = run.load_module(output / "last.pt", "cpu")
    assert checkpoint["step"] == 0 and "optimizer" not in checkpoint
    assert not torch.equal(result.base_keys, before["base_keys"])
    assert not torch.equal(result.base_values, before["base_values"])
    for key in ["A", "B", "b", "Wq.weight", "Wk.weight", "Wv.weight", "query_center", "support_center", "value_center"]:
        torch.testing.assert_close(result.state_dict()[key], before[key], rtol=0, atol=0)
    audit = torch.load(output / "cluster_initialization.pt", weights_only=True)
    assert audit["counts"].sum() == len(p["rows"])


@pytest.mark.parametrize("operation", ["initialize", "train"])
def test_confirm_features_are_rejected_before_any_model_or_optimizer_access(operation):
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("Confirmation features reached model execution")
    with pytest.raises(ValueError, match="Train-only"):
        if operation == "initialize":
            run.initialize({"split": "confirm"}, Forbidden())
        else:
            run.train(Forbidden(), Forbidden(), {"split": "confirm"}, Forbidden())


@pytest.mark.parametrize("config", [dict(train_B=True), dict(value_mlp_hidden=8)])
def test_uncontrolled_reader_or_writer_architecture_cannot_silently_escape_optimizer(config):
    m = SimpleNamespace(train_B=False, value_mlp_hidden=0)
    for key, value in config.items():
        setattr(m, key, value)
    with pytest.raises(ValueError, match="frozen B.*linear"):
        run.train(None, m, {"split": "train"}, None)


def test_training_forks_share_schedule_rebuild_float_banks_and_apply_routing_and_freeze_flags(tmp_path, monkeypatch):
    calls = []
    def fake_step(backend, m, batch, method, scale, generator):
        assert method == "base"
        assert batch["q_views"].dtype == batch["key_banks"][0].dtype == batch["value_banks"][0].dtype == torch.float32
        assert not m.A.requires_grad and not m.B.requires_grad
        assert all(p.requires_grad for p in [m.Wq.weight, m.Wk.weight, m.Wv.weight, m.b])
        assert m.base_keys.requires_grad == m.base_values.requires_grad == m.base_trainable
        assert not backend.__dict__.get("teacher_context_passed_to_student", False)
        calls.append(dict(mode=m.effective_routing_mode, columns=list(batch["columns"]),
                          base_trainable=m.base_trainable))
        query = batch["q_views"][batch["columns"], 0].unsqueeze(1)
        result = m(query, m.encode_key(batch["key_banks"][0]), m.encode_value(batch["value_banks"][0]))
        loss = result.square().mean()
        return loss, {"total_loss": float(loss.detach()), "target_tokens": 8}, []
    monkeypatch.setattr(run, "interface_step", fake_step)
    backend = SimpleNamespace(device=torch.device("cpu"))
    configurations = [("sparse", False), ("sparse", True), ("dense_warm", True), ("straight_through", True)]
    schedules, statuses = [], []
    for i, (routing, trainable) in enumerate(configurations):
        output = tmp_path / str(i)
        output.mkdir()
        m, packet = tiny_module(), features()
        original_base = (m.base_keys.detach().clone(), m.base_values.detach().clone())
        args = SimpleNamespace(alpha_base=.25, base_trainable=trainable, seed=63042, batch_size=8,
                               updates=4, routing=routing, run_dir=output)
        start = len(calls)
        statuses.append(run.train(backend, m, packet, args))
        sequence = calls[start:]
        expected = ["dense", "dense", "sparse", "sparse"] if routing == "dense_warm" else [routing] * 4
        assert [c["mode"] for c in sequence] == expected
        rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
        assert all(0 <= row["base_retained_dense_mass_min"] <= row["base_retained_dense_mass_mean"] <= 1 for row in rows)
        schedules.append([r["sample"] for r in rows])
        schedule_digest = hashlib.sha256()
        for r in rows:
            schedule_digest.update(json.dumps(r["sample"], sort_keys=True).encode())
        assert statuses[-1]["schedule_sha256"] == schedule_digest.hexdigest()
        checkpoint = torch.load(output / "last.pt", weights_only=True)
        assert checkpoint["step"] == 4
        assert {int(s["step"]) for s in checkpoint["optimizer"]["state"].values()} == {4}
        assert len(checkpoint["optimizer"]["param_groups"]) == (4 if trainable else 3)
        assert checkpoint["architecture"]["base_trainable"] == trainable
        assert checkpoint["architecture"]["alpha_base"] == float(checkpoint["module"]["dictionary_alpha"]) == .25
        assert statuses[-1]["target_unique_count"] == 32
        if trainable:
            assert statuses[-1]["key_gradient_coverage"] > 0 and statuses[-1]["value_gradient_coverage"] > 0
            assert not torch.equal(m.base_values, original_base[1])
        else:
            assert statuses[-1]["key_gradient_coverage"] == statuses[-1]["value_gradient_coverage"] == 0
            assert torch.equal(m.base_keys, original_base[0]) and torch.equal(m.base_values, original_base[1])
    assert all(s == schedules[0] for s in schedules)
    assert len({s["schedule_sha256"] for s in statuses}) == 1


def test_teacher_strategy_training_changes_only_teacher_contexts(tmp_path, monkeypatch):
    captured=[]
    def fake_step(backend, module, batch, method, scale, generator):
        captured.append(deepcopy(batch))
        # Exercise the actual train loop, backward, optimizer and checkpoint
        # path with a tiny differentiable student; no language model needed.
        query=batch["q_views"][batch["columns"],0].unsqueeze(1)
        output=module(query,module.encode_key(batch["key_banks"][0]),module.encode_value(batch["value_banks"][0]))
        loss=output.square().mean()
        return loss,dict(total_loss=float(loss.detach()),target_tokens=8),[]
    monkeypatch.setattr(run,"interface_step",fake_step)
    strategies=(None,"baseline","quote_instruction","gold_annotated")
    snapshots={}
    statuses=[]
    for strategy in strategies:
        directory=tmp_path/(strategy or "legacy_missing_argument")
        directory.mkdir()
        packet=features()
        before=deepcopy(packet)
        args=SimpleNamespace(alpha_base=.25,base_trainable=False,seed=63042,batch_size=8,
                             updates=1,routing="sparse",run_dir=directory)
        if strategy is not None:args.teacher_strategy=strategy
        status=run.train(SimpleNamespace(device=torch.device("cpu")),tiny_module(),packet,args)
        assert status["complete"] and status["updates"]==1
        statuses.append(status)
        snapshots[strategy]=captured[-1]
        # Teacher interventions must not alter the writer observations or
        # cached record/answer metadata used to construct later batches.
        assert packet["rows"]==before["rows"]
        for name in ("q","last","pool"):
            torch.testing.assert_close(packet[name],before[name].float(),rtol=0,atol=0)
        if strategy is None:
            sample=json.loads((directory/"training.jsonl").read_text().splitlines()[0])["sample"]
            expected_contexts=[[before["rows"][i]["supports"][world][0] for i in sample["targets"]]
                               for world in (0,1,0)]
            assert captured[-1]["contexts"]==expected_contexts

    def identical(actual, expected):
        if isinstance(expected,torch.Tensor):
            assert isinstance(actual,torch.Tensor) and actual.dtype==expected.dtype
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        elif isinstance(expected,dict):
            assert actual.keys()==expected.keys()
            for key in expected:identical(actual[key],expected[key])
        elif isinstance(expected,(list,tuple)):
            assert type(actual) is type(expected) and len(actual)==len(expected)
            for left,right in zip(actual,expected):identical(left,right)
        else:assert actual==expected

    baseline=snapshots[None]
    identical(snapshots["baseline"],baseline)
    for strategy in strategies[2:]:
        actual=snapshots[strategy]
        # Checking every non-context field also prevents a future intervention
        # from adding an unreviewed gold side channel to the student batch.
        assert actual.keys()==baseline.keys()
        identical({k:v for k,v in actual.items() if k!="contexts"},
                  {k:v for k,v in baseline.items() if k!="contexts"})
    quote_suffixes=[]
    for world in range(3):
        for index,source in enumerate(baseline["contexts"][world]):
            answer=baseline["answers"][world][index]
            quote=snapshots["quote_instruction"]["contexts"][world][index]
            gold=snapshots["gold_annotated"]["contexts"][world][index]
            assert quote.startswith(source+"\n\n") and gold.startswith(source+"\n\n")
            suffix=quote[len(source):]
            quote_suffixes.append(suffix)
            assert answer not in suffix and "<requested_span>" not in suffix
            assert quote.count(answer)==source.count(answer)
            assert gold.endswith("<requested_span>"+answer+"</requested_span>")
            assert gold.count(answer)==source.count(answer)+1
    assert len(set(quote_suffixes))==1
    assert len({s["schedule_sha256"] for s in statuses})==1


def test_prepare_repairs_real_donor_tokens_and_stores_only_canonical_training_half_features(tmp_path, monkeypatch):
    rows = simple_rows(4)
    answers = ["apple dawn dusk", "amber early late", "blue sky moon", "cyan stone tree"]
    for i, row in enumerate(rows):
        row["a"], row["b"] = answers[i], answers[(i + 1) % 4]
        row["supports"] = [[f"Note E{i}. {answer}", f"HELDOUT E{i}: {answer}"] for answer in (row["a"], row["b"])]
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            first = {"apple": 1, "amber": 1, "blue": 2, "cyan": 3}[text.split()[0]]
            return [first, 7, 8]
    captured = {}
    def layer_features(texts):
        captured["queries"] = texts
        return torch.arange(len(texts) * 5).view(len(texts), 5).float() + .125
    def support_features(backend, texts, batch_size):
        captured["supports"] = texts
        last = torch.arange(len(texts) * 5).view(len(texts), 5).float() + .25
        return last, last + .5
    monkeypatch.setattr(coldstart_data, "load_records", lambda *args: deepcopy(rows))
    monkeypatch.setattr(run, "support_features", support_features)
    source = tmp_path / "data"
    (source / "wikipedia").mkdir(parents=True)
    (source / "manifest.json").write_text('{}')
    (source / "wikipedia" / "train.jsonl").write_text('\n'.join(json.dumps(r) for r in rows))
    output = tmp_path / "features"
    output.mkdir()
    backend = SimpleNamespace(tokenizer=Tokenizer(), layer_features=layer_features)
    args = SimpleNamespace(data_dir=source, domain="wikipedia", splits="train", run_dir=output)
    result = run.prepare(backend, args)
    saved = torch.load(output / "train.pt", weights_only=True)
    assert result["train"]["views"] == 1
    assert not any("HELDOUT" in s for s in captured["queries"] + captured["supports"])
    assert saved["q"].shape == (4, 1, 5) and saved["last"].shape == saved["pool"].shape == (4, 2, 1, 5)
    assert all(saved[k].dtype == torch.float16 for k in ["q", "last", "pool"])
    assert saved["writer_prefix"] == "Remember this information: "
    assert saved["rows"][0]["b"] == answers[2]
    assert saved["rows"][0]["provenance"]["token_repair"]["donor_id"] == rows[2]["id"]
    for r in saved["rows"]:
        assert r["a_tokens"][0] != r["b_tokens"][0]
        assert r["supports"][1][0] == r["supports"][0][0].replace(r["a"], r["b"], 1)
        assert r["b"] in [other["a"] for other in saved["rows"] if other["entity"] != r["entity"]]
