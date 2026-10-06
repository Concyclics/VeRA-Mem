"""Causal intervention and optimizer-boundary tests with an actual tiny model."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest
import torch
from torch.nn import functional as F

from vera_mem import augmentation_data as expressions
from vera_mem import counterfactual_run as run
from test_counterfactual_backend import make_backend


def batch_inputs(replay=False):
    generator = torch.Generator().manual_seed(61)
    queries = torch.randn(4, 8, 5, generator=generator)
    support = torch.randn(4, 2, 4, 5, generator=generator)
    columns = [2, 0]
    selected_views = [0, 2, 1, 3]
    base = support[torch.arange(4), 0, selected_views]
    alternate = support[columns, 1, [selected_views[c] for c in columns]]
    paraphrase = support[columns, 0, [(selected_views[c] + 1) % 4 for c in columns]]
    all_alternative = support[:, 0].clone()
    all_alternative[columns] = support[columns, 1]
    return dict(
        questions=["q one", "a longer q two"],
        contexts=[["value a for q1", "value bb for q2"],
                  ["value dddd for q1", "value e for q2"],
                  ["q1 has value a", "q2 has value bb"]],
        # Deliberately different continuation lengths across worlds. First
        # prediction for the second row has a different flattened index in B.
        answers=[["a", "bb"], ["dddd", "e"], ["a", "bb"]],
        banks=run.independent_banks(base, alternate, paraphrase, columns),
        columns=columns, q_views=queries, s_views=[support[:, 0], all_alternative],
        replay_support=support[:, 0, 0],
        replay_questions=["canonical1", "canonical2"] if replay else None,
        target_ids=["fact2", "fact0"], episode_ids=[f"fact{i}" for i in range(4)],
    )


def track_forward(backend, monkeypatch):
    events = []
    original = backend.forward_sequences
    def tracked(questions, continuations, contexts=None, teacher=False):
        event = dict(teacher=teacher, questions=list(questions), contexts=contexts,
                     tokens=[list(row) for row in continuations],
                     keys=backend.vector_keys, values=backend.vector_values,
                     bank_has_grad=backend.vector_values.requires_grad)
        output = original(questions, continuations, contexts=contexts, teacher=teacher)
        event["output"] = output
        events.append(event)
        return output
    monkeypatch.setattr(backend, "forward_sequences", tracked)
    return events


def test_independent_banks_replace_exactly_each_rows_target_and_preserve_input_graphs():
    base = torch.arange(20.).reshape(5, 4).requires_grad_()
    alternative = torch.tensor([[101., 102, 103, 104], [201., 202, 203, 204]], requires_grad=True)
    paraphrase = torch.tensor([[301., 302, 303, 304], [401., 402, 403, 404]], requires_grad=True)
    columns = [3, 1]
    original = base.detach().clone()
    a, b, p = run.independent_banks(base, alternative, paraphrase, columns)
    assert a.shape == b.shape == p.shape == (2, 5, 4)
    for row, column in enumerate(columns):
        assert (b[row] != a[row]).any(-1).nonzero().flatten().tolist() == [column]
        assert (p[row] != a[row]).any(-1).nonzero().flatten().tolist() == [column]
        torch.testing.assert_close(b[row, column], alternative[row])
        torch.testing.assert_close(p[row, column], paraphrase[row])
    torch.testing.assert_close(base, original)
    # A loss on intervention row 0 cannot update row 1's alternative observation.
    b[0].sum().backward()
    assert alternative.grad[0].eq(1).all() and alternative.grad[1].eq(0).all()
    assert base.grad[columns[0]].eq(0).all()
    assert base.grad[columns[1]].eq(1).all()
    assert paraphrase.grad is None


def test_make_batch_preserves_same_question_identity_and_rewrites_only_target_observations():
    originals = expressions.datasets(train_size=16)["train"][:4]
    alternatives = [replace(ex, answer=("zinc" if ex.answer != "zinc" else "amber")) for ex in originals]
    generator = torch.Generator().manual_seed(315)
    packet = dict(pairs=[dict(original=asdict(a), alternative=asdict(b)) for a, b in zip(originals, alternatives)],
                  q=torch.randn(4, 8, 5, generator=generator),
                  s=torch.randn(4, 2, 4, 5, generator=generator),
                  template_ids=dict(q=[t.id for t in expressions.get_templates("train", "query")],
                                    s=[t.id for t in expressions.get_templates("train", "support")]))
    sample = dict(episode=[3, 1, 0, 2], targets=[0, 3], mapping=[2, 0],
                  qviews=[7, 2, 4, 1], sviews=[3, 2, 0, 1])
    batch = run.make_batch(packet, sample, step=4, device="cpu")
    assert batch["questions"] == [expressions.render_question(originals[0], "train_query_04"),
                                  expressions.render_question(originals[3], "train_query_07")]
    assert batch["answers"] == [[originals[0].answer, originals[3].answer],
                                [alternatives[0].answer, alternatives[3].answer],
                                [originals[0].answer, originals[3].answer]]
    for row, (identity, column) in enumerate(zip(sample["targets"], sample["mapping"])):
        support_view = sample["sviews"][column]
        rewrite_view = (support_view + 1) % 4
        for world, example, view in ((0, originals[identity], support_view),
                                     (1, alternatives[identity], support_view),
                                     (2, originals[identity], rewrite_view)):
            expected = expressions.render_support(example, packet["template_ids"]["s"][view])
            assert batch["contexts"][world][row] == expected
        torch.testing.assert_close(batch["banks"][1][row, column], packet["s"][identity, 1, support_view])
        torch.testing.assert_close(batch["banks"][2][row, column], packet["s"][identity, 0, rewrite_view])
        changed = (batch["banks"][1][row] != batch["banks"][0][row]).any(-1)
        assert changed.nonzero().flatten().tolist() == [column]
    assert batch["replay_questions"] is not None
    assert run.make_batch(packet, sample, step=3, device="cpu")["replay_questions"] is None


@pytest.mark.parametrize("method", run.METHODS)
def test_paired_step_teacher_context_is_private_and_each_world_bank_is_differentiable(method, monkeypatch):
    backend = make_backend()
    module = backend.vector_vera
    batch = batch_inputs(replay=True)
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    events = track_forward(backend, monkeypatch)
    loss, metrics, trajectories = run.paired_step(
        backend, module, batch, method, False, torch.Generator().manual_seed(8))
    assert [event["teacher"] for event in events] == [True, True, True, False, False, False, False]
    for world in range(3):
        teacher, student = events[world], events[3 + world]
        assert teacher["questions"] == student["questions"] == batch["questions"]
        assert teacher["contexts"] == batch["contexts"][world]
        assert student["contexts"] is None
        assert teacher["tokens"] == student["tokens"] == run.gold_tokens(backend, batch["answers"][world])
        assert not teacher["output"]["logits"].requires_grad
        assert not teacher["output"]["hidden"].requires_grad
        assert student["bank_has_grad"]
        assert student["keys"].ndim == 3 and student["keys"].shape[:2] == (2, 4)
        torch.testing.assert_close(student["keys"], module.encode_key(batch["banks"][world]))
        torch.testing.assert_close(student["values"], module.encode_value(batch["banks"][world]))
    assert len({id(event["values"]) for event in events[3:6]}) == 3
    assert events[-1]["contexts"] is None and events[-1]["values"].ndim == 2
    expected_total = (metrics["main_loss"] + .2 * metrics["actual_address_loss"]
                      + .2 * metrics["style_address_loss"] + .05 * metrics["paraphrase_loss"]
                      + .25 * metrics["replay_loss"])
    if method in ("behavior", "hidden", "mixed"):
        expected_total += .1 * metrics["behavior_loss"]
    if method in ("hidden", "mixed"):
        expected_total += .1 * metrics["hidden_loss"]
    assert metrics["total_loss"] == pytest.approx(expected_total, rel=1e-5)
    assert metrics["main_loss"] == pytest.approx(metrics["forward_kl"] + .5 * metrics["first_token_ce"])
    assert all(trajectory["sampled_world"] is None for trajectory in trajectories)
    # Building or backpropagating the three-world objective never steps weights.
    for name, value in module.named_parameters():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    loss.backward()
    for name, value in module.named_parameters():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0, name
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_hidden_and_behavior_receive_only_first_prediction_positions_despite_unequal_gold_lengths(monkeypatch):
    backend = make_backend()
    events = track_forward(backend, monkeypatch)
    captured = {}
    original_hidden, original_behavior = run.hidden_delta_loss, run.pair_behavior_loss
    def hidden(sa, sb, ta, tb, **kwargs):
        captured["hidden"] = [value.detach().clone() for value in (sa, sb, ta, tb)]
        return original_hidden(sa, sb, ta, tb, **kwargs)
    def behavior(sa, sb, ta, tb, ids_a, ids_b, **kwargs):
        captured["logits"] = [value.detach().clone() for value in (sa, sb, ta, tb)]
        captured["ids"] = [ids_a.tolist(), ids_b.tolist()]
        return original_behavior(sa, sb, ta, tb, ids_a, ids_b, **kwargs)
    monkeypatch.setattr(run, "hidden_delta_loss", hidden)
    monkeypatch.setattr(run, "pair_behavior_loss", behavior)
    batch = batch_inputs()
    run.paired_step(backend, backend.vector_vera, batch, "hidden", False, None)
    assert len(events[0]["tokens"][0]) != len(events[1]["tokens"][0])
    for component in ("hidden", "logits"):
        for selected, event in zip(captured[component], [events[3], events[4], events[0], events[1]]):
            # First row starts at zero; second starts after *this branch's*
            # first trajectory, not at an index borrowed from another world.
            expected = torch.stack([event["output"][component][0],
                                    event["output"][component][len(event["tokens"][0])]])
            torch.testing.assert_close(selected, expected)
    assert captured["ids"] == [[row[0] for row in run.gold_tokens(backend, batch["answers"][world])]
                               for world in (0, 1)]
    assert all(not events[world]["output"]["hidden"].requires_grad for world in range(3))


def test_future_gold_suffix_changes_cannot_change_first_position_delta_or_paraphrase_losses():
    backend = make_backend()
    batch = batch_inputs()
    _, left, _ = run.paired_step(backend, backend.vector_vera, batch, "hidden", False, None)
    changed = dict(batch, answers=[["aaaaaa", "bc"], ["da", "eeeee"], ["aaaaaa", "bc"]])
    _, right, _ = run.paired_step(backend, backend.vector_vera, changed, "hidden", False, None)
    for name in ("behavior_loss", "hidden_loss", "paraphrase_loss", "first_token_ce"):
        assert left[name] == pytest.approx(right[name], abs=1e-6), name


def test_on_policy_uses_one_sampled_prefix_in_all_worlds_rebuilds_banks_and_keeps_gold_first_ce(monkeypatch):
    backend = make_backend()
    module = backend.vector_vera
    batch = batch_inputs()
    events = track_forward(backend, monkeypatch)
    sampled = [[7, 8, 10, 12], [9, 1]]
    seed = 43
    chosen_worlds = torch.randint(2, (2,), generator=torch.Generator().manual_seed(seed)).tolist()
    calls = []
    def sample(questions, max_new_tokens, temperature, generator):
        calls.append(1)
        assert questions == batch["questions"] and max_new_tokens == 4 and temperature == 1.
        assert not backend.vector_keys.requires_grad and not backend.vector_values.requires_grad
        expected = torch.stack([batch["banks"][world][row] for row, world in enumerate(chosen_worlds)])
        torch.testing.assert_close(backend.vector_keys, module.encode_key(expected))
        torch.testing.assert_close(backend.vector_values, module.encode_value(expected))
        backend.last_sampling_cost = dict(forward_calls=6, unpadded_input_tokens=73, padded_input_tokens=73)
        return sampled
    monkeypatch.setattr(backend, "sample_student", sample)
    loss, metrics, trajectories = run.paired_step(
        backend, module, batch, "mixed", True, torch.Generator().manual_seed(seed))
    assert len(calls) == 1
    assert all(event["tokens"] == sampled for event in events)
    assert all(not event["bank_has_grad"] for event in events[:3])
    assert all(event["bank_has_grad"] for event in events[3:])
    assert metrics["sampled_tokens"] == 6 and metrics["rollout_forward_calls"] == 6
    assert metrics["rollout_input_tokens"] == 73
    expected_ce, expected_fkl = [], []
    for world, event in enumerate(events[3:]):
        branch, teacher = event["output"], events[world]["output"]
        logits = branch["logits"][[0, len(sampled[0])]]
        labels = torch.tensor([row[0] for row in run.gold_tokens(backend, batch["answers"][world])])
        expected_ce.append(F.cross_entropy(logits, labels))
        token_kl = F.kl_div(branch["logits"].log_softmax(-1), teacher["logits"].softmax(-1), reduction="none").sum(-1)
        expected_fkl.append((token_kl[:4].mean() + token_kl[4:].mean()) / 2)
    assert metrics["first_token_ce"] == pytest.approx(float(torch.stack(expected_ce).mean().detach()))
    assert metrics["forward_kl"] == pytest.approx(float(torch.stack(expected_fkl).mean().detach()), abs=1e-6)
    assert [row["sampled_world"] for row in trajectories] == chosen_worlds
    assert trajectories[0]["continuation_token_ids"] == [sampled[0]] * 3
    assert trajectories[0]["continuation_token_ids"][0][-1] != backend.tokenizer.eos_token_id
    loss.backward()
    assert module.Wv.weight.grad.abs().sum() > 0 and module.Wk.weight.grad.abs().sum() > 0


def training_packet(size=4):
    pairs = [dict(original=dict(answer="a" if row % 2 == 0 else "b"),
                  alternative=dict(answer="b" if row % 2 == 0 else "a")) for row in range(size)]
    return dict(protocol=run.PROTOCOL, split="train", model_revision=run.REVISION,
                pairs=pairs, q=torch.zeros(size, 8, 9728), s=torch.zeros(size, 2, 4, 9728))


def test_training_outer_optimizer_steps_once_per_joint_update_and_mixed_cadence_after_resume(tmp_path, monkeypatch):
    backend = make_backend()
    module = backend.vector_vera
    packet = training_packet()
    optimizers = []
    real_adam = torch.optim.Adam
    class RecordingAdam(real_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0
            optimizers.append(self)
        def step(self, *args, **kwargs):
            self.step_calls += 1
            return super().step(*args, **kwargs)
    monkeypatch.setattr(run.torch.optim, "Adam", RecordingAdam)
    samples = []
    def make_batch(packet, sample, step, device):
        samples.append(deepcopy(sample))
        return batch_inputs(replay=step % 4 == 0)
    monkeypatch.setattr(run, "make_batch", make_batch)
    def sample(questions, max_new_tokens, temperature, generator):
        assert not backend.vector_keys.requires_grad and not backend.vector_values.requires_grad
        backend.last_sampling_cost = dict(forward_calls=6, unpadded_input_tokens=73, padded_input_tokens=73)
        return [[7, 8, 10, 12], [9, 1]]
    monkeypatch.setattr(backend, "sample_student", sample)
    cfg = dict(method="base", start_step=0, updates=2, batch_size=2, seed=42)
    warm = tmp_path / "warm"
    warm.mkdir()
    warm_status = run.train(backend, module, packet, {}, cfg, warm)
    checkpoint = torch.load(warm / "last.pt", weights_only=True)
    assert warm_status["complete"] and checkpoint["step"] == 2
    continued = tmp_path / "continued"
    continued.mkdir()
    status = run.train(backend, module, packet, checkpoint,
                       {**cfg, "method": "mixed", "start_step": 2, "updates": 4}, continued)
    assert len(optimizers) == 2
    assert [optimizer.step_calls for optimizer in optimizers] == [2, 4]
    # All four continuation steps load the same warm optimizer history and take
    # one shared update, not one update for each of the A/B/P branches.
    assert all(int(state["step"]) == 6 for state in optimizers[-1].state.values())
    logs = [json.loads(line) for line in (continued / "training.jsonl").read_text().splitlines()]
    assert [row["on_policy"] for row in logs] == [False, False, False, True]
    assert [row["step"] for row in logs] == [3, 4, 5, 6]
    assert [row["replay_target_tokens"] > 0 for row in logs] == [False, True, False, False]
    expected_sampler = run.CounterfactualSampler(packet["pairs"], 2, 42)
    assert samples == [expected_sampler.next() for _ in range(6)]
    assert status["complete"] and status["step"] == 6 and status["updates"] == 4
    assert backend.mode == "none" and backend.vector_keys is None and backend.vector_values is None
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_counterfactual_sampler_keeps_same_answer_wrong_entity_negatives_and_reproducible_resume():
    vocabulary = [f"answer{index}" for index in range(16)]
    pairs = [dict(original=dict(answer=vocabulary[row % 16]),
                  alternative=dict(answer=vocabulary[(row + 1) % 16])) for row in range(128)]
    left, right = [run.CounterfactualSampler(pairs, 8, 42) for _ in range(2)]
    for _ in range(12):
        sample = left.next()
        assert sample == right.next()
        assert len(sample["episode"]) == len(set(sample["episode"])) == 72
        assert [sample["episode"][column] for column in sample["mapping"]] == sample["targets"]
        for target in sample["targets"]:
            for world in ("original", "alternative"):
                answer = pairs[target][world]["answer"]
                negative_ids = [index for index in sample["episode"]
                                if index != target and pairs[index]["original"]["answer"] == answer]
                assert len(negative_ids) >= 2


@pytest.mark.parametrize("change", [
    {"protocol": "other"}, {"split": "confirmation"}, {"split": "dev"}, {"model_revision": "other"},
    {"q": torch.zeros(4, 3, 9728)}, {"q": torch.zeros(4, 8, 5)},
    {"s": torch.zeros(4, 2, 3, 9728)}, {"s": torch.zeros(3, 2, 4, 9728)},
])
def test_training_rejects_wrong_split_provenance_or_feature_shapes_before_model_work(change, tmp_path):
    packet = {**training_packet(), **change}
    with pytest.raises(ValueError, match="train-only|feature shapes"):
        run.train(None, None, packet, {}, {}, tmp_path)


@pytest.mark.parametrize("checkpoint", [
    {}, {"protocol": "wrong", "step": 2}, {"protocol": run.PROTOCOL, "step": 1},
])
def test_continuation_rejects_wrong_warm_checkpoint(checkpoint, tmp_path):
    backend = make_backend()
    cfg = dict(method="hidden", start_step=2, updates=1, batch_size=2, seed=42)
    with pytest.raises(ValueError, match="fixed common warm"):
        run.train(backend, backend.vector_vera, training_packet(), checkpoint, cfg, tmp_path)


def test_paired_step_rejects_first_token_collision_before_forward_or_sampling(monkeypatch):
    backend = make_backend()
    batch = batch_inputs()
    batch["answers"][1][0] = "another"  # Same first token as original "a".
    def forbidden(*args, **kwargs):
        raise AssertionError("Model or rollout ran before first-token pair validation")
    monkeypatch.setattr(backend, "forward_sequences", forbidden)
    monkeypatch.setattr(backend, "sample_student", forbidden)
    with pytest.raises(ValueError, match="distinct answer tokens"):
        run.paired_step(backend, backend.vector_vera, batch, "mixed", True, None)
