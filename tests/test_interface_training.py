"""Actual causal CPU forwards test the new objective and bank contracts."""
from copy import deepcopy
import json

import pytest
import torch
from torch.nn import functional as F

from vera_mem import interface_training as run
from vera_mem.context_distillation_run import gold_tokens, supervised_loss
from test_counterfactual_backend import make_backend
from test_counterfactual_run import batch_inputs


def batch():
    old = batch_inputs()
    return dict(questions=old["questions"], contexts=old["contexts"], answers=old["answers"],
                key_banks=old["banks"],
                # A different feature domain exposes accidentally reusing keys.
                value_banks=tuple(bank * .4 + .6 for bank in old["banks"]),
                columns=old["columns"], q_views=old["q_views"], key_views=old["s_views"],
                key_consistency_weight=.05)


def track(backend, monkeypatch):
    events = []
    original = backend.forward_sequences
    def forward(questions, continuations, contexts=None, teacher=False):
        event = dict(teacher=teacher, questions=list(questions), contexts=contexts,
                     tokens=deepcopy(continuations), keys=backend.vector_keys, values=backend.vector_values,
                     grad_enabled=torch.is_grad_enabled())
        event["output"] = original(questions, continuations, contexts=contexts, teacher=teacher)
        events.append(event)
        return event["output"]
    monkeypatch.setattr(backend, "forward_sequences", forward)
    return events


@pytest.mark.parametrize("method", run.METHODS)
def test_gold_branches_keep_context_private_banks_separate_and_backbone_frozen(method, monkeypatch):
    backend = make_backend()
    module, data = backend.vector_vera, batch()
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    events = track(backend, monkeypatch)
    loss, metrics, trajectories = run.interface_step(backend, module, data, method, 6., None)
    assert len(events) == 6
    assert all(event["teacher"] and not event["grad_enabled"] for event in events[:3])
    for world, event in enumerate(events[:3]):
        assert event["contexts"] == data["contexts"][world]
        assert not event["output"]["logits"].requires_grad
        assert not event["output"]["hidden"].requires_grad
    for world, event in enumerate(events[3:]):
        assert not event["teacher"] and event["contexts"] is None
        assert event["tokens"] == gold_tokens(backend, data["answers"][world])
        assert event["tokens"] == events[world]["tokens"]
        torch.testing.assert_close(event["keys"], module.encode_key(data["key_banks"][world]))
        torch.testing.assert_close(event["values"], module.encode_value(data["value_banks"][world]))
        assert event["keys"].requires_grad and event["values"].requires_grad
        assert event["output"]["logits"].requires_grad
    expected_ce = torch.stack([supervised_loss(event["output"], event["tokens"])[0] for event in events[3:]]).mean()
    assert metrics["full_sequence_ce"] == pytest.approx(float(expected_ce.detach()))
    assert metrics["replay_target_tokens"] == 0 and metrics["sampled_tokens"] == 0
    assert trajectories[0]["sampled_world"] is None
    json.dumps(metrics, allow_nan=False)
    # The step constructs one graph; no optimizer has changed any parameter.
    for name, value in module.named_parameters():
        torch.testing.assert_close(before[name], value.detach(), atol=0, rtol=0)
    loss.backward()
    for value in (module.Wq.weight, module.Wk.weight, module.Wv.weight, module.b):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert all(value.grad is None for value in backend.model.parameters())


def test_gold_suffix_changes_do_not_leak_into_first_prediction_objectives():
    backend = make_backend()
    data = batch()
    _, left, _ = run.interface_step(backend, backend.vector_vera, data, "hidden", 6., None)
    changed = dict(data, answers=[["aaaaaa", "bc"], ["da", "eeeee"], ["aaaaaa", "bc"]])
    _, right, _ = run.interface_step(backend, backend.vector_vera, changed, "hidden", 6., None)
    for field in ("clipped_behavior_loss", "normalized_behavior_loss", "hidden_loss", "paraphrase_loss", "first_token_ce"):
        assert left[field] == pytest.approx(right[field], abs=1e-6), field
    assert abs(left["full_sequence_ce"] - right["full_sequence_ce"]) > 1e-4


def test_on_policy_changes_only_fkl_prefix_and_reports_additional_cost(monkeypatch):
    backend = make_backend()
    module, data = backend.vector_vera, batch()
    gold_loss, gold_metrics, _ = run.interface_step(backend, module, data, "normalized", 6., None)
    events = track(backend, monkeypatch)
    sampled = [[7, 8, 10, 12], [9, 1]]
    seed = 43
    choices = torch.randint(2, (2,), generator=torch.Generator().manual_seed(seed)).tolist()
    sampling_calls = []
    def sample(questions, max_new_tokens, temperature, generator):
        sampling_calls.append(1)
        assert questions == data["questions"] and max_new_tokens == 8 and temperature == 1.
        assert not backend.vector_keys.requires_grad and not backend.vector_values.requires_grad
        assert backend.vector_override is None and backend.oracle_values is None and backend.vector_store is None
        chosen_keys = torch.stack([data["key_banks"][world][row] for row, world in enumerate(choices)])
        chosen_values = torch.stack([data["value_banks"][world][row] for row, world in enumerate(choices)])
        torch.testing.assert_close(backend.vector_keys, module.encode_key(chosen_keys))
        torch.testing.assert_close(backend.vector_values, module.encode_value(chosen_values))
        backend.last_sampling_cost = dict(forward_calls=6, unpadded_input_tokens=73, padded_input_tokens=73)
        return deepcopy(sampled)
    monkeypatch.setattr(backend, "sample_student", sample)
    loss, metrics, trajectories = run.interface_step(
        backend, module, data, "on_policy", 6., torch.Generator().manual_seed(seed), on_policy=True)
    assert sampling_calls == [1] and len(events) == 12
    assert [event["teacher"] for event in events] == [True] * 3 + [False] * 3 + [True] * 3 + [False] * 3
    for world in range(3):
        assert events[world]["tokens"] == events[world + 3]["tokens"] == gold_tokens(backend, data["answers"][world])
        assert events[world + 6]["tokens"] == events[world + 9]["tokens"] == sampled
    for field in ("gold_forward_kl", "full_sequence_ce", "first_token_ce", "actual_address_loss",
                  "style_address_loss", "key_consistency_loss", "behavior_loss", "hidden_loss", "paraphrase_loss"):
        assert metrics[field] == pytest.approx(gold_metrics[field], abs=1e-6), field
    # Only FKL changed; full gold CE and all common auxiliaries remain identical.
    torch.testing.assert_close(loss - gold_loss, loss.new_tensor(metrics["forward_kl"] - gold_metrics["forward_kl"]), atol=2e-6, rtol=1e-5)
    assert metrics["sampled_tokens"] == 6 and metrics["rollout_forward_calls"] == 6
    assert metrics["rollout_input_tokens"] == 73 and metrics["fkl_prefix_source"] == "student_world_mixture"
    assert metrics["target_tokens"] == metrics["gold_target_tokens"] + 3 * 6
    for teacher, field in ((True, "teacher_input_tokens"), (False, "student_input_tokens")):
        expected = sum(sum(len(backend.prompt_ids(q, None if e["contexts"] is None else e["contexts"][i])) + len(t)
                           for i, (q, t) in enumerate(zip(e["questions"], e["tokens"])))
                       for e in events if e["teacher"] == teacher)
        assert metrics[field] == expected
    assert [t["sampled_world"] for t in trajectories] == choices
    assert trajectories[0]["continuation_token_ids"] == [sampled[0]] * 3
    assert trajectories[0]["continuation_token_ids"][0][-1] != backend.tokenizer.eos_token_id
    loss.backward()
    assert module.Wv.weight.grad.abs().sum() > 0 and module.Wk.weight.grad.abs().sum() > 0
    assert all(value.grad is None for value in backend.model.parameters())


def test_non_sampling_on_policy_step_is_identical_to_normalized_arm():
    backend = make_backend()
    left, lm, _ = run.interface_step(backend, backend.vector_vera, batch(), "normalized", 6., None)
    right, rm, _ = run.interface_step(backend, backend.vector_vera, batch(), "on_policy", 6., None)
    torch.testing.assert_close(left, right, atol=0., rtol=0.)
    assert lm == rm


def test_single_factor_loss_increments_and_key_switch():
    backend = make_backend()
    module, data = backend.vector_vera, batch()
    results = {m: run.interface_step(backend, module, data, m, 6., None) for m in ("base", "clip", "normalized", "hidden")}
    base = results["base"][0]
    for method in ("clip", "normalized"):
        loss, metrics, _ = results[method]
        torch.testing.assert_close(loss, base + .1 * loss.new_tensor(metrics["behavior_loss"]))
    loss, metrics, _ = results["hidden"]
    torch.testing.assert_close(loss, results["normalized"][0] + .1 * loss.new_tensor(metrics["hidden_loss"]))
    no_key, no_metrics, _ = run.interface_step(backend, module, {**data, "key_consistency_weight": 0.}, "base", 6., None)
    torch.testing.assert_close(base, no_key + .05 * no_key.new_tensor(no_metrics["key_consistency_loss"]))
    expected_key = (1 - F.cosine_similarity(module.encode_key(data["key_views"][0])[data["columns"]],
                                           module.encode_key(data["key_views"][1])[data["columns"]], dim=-1)).mean()
    assert results["base"][1]["key_consistency_loss"] == pytest.approx(float(expected_key.detach()))


@pytest.mark.parametrize("error", ["method", "on_policy", "first_token", "paraphrase", "columns", "weight"])
def test_invalid_causal_contract_is_rejected_before_any_forward(error, monkeypatch):
    backend = make_backend()
    data, method, on_policy = batch(), "base", False
    if error == "method": method = "typo"
    if error == "on_policy": on_policy = True
    if error == "first_token": data["answers"][1][0] = data["answers"][0][0] + "changed suffix"
    if error == "paraphrase": data["answers"][2][0] = "different"
    if error == "columns": data["columns"] = [0, 0]
    if error == "weight": data["key_consistency_weight"] = float("nan")
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid protocol should fail before model forward")
    monkeypatch.setattr(backend, "forward_sequences", forbidden)
    with pytest.raises(ValueError):
        run.interface_step(backend, backend.vector_vera, data, method, 6., None, on_policy=on_policy)
