"""Causal CPU bank, format assignment, full-answer and frozen-state checks."""
import copy
from collections import Counter, defaultdict
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vera_mem.interface_eval import evaluate_interface, view_assignments
from vera_mem.vector_store import PersistentVectorDB


ANSWERS = ("apple amber birch", "river cedar flint", "cloud jade moss", "tiger maple quartz")


class TinyMemory(nn.Module):
    rank, key_dim, top_k, temperature = 1, 2, 1, .05

    def __init__(self, writer_mode="last_token"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.fixed = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.writer_mode = writer_mode

    def encode_key(self, features):
        return features[..., :2] * self.scale

    def encode_value(self, features):
        return features[..., 2:] * self.scale


class FakeBackend:
    device = torch.device("cpu")

    def __init__(self, module, *, wrapped_teacher=False, budget_hit=False, mutate=None):
        self.model = nn.Linear(1, 1).requires_grad_(False)
        self.vector_vera = module
        self.mode = "original"
        for name in ("vector_store", "vector_override", "oracle_values", "vector_keys", "vector_values"):
            setattr(self, name, object())
        self.trace_retrieval = False
        self.wrapped_teacher, self.budget_hit, self.mutate = wrapped_teacher, budget_hit, mutate
        self.generations = []

    def generate(self, question, max_new_tokens, **kwargs):
        assert max_new_tokens == 32
        assert not torch.is_grad_enabled() and not self.model.training
        assert all(not p.requires_grad for p in self.vector_vera.parameters())
        assert self.oracle_values is None and self.vector_keys is None and self.vector_values is None
        assert self.retrieval_trace == []
        index = int(question[1])
        qindex = 1 - index if question.endswith("h") else index
        query = torch.eye(2)[qindex].view(1, 1, 2)
        if self.mode == "none":
            assert set(kwargs) == {"context"}
            assert self.vector_override is None and not self.trace_retrieval
            prediction = kwargs["context"].split("=")[-1]
            if self.wrapped_teacher and index == 0:
                prediction = "The answer is: " + prediction
        else:
            assert self.mode == "vector_vera" and not kwargs
            assert isinstance(self.vector_store, PersistentVectorDB)
            assert self.vector_store.keys.device.type == "cpu"
            info = self.vector_store.search(query)
            self.prefill_retrieval = info
            if self.vector_override is None:
                assert self.trace_retrieval
                first = info["indices"][:, -1]
                self.retrieval_trace.extend([
                    {"phase": "prefill", "indices": first},
                    {"phase": "decode", "indices": (first + 1) % 2},
                    {"phase": "decode", "indices": first},
                ])
                value = info["mixed_value"].reshape(-1)[0]
            else:
                assert not self.trace_retrieval
                assert torch.count_nonzero(self.vector_override) == 0
                value = self.vector_override.reshape(-1)[0]
            prediction = ANSWERS[int(value.round())]
        self.generations.append(dict(question=question, context=kwargs.get("context"),
            mode=self.mode, prediction=prediction, bank_hash=self.vector_store.hash(),
            timestamps=self.vector_store.timestamps, keys=self.vector_store.keys,
            values=self.vector_store.values))
        return prediction, 32 if self.budget_hit else 4, .01

    def score(self, question, answer, **kwargs):
        assert not self.trace_retrieval and not torch.is_grad_enabled()
        assert (self.mode == "none") == ("context" in kwargs)
        if self.mutate == "bank":
            self.vector_store._values[0] += 1
            self.mutate = None
        elif self.mutate == "weights":
            self.vector_vera.scale.add_(1)
            self.mutate = None
        return SimpleNamespace(nll_sum=6., tokens=len(answer.split()))


def packet():
    rows = []
    features = torch.empty(2, 2, 2, 3)
    for i in range(2):
        answers = [ANSWERS[2*i], ANSWERS[2*i+1]]
        rows.append(dict(id=f"record-{i}", entity=f"entity-{i}", relation="arrival phrase",
            a=answers[0], b=answers[1], a_tokens=[1, 2, 3], b_tokens=[4, 5, 6],
            questions=[f"q{i}", f"q{i}h"],
            supports=[[f"q{i}={answer}", f"heldout:q{i}={answer}"] for answer in answers]))
        for world in range(2):
            for style in range(2):
                features[i, world, style, :2] = torch.eye(2)[i if style == 0 else 1-i]
                features[i, world, style, 2] = 2*i + world
    pool = features.clone()
    pool[..., :2] = 99  # Keys must NEVER be derived from pooled writer features.
    pool[..., 2] = (pool[..., 2] + 2) % 4
    return dict(split="confirm", rows=rows, last=features, pool=pool)


def run(tmp_path, *, teacher=True, writer_mode="last_token", max_cases=None, **kwargs):
    module = TinyMemory(writer_mode)
    backend = FakeBackend(module, **kwargs)
    result = evaluate_interface(backend, module, packet(), tmp_path,
        include_teacher=teacher, max_new_tokens=32, max_cases=max_cases)
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    return result, rows, backend, module


def load_bank(snapshot, path):
    torch.save(snapshot, path)
    return PersistentVectorDB.load(path)


def test_real_cpu_banks_four_phase_axes_teacher_controls_and_state_restore(tmp_path):
    result, rows, backend, module = run(tmp_path)
    assert result["complete"] and result["shared_weights_before"] == result["shared_weights_after"]
    assert result["state_before"] == result["state_after"]
    assert result["state_after"]["requires_grad"] == {"scale": True, "fixed": False}
    assert module.training and module.scale.requires_grad and backend.model.training
    assert backend.mode == "original" and not isinstance(backend.vector_store, PersistentVectorDB)
    assert not backend.trace_retrieval
    phases = result["phases"]
    for phase in ("CC", "CH", "HC", "HH"):
        assert phases[phase]["methods"]["real"]["paired_switch_em"] == (1 if phase in {"CC", "HH"} else 0)
        assert phases[phase]["methods"]["teacher"]["paired_switch_em"] == 1
        assert phases[phase]["teacher_acceptance"]["strict_pair_qualified"] == 2
        assert ("shuffled" in phases[phase]["methods"]) == (phase in {"CC", "HC"})
        assert ("canonical_key" in phases[phase]["methods"]) == (phase == "HC")
    assert phases["HC"]["methods"]["canonical_key"]["paired_switch_em"] == 1
    assert phases["CC"]["methods"]["real"]["a_decode_correct_residency"] == .5
    assert phases["CC"]["methods"]["real"]["a_answer_token_nll"] == 2
    assert result["generation_calls"] == len(rows) == len(backend.generations) == 48
    assert result["generation_tokens"] == 4 * len(rows)
    assert all(row["teacher_context"] is None for row in rows if row["method"] != "teacher")
    assert all(row["bank_hash_before"] == row["bank_hash_after"] for row in rows)
    assert all(not row["budget_hit"] for row in rows)
    assert sum(row["method"] == "empty" for row in rows) == 4


def test_independent_target_updates_all_bank_hashes_and_shuffled_values_reconstruct(tmp_path):
    result, rows, backend, _ = run(tmp_path, teacher=False)
    for phase in result["phases"]:
        saved = torch.load(tmp_path / f"banks/{phase}.pt", weights_only=True)
        bases = {kind: load_bank(snapshot, tmp_path / f"replay_{phase}_{kind}.pt")
                 for kind, snapshot in saved["bases"].items()}
        bank_hashes = {}
        for patch in saved["interventions"]:
            index = patch["index"]
            for kind, replacement in patch["banks"].items():
                base = bases[kind]
                assert base.hash() == saved["base_hashes"][kind] == replacement["parent_hash"]
                replay = copy.deepcopy(base)
                replay.write(patch["id"], replacement["write_key"], replacement["value"], replacement["timestamp"])
                assert replay.hash() == replacement["bank_hash"]
                assert torch.equal(replay.keys[index], replacement["key"])
                assert torch.equal(replay.keys[1-index], base.keys[1-index])
                assert torch.equal(replay.values[1-index], base.values[1-index])
                assert replay.timestamps[1-index] == base.timestamps[1-index]
                for world, bank in (("A", base), ("B", replay)):
                    bank_hashes[(patch["id"], kind, world)] = bank.hash()
        events = [(row, event) for row, event in zip(rows, backend.generations) if row["phase"] == phase]
        for row, event in events:
            if row["method"] in {"real", "canonical_key"}:
                assert event["bank_hash"] == bank_hashes[(row["target_id"], row["method"], row["world"])]
            if row["method"] == "shuffled":
                real = next(e for r, e in events if r["target_id"] == row["target_id"] and r["world"] == row["world"] and r["method"] == "real")
                assert torch.equal(event["keys"], real["keys"])
                assert event["timestamps"] == real["timestamps"]
                assert torch.equal(event["values"], real["values"][saved["shuffle_permutation"]])


def test_pooled_value_writer_keeps_last_token_keys_even_in_canonical_key_diagnostic(tmp_path):
    result, _, _, _ = run(tmp_path, teacher=False, writer_mode="masked_mean")
    saved = torch.load(tmp_path / "banks/HC.pt", weights_only=True)
    real = load_bank(saved["bases"]["real"], tmp_path / "real.pt")
    canonical = load_bank(saved["bases"]["canonical_key"], tmp_path / "canon.pt")
    assert torch.equal(real.keys, torch.eye(2).flip(0))
    assert torch.equal(canonical.keys, torch.eye(2))
    assert torch.equal(real.values, torch.tensor([[2.], [0.]]))
    assert torch.equal(real.values, canonical.values)
    assert result["writer_mode"] == "masked_mean"


def test_teacher_strict_acceptance_is_separate_from_containment_and_budget_hit(tmp_path):
    result, rows, _, _ = run(tmp_path, wrapped_teacher=True, budget_hit=True)
    for phase in result["phases"].values():
        assert phase["teacher_acceptance"]["strict_pair_qualified"] == 1
        teacher = phase["methods"]["teacher"]
        assert teacher["count"] == 2 and teacher["paired_switch_em"] == .5
        assert teacher["paired_answer_containment_rate"] == 1
        assert teacher["a_budget_hit_rate"] == teacher["b_budget_hit_rate"] == 1
        assert all(value["count"] == 1 for value in phase["teacher_qualified_methods"].values())
    assert all(row["budget_hit"] for row in rows)


def test_target_subset_keeps_full_bank_and_is_repeatable_without_global_rng(tmp_path):
    import random
    random.seed(53)
    state = random.getstate()
    first, _, _, _ = run(tmp_path / "first", teacher=False, max_cases=1)
    second, _, _, _ = run(tmp_path / "second", teacher=False, max_cases=1)
    assert state == random.getstate()
    assert first["bank_records"] == 2 and first["evaluated_facts"] == 1
    a = json.loads((tmp_path / "first/assignments.json").read_text())
    b = json.loads((tmp_path / "second/assignments.json").read_text())
    assert a == b and len(a["selected_case_ids"]) == 1
    for phase in first["phases"]:
        saved = torch.load(tmp_path / f"first/banks/{phase}.pt", weights_only=True)
        base = load_bank(saved["bases"]["real"], tmp_path / f"{phase}.pt")
        assert len(base) == 2 and len(saved["interventions"]) == 1


def test_style_assignments_are_answer_blind_order_invariant_independent_and_balanced():
    rows = [dict(id=f"id-{i:03d}", entity=f"e-{i}", relation=f"r-{i % 4}", a="one", b="two") for i in range(100)]
    assigned = view_assignments(rows, 5, 5)
    reordered = [dict(row, a="CHANGED", b="LABELS") for row in reversed(rows)]
    assert {r["id"]: r for r in assigned} == {r["id"]: r for r in view_assignments(reordered, 5, 5)}
    assert any(r["q"] != r["s"] for r in assigned)
    for axis in ("q", "s"):
        counts = defaultdict(Counter)
        for item in assigned:
            counts[item["relation"]][item[axis]] += 1
        assert all(set(count) == {1, 2, 3, 4} and max(count.values()) - min(count.values()) <= 1 for count in counts.values())


@pytest.mark.parametrize("mutation,message", [("bank", "database mutated"), ("weights", "module weights")])
def test_detects_illegal_mutation_and_restores_flags_even_on_exception(tmp_path, mutation, message):
    module = TinyMemory()
    backend = FakeBackend(module, mutate=mutation)
    original_store = backend.vector_store
    with pytest.raises(RuntimeError, match=message):
        evaluate_interface(backend, module, packet(), tmp_path)
    assert backend.mode == "original" and backend.vector_store is original_store and backend.model.training
    assert module.scale.requires_grad and not module.fixed.requires_grad and module.training
    assert not backend.trace_retrieval
    assert not json.loads((tmp_path / "metrics.json").read_text())["complete"]


def test_generation_exception_restores_eval_mode_and_original_override(tmp_path):
    module = TinyMemory()
    module.eval()
    backend = FakeBackend(module)
    backend.model.eval()
    previous = backend.vector_override
    def fail(*args, **kwargs):
        assert backend.trace_retrieval
        raise RuntimeError("generation failure")
    backend.generate = fail
    with pytest.raises(RuntimeError, match="generation failure"):
        evaluate_interface(backend, module, packet(), tmp_path)
    assert backend.vector_override is previous and not backend.trace_retrieval
    assert not module.training and not backend.model.training
    assert module.scale.requires_grad and not module.fixed.requires_grad


@pytest.mark.parametrize("case", ["train", "duplicate", "shape", "budget", "equal", "max_cases"])
def test_reject_invalid_packets_before_generation(tmp_path, case):
    data, kwargs = packet(), {}
    if case == "train":
        data["split"] = "train"
    elif case == "duplicate":
        data["rows"][1]["id"] = data["rows"][0]["id"]
    elif case == "shape":
        data["pool"] = data["pool"][:, :, 0]
    elif case == "budget":
        data["rows"][0]["b_tokens"] = [1] * 32
    elif case == "equal":
        data["rows"][0]["b"] = data["rows"][0]["a"].upper() + "."
    else:
        kwargs["max_cases"] = 0
    module = TinyMemory()
    backend = FakeBackend(module)
    with pytest.raises(ValueError):
        evaluate_interface(backend, module, data, tmp_path, **kwargs)
    assert not backend.generations


def test_refuses_overwriting_an_existing_evaluation(tmp_path):
    run(tmp_path, teacher=False)
    with pytest.raises(FileExistsError):
        run(tmp_path, teacher=False)
