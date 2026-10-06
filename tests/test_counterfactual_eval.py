"""Causal bank-intervention evaluation checks without model downloads or GPUs."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vera_mem.counterfactual_eval import evaluate_counterfactual
from vera_mem.vector_store import PersistentVectorDB


WORDS = ("apple", "banana", "cherry", "date")


class TinyMemory(nn.Module):
    rank, key_dim, top_k, temperature = 1, 2, 1, .05

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def encode_key(self, features):
        return features[..., :2] * self.scale

    def encode_value(self, features):
        return features[..., 2:] * self.scale


class FakeBackend:
    device = torch.device("cpu")

    def __init__(self, module, wrong=False, mutate=None):
        self.model = nn.Linear(1, 1).requires_grad_(False)
        self.vector_vera = module
        self.mode = "original"
        self.vector_store = object()
        self.vector_override = object()
        self.oracle_values = object()
        self.vector_keys = object()
        self.vector_values = object()
        self.trace_retrieval = False
        self.wrong, self.mutate = wrong, mutate
        self.generations, self.scores = [], []

    def generate(self, question, max_new_tokens, **kwargs):
        assert max_new_tokens == 3
        assert not torch.is_grad_enabled()
        assert all(not p.requires_grad for p in self.vector_vera.parameters())
        assert self.oracle_values is None
        assert self.vector_keys is None and self.vector_values is None
        assert self.retrieval_trace == []
        index = int(question[1:])
        query = torch.eye(2)[index].view(1, 1, 2)
        if self.mode == "none":
            assert set(kwargs) == {"context"}
            assert self.vector_override is None and not self.trace_retrieval
            prediction = kwargs["context"].split("=")[-1]
        else:
            assert self.mode == "vector_vera" and not kwargs  # No context, ID, answer or world supplied.
            assert isinstance(self.vector_store, PersistentVectorDB)
            assert self.vector_store.keys.device.type == "cpu"
            info = self.vector_store.search(query)
            self.prefill_retrieval = info
            if self.vector_override is None:
                assert self.trace_retrieval
                first = info["indices"][:, -1]
                # Vary decode hits so aggregation cannot confuse prefill and decode.
                self.retrieval_trace.extend([
                    {"phase": "prefill", "indices": first},
                    {"phase": "decode", "indices": (first + 1) % 2},
                    {"phase": "decode", "indices": first},
                ])
                value = info["mixed_value"].reshape(-1)[0]
            else:
                assert not self.trace_retrieval
                value = self.vector_override.reshape(-1)[0]
            prediction = WORDS[int(value.round())]
        if self.wrong:
            prediction = "wrong-" + prediction
        self.generations.append(dict(question=question, context=kwargs.get("context"),
            mode=self.mode, prediction=prediction, bank_hash=self.vector_store.hash(),
            timestamps=self.vector_store.timestamps, keys=self.vector_store.keys,
            values=self.vector_store.values,
            override=self.vector_override.detach().clone() if self.vector_override is not None else None))
        return prediction, 3, .01

    def score(self, question, answer, **kwargs):
        assert not self.trace_retrieval and not torch.is_grad_enabled()
        assert (self.mode == "none") == ("context" in kwargs)
        self.scores.append((question, answer))
        if self.mutate == "bank":
            self.vector_store._values[0] += 1
            self.mutate = None
        elif self.mutate == "weights":
            self.vector_vera.scale.add_(1)
            self.mutate = None
        return SimpleNamespace(nll_sum=float(len(answer)), tokens=len(answer))


def packet(unrelated=False):
    return {"phases": {"canonical_support/canonical_query": {
        "evaluate_unrelated": unrelated,
        "cases": [
            dict(id="entity-0", question="q0", answer_a="apple", answer_b="banana",
                 support_a="q0=apple", support_b="q0=banana", support_p="paraphrase:q0=apple"),
            dict(id="entity-1", question="q1", answer_a="date", answer_b="cherry",
                 support_a="q1=date", support_b="q1=cherry", support_p="paraphrase:q1=date"),
        ],
        # A/B/P can change keys as well as values. Different P key norms test
        # normalization in the DB while preserving the underlying address.
        "supports": torch.tensor([[[1., 0., 0.], [2., 0., 1.], [3., 0., 0.]],
                                  [[0., 1., 3.], [0., 2., 2.], [0., 3., 3.]]]),
    }}}


def run(tmp_path, *, unrelated=False, teacher=True, wrong=False, mutate=None):
    module = TinyMemory()
    backend = FakeBackend(module, wrong=wrong, mutate=mutate)
    result = evaluate_counterfactual(backend, module, packet(unrelated), tmp_path,
                                     include_teacher=teacher, max_new_tokens=3)
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    phase = result["phases"]["canonical_support/canonical_query"]
    return result, phase, rows, backend, module


def test_real_cpu_bank_joint_switch_paraphrase_teacher_and_empty_generation(tmp_path):
    result, phase, rows, backend, module = run(tmp_path)
    assert result["complete"] and result["shared_weights_before"] == result["shared_weights_after"]
    assert phase["methods"]["real"]["paired_switch_em"] == 1
    assert phase["methods"]["real"]["changed_rate"] == 1
    assert phase["methods"]["real"]["a_recall_at_1"] == 1
    assert phase["methods"]["real"]["a_decode_correct_residency"] == .5
    assert phase["methods"]["oracle"]["paired_switch_em"] == 1
    assert phase["methods"]["teacher"]["paired_switch_em"] == 1
    assert phase["methods"]["empty"]["paired_switch_em"] == 0
    assert phase["paraphrase"]["joint_em"] == 1
    assert phase["paraphrase"]["prediction_unchanged_rate"] == 1
    assert phase["teacher_paraphrase"]["joint_em"] == 1
    assert phase["unrelated"] is None
    assert len(rows) == len(backend.generations) == result["generation_calls"] == 22
    assert sum(row["method"] == "empty" for row in rows) == 2
    assert all(set(row["answer_results"]) == {"A", "B"} for row in rows if row["method"] == "empty")
    assert result["generation_tokens"] == 66
    assert result["answer_scoring_tokens"] == sum(row["answer_scoring_tokens"] for row in rows)
    assert all(row["bank_hash_before"] == row["bank_hash_after"] for row in rows)
    assert all(row["teacher_context"] is None for row in rows if row["method"] != "teacher")
    # Evaluation restores caller flags; it does not permanently disable training.
    assert module.training and module.scale.requires_grad and backend.mode == "original"
    assert not isinstance(backend.vector_store, PersistentVectorDB)
    assert not backend.trace_retrieval


def test_independent_replacements_and_exact_shuffled_control_reconstruct(tmp_path):
    result, phase, rows, backend, _ = run(tmp_path, teacher=False)
    saved = torch.load(tmp_path / result["artifacts"]["banks"][0], weights_only=True)
    base_path = tmp_path / "base.pt"
    torch.save(saved["base"], base_path)
    base = PersistentVectorDB.load(base_path)
    assert base.hash() == saved["base_hash"] == phase["bank_A_hash"]
    for intervention in saved["interventions"]:
        target = intervention["index"]
        other = 1 - target
        for world in ("B", "P"):
            replacement = intervention[world]
            replay = copy.deepcopy(base)
            replay.write(intervention["id"], replacement["write_key"], replacement["value"], replacement["timestamp"])
            assert replay.hash() == replacement["bank_hash"]
            assert torch.equal(replay.keys[target], replacement["key"])
            assert torch.equal(replay.keys[other], base.keys[other])
            assert torch.equal(replay.values[other], base.values[other])
            assert replay.timestamps[other] == base.timestamps[other]
    # Match JSONL read order to backend calls and compare shuffled payloads
    # against the corresponding real bank under the same permutation.
    paired = list(zip(rows, backend.generations))
    for target in base.ids:
        for world in ("A", "B"):
            real = next(event for row, event in paired if row["target_id"] == target and row["method"] == "real" and row["world"] == world)
            shuffled = next(event for row, event in paired if row["target_id"] == target and row["method"] == "shuffled" and row["world"] == world)
            assert torch.equal(shuffled["keys"], real["keys"])
            assert shuffled["timestamps"] == real["timestamps"]
            assert torch.equal(shuffled["values"], real["values"][saved["shuffle_permutation"]])


def test_unrelated_question_answers_stay_correct_and_only_selected_phase_runs_it(tmp_path):
    result, phase, rows, _, _ = run(tmp_path, unrelated=True, teacher=False)
    assert result["generation_calls"] == 20  # 8 target conditions + 2 unrelated, per case.
    assert phase["unrelated"]["joint_em"] == 1
    assert phase["unrelated"]["prediction_unchanged_rate"] == 1
    other = [row for row in rows if row["unrelated"]]
    assert len(other) == 4
    assert all(row["target_id"] != row["query_id"] for row in other)
    assert "teacher" not in phase["methods"]
    assert phase["teacher_paraphrase"] is None


def test_wrong_to_wrong_changes_cannot_pass_paired_or_paraphrase_em(tmp_path):
    _, phase, _, _, _ = run(tmp_path, wrong=True, teacher=False)
    assert phase["methods"]["real"]["changed_rate"] == 1
    assert phase["methods"]["real"]["paired_switch_em"] == 0
    assert phase["methods"]["real"]["changed_but_not_both_correct_rate"] == 1
    assert phase["paraphrase"]["prediction_unchanged_rate"] == 1
    assert phase["paraphrase"]["joint_em"] == 0


@pytest.mark.parametrize("mutation,message", [("bank", "database mutated"), ("weights", "module weights")])
def test_detects_illegal_mutation_and_restores_backend_flags(tmp_path, mutation, message):
    module = TinyMemory()
    backend = FakeBackend(module, mutate=mutation)
    original_store = backend.vector_store
    with pytest.raises(RuntimeError, match=message):
        evaluate_counterfactual(backend, module, packet(), tmp_path, include_teacher=False, max_new_tokens=3)
    assert backend.mode == "original" and backend.vector_store is original_store
    assert module.scale.requires_grad and module.training and not backend.trace_retrieval
    assert not json.loads((tmp_path / "metrics.json").read_text())["complete"]


def test_rejects_normalized_equal_answers_and_duplicate_ids_before_generation(tmp_path):
    module = TinyMemory()
    backend = FakeBackend(module)
    data = packet()
    phase = next(iter(data["phases"].values()))
    phase["cases"][0]["answer_b"] = "APPLE."
    with pytest.raises(ValueError, match="answers must differ"):
        evaluate_counterfactual(backend, module, data, tmp_path)
    phase["cases"][0]["answer_b"] = "banana"
    phase["cases"][1]["id"] = phase["cases"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        evaluate_counterfactual(backend, module, data, tmp_path)
    assert not backend.generations


def test_generation_failure_disables_trace_and_restores_callers_state(tmp_path):
    module = TinyMemory()
    backend = FakeBackend(module)
    original = backend.vector_override
    def fail(*args, **kwargs):
        assert backend.trace_retrieval
        raise RuntimeError("interrupted generation")
    backend.generate = fail
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_counterfactual(backend, module, packet(), tmp_path, include_teacher=False, max_new_tokens=3)
    assert backend.vector_override is original
    assert not backend.trace_retrieval and backend.mode == "original"
