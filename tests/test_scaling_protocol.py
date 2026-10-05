"""CPU checks of the data-scaling and validation reporting protocol."""

from collections import Counter
from types import SimpleNamespace
import random

import pytest
import torch
from torch.nn import functional as F

from vera_mem.data import Example, MEMORY_WORDS
from vera_mem.scaling_run import datasets, stats, validation, evaluate


def test_training_sizes_are_nested_balanced_and_entity_disjoint_from_evaluation():
    data = datasets()
    identifiers = {name: {example.id for example in rows} for name, rows in data.items()}
    assert {name: len(rows) for name, rows in data.items()} == {
        "train": 4096, "dev": 64, "stream": 128, "control": 64,
    }
    for name, rows in data.items():
        assert len(identifiers[name]) == len(rows)
        assert len({row.metadata["entity"] for row in rows}) == len(rows)
        for other in data:
            if other != name:
                assert identifiers[name].isdisjoint(identifiers[other])
    previous_ids = []
    for size in (32, 128, 1024, 4096):
        selected = data["train"][:size]
        current_ids = [row.id for row in selected]
        assert current_ids[:len(previous_ids)] == previous_ids
        assert Counter(row.answer for row in selected) == {
            word: size // len(MEMORY_WORDS) for word in MEMORY_WORDS
        }
        previous_ids = current_ids
    # The hidden control labels are not observations available to an online
    # learner. Their metadata distinguishes guessing from a memorization test.
    assert all(row.metadata["never_written"] for row in data["control"])
    assert all(not row.metadata["never_written"] for row in data["stream"])


def test_data_scaling_pool_is_reproducible_without_consuming_training_rng():
    random.seed(19)
    before = random.getstate()
    first = datasets()
    assert random.getstate() == before
    random.seed(817)
    second = datasets()
    assert first == second
    # Fact identity and revealed content remain the same at every data budget;
    # the small run is not an independently sampled easier problem.
    for left, right in zip(first["train"][:128], second["train"][:1024]):
        assert (left.id, left.question, left.support, left.answer) == (
            right.id, right.question, right.support, right.answer,
        )


def test_stats_use_answer_token_weighting_and_only_eligible_retrieval_queries():
    rows = [
        dict(id="a", method="real", expected_in_bank=True,
             selected_ids=["other", "a"], em=1, nll_sum=2., tokens=1),
        dict(id="b", method="shuffled", expected_in_bank=True,
             selected_ids=["b", "other"], em=0, nll_sum=8., tokens=4),
        dict(id="c", method="real", expected_in_bank=False,
             selected_ids=["c"], em=0, nll_sum=7., tokens=2),
        dict(id="d", method="oracle", expected_in_bank=True,
             selected_ids=["d"], em=1, nll_sum=3., tokens=3),
    ]
    result = stats(rows)
    assert result["count"] == 4
    assert result["em"] == .5
    assert result["answer_token_nll"] == 20 / 10
    assert result["recall_at_1"] == .5
    assert result["recall_at_4"] == 1.
    assert "recall_at_1" not in stats(rows[2:])
    assert stats([]) == {"count": 0}


class IdentityMemory:
    """Make the dev/background row mapping observable without a language model."""

    @staticmethod
    def encode_query(x):
        return F.normalize(x, dim=-1)

    @staticmethod
    def encode_key(x):
        return F.normalize(x, dim=-1)

    @staticmethod
    def encode_value(x):
        return x * 2


class RecordingValidationBackend:
    def __init__(self, supports):
        self.supports = supports
        self.mode = "none"
        self.vector_store = object()
        self.vector_override = object()
        self.oracle_values = torch.tensor([999.])
        self.calls = []

    def batched_loss(self, questions, answers, include_eos):
        assert not include_eos
        assert not torch.is_grad_enabled()
        assert self.mode == "vector_vera"
        assert self.vector_store is None and self.vector_override is None
        indices = [int(question) for question in questions]
        is_oracle = self.oracle_values is not None
        self.calls.append((is_oracle, indices))
        if is_oracle:
            torch.testing.assert_close(self.oracle_values, 2 * self.supports[indices])
        # Unequal token counts expose accidental unweighted averaging. These
        # are synthetic losses to verify aggregation, not model-quality scores.
        counts = torch.tensor([len(answer) for answer in answers])
        per_example_nll = torch.tensor([index + 1. for index in indices])
        if is_oracle:
            per_example_nll *= .5
        self.last_answer_token_counts = counts
        self.last_answer_nll_sums = per_example_nll * counts
        self.last_answer_residual_ratio = torch.full((int(counts.sum()),), .25 if is_oracle else .125)
        return (self.last_answer_nll_sums / counts).mean()


def test_validation_background_competes_but_oracle_values_match_dev_rows_and_reset():
    supports = torch.eye(4)
    questions = supports.clone()
    # The first query matches a background row more closely than its gold row.
    questions[0] = torch.tensor([.8, .6, 0., 0.])
    background = questions[:1].clone()
    examples = [Example(str(i), str(i), "x" * (i + 1), "support", "paraphrase") for i in range(4)]
    backend = RecordingValidationBackend(supports)
    module = IdentityMemory()
    with_background = validation(backend, module, examples, questions, supports,
                                 batch_size=3, background_support=background)
    assert with_background["retrieval_at_1"] == .75
    assert with_background["retrieval_at_4"] == 1.
    assert with_background["real_answer_token_nll"] == pytest.approx(30 / 10)
    assert with_background["oracle_answer_token_nll"] == pytest.approx(15 / 10)
    assert with_background["real_post_addition_residual_to_base_rms_mean"] == .125
    assert with_background["oracle_post_addition_residual_to_base_rms_mean"] == .25
    assert backend.calls == [(False, [0, 1, 2]), (False, [3]),
                             (True, [0, 1, 2]), (True, [3])]
    assert backend.oracle_values is None
    assert backend.vector_keys.shape == (5, 4)
    # Validation leaves its bank installed; the training loop must overwrite
    # that bank before its next episode. A second validation installs its own
    # bank, and a preceding oracle batch cannot leak into its real condition.
    without_background = validation(backend, module, examples, questions, supports, batch_size=2)
    assert without_background["retrieval_at_1"] == 1.
    assert backend.vector_keys.shape == (4, 4)
    assert backend.oracle_values is None
    assert backend.calls[4:6] == [(False, [0, 1]), (False, [2, 3])]


def test_decode_residency_is_token_weighted_and_switches_have_explicit_denominator():
    rows = [
        dict(id="a", method="real", expected_in_bank=True, selected_ids=["a"],
             em=1, nll_sum=1., tokens=1, decode_correct_residency=1.,
             decode_correct_hits=1, decode_query_count=1, switch_count=0, retrieval_transition_count=1),
        dict(id="b", method="real", expected_in_bank=True, selected_ids=["b"],
             em=0, nll_sum=2., tokens=1, decode_correct_residency=1/3,
             decode_correct_hits=1, decode_query_count=3, switch_count=2, retrieval_transition_count=3),
    ]
    result = stats(rows)
    assert result["decode_correct_residency"] == .5  # 2/4, not mean(1, 1/3).
    assert result["decode_query_count"] == 4 and result["decode_examples"] == 2
    assert result["switch_count_mean"] == 1
    assert result["top1_switch_rate"] == .5


class TracingEvaluationBackend:
    def generate(self, question, max_new_tokens):
        assert self.trace_retrieval and self.retrieval_trace == []
        self.prefill_retrieval = {"indices": torch.tensor([[[0]]])}
        self.retrieval_trace.extend([
            dict(phase="prefill", indices=torch.tensor([[0]])),
            dict(phase="decode", indices=torch.tensor([[1]])),
            dict(phase="decode", indices=torch.tensor([[0]])),
        ])
        return "apple", 3, .1

    def score(self, question, answer):
        assert not self.trace_retrieval  # Teacher-forced evidence must never enter the trace.
        assert len(self.retrieval_trace) == 3
        return SimpleNamespace(nll_sum=2., tokens=1)


def test_evaluate_serializes_generated_prefix_addresses_and_disables_scoring_trace(tmp_path):
    backend = TracingEvaluationBackend()
    db = SimpleNamespace(ids=["a", "b"])
    example = Example("a", "question", "apple", "support", "paraphrase")
    rows = evaluate(backend, SimpleNamespace(rank=2), db, [example], "real", "final", tmp_path)
    row = rows[0]
    assert row["selected_ids"] == ["a"]
    assert [x["selected_ids"] for x in row["token_retrieval_sequence"]] == [["a"], ["b"], ["a"]]
    assert row["decode_correct_residency"] == .5
    assert row["decode_query_count"] == 2
    assert row["switch_count"] == 2
    assert row["retrieval_transition_count"] == 2
    assert not backend.trace_retrieval
    assert (tmp_path / "predictions.jsonl").exists()


def test_failed_generation_also_disables_trace_before_any_scoring(tmp_path):
    class FailingBackend:
        def generate(self, *_args, **_kwargs):
            assert self.trace_retrieval
            raise RuntimeError("generation failed")

    backend = FailingBackend()
    with pytest.raises(RuntimeError, match="generation failed"):
        evaluate(backend, SimpleNamespace(rank=2), SimpleNamespace(ids=[]),
                 [Example("a", "q", "apple", "s", "p")], "real", "final", tmp_path)
    assert not backend.trace_retrieval
