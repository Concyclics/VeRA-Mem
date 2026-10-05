"""CPU model tests for variable-length batching and differentiable oracle reads."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vera_mem.scaling_backend import ScalingQwenBackend
from vera_mem.vector_vera import VectorVeRA


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return messages[-1]["content"] + "|"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [2 + ord(char) % 19 for char in text]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(21, 5)
        self.down_proj = nn.Linear(5, 5, bias=False)
        self.calls = []
        self.fail = False

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        if self.fail:
            raise RuntimeError("fake model failure")
        # A prefix-dependent causal feature makes both the wrong gather position
        # and the wrong label shift observable, unlike a context-free stub.
        x = (self.embedding(input_ids) * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        return SimpleNamespace(last_hidden_state=self.down_proj(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(5, 21, bias=False)


def make_backend():
    torch.manual_seed(813)
    backend = object.__new__(ScalingQwenBackend)
    backend.model = TinyModel().requires_grad_(False).eval()
    backend.tokenizer = TinyTokenizer()
    backend.max_input_tokens = 100
    backend.target = backend.model.model.down_proj
    backend.mode = "none"
    backend.bank = backend.reader = None
    backend.slot = 0
    backend.value = None
    backend.injection_start = 0
    backend.feature_cache = {}
    backend.layer_feature_cache = {}
    backend.capture_layer_input = False
    backend.captured_input = None
    backend.vector_vera = VectorVeRA(5, 5, rank=3, key_dim=3, top_k=2)
    backend.vector_vera.b.data.fill_(0.5)
    backend.vector_keys = backend.vector_values = backend.vector_store = None
    backend.vector_override = None
    backend.last_retrieval = backend.prefill_retrieval = None
    backend.oracle_values = None
    backend._answer_capture_batch_indices = backend._answer_capture_positions = None
    backend.hook = backend.target.register_forward_hook(backend._inject)
    return backend


def explicit_loss(backend, question, answer, include_eos):
    prompt = backend.prompt_ids(question)
    target = backend.tokenizer.encode(answer)
    if include_eos:
        target.append(backend.tokenizer.eos_token_id)
    ids = torch.tensor([prompt + target])
    hidden = backend.model.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    logits = backend.model.lm_head(hidden)[0]
    # This reference computes full vocabulary logits, then selects the usual
    # shifted causal-LM positions independently of the batched implementation.
    all_labels = ids[0, 1:]
    losses = F.cross_entropy(logits[:-1], all_labels, reduction="none")
    return losses[len(prompt) - 1:].mean()


def test_layer_features_gather_each_unpadded_prompt_last_token_and_restore_mode():
    backend = make_backend()
    backend.mode = "vector_vera"
    sentinel = torch.tensor([77.0])
    backend.captured_input = sentinel
    texts = ["x", "longer text", "mid"]
    features = backend.layer_features(texts, batch_size=2)
    expected = torch.stack([
        backend.model.model.embedding(torch.tensor(backend.prompt_ids(text))).sum(dim=0)
        for text in texts
    ])
    torch.testing.assert_close(features, expected)
    assert features.shape == (3, 5)
    assert features.device.type == "cpu" and features.dtype == torch.float32
    assert not features.requires_grad
    assert backend.mode == "vector_vera"
    assert backend.captured_input is sentinel
    assert not backend.capture_layer_input
    assert not backend.target._forward_pre_hooks
    ids, mask = backend.model.model.calls[0]
    assert mask[0].sum() < mask[1].sum()
    assert torch.all(ids[mask == 0] == backend.tokenizer.pad_token_id)
    assert backend.layer_feature_cache["mid"].shape == (5,)


def test_feature_failure_removes_temporary_hook_and_restores_mode():
    backend = make_backend()
    backend.mode = "vector_vera"
    backend.model.model.fail = True
    with pytest.raises(RuntimeError, match="fake model failure"):
        backend.layer_features(["a"])
    assert backend.mode == "vector_vera"
    assert not backend.target._forward_pre_hooks
    assert backend.layer_feature_cache == {}


@pytest.mark.parametrize("include_eos", [False, True])
def test_batch_loss_matches_individual_causal_shift_not_padding_or_token_weighted_mean(include_eos):
    backend = make_backend()
    questions, answers = ["q", "long question", "qqq"], ["abcd", "z", "uv"]
    expected = torch.stack([
        explicit_loss(backend, q, a, include_eos) for q, a in zip(questions, answers)
    ]).mean()
    actual = backend.batched_loss(questions, answers, include_eos=include_eos)
    torch.testing.assert_close(actual, expected)
    counts = torch.tensor([len(answer) + int(include_eos) for answer in answers])
    assert torch.equal(backend.last_answer_token_counts, counts)
    torch.testing.assert_close(actual, (backend.last_answer_nll_sums / counts).mean())
    assert backend.last_answer_query_inputs.shape == (int(counts.sum()), 5)
    assert backend.last_answer_batch_indices.tolist() == [0] * counts[0] + [1] * counts[1] + [2] * counts[2]
    assert backend._answer_capture_positions is None
    assert backend._answer_capture_batch_indices is None
    # First supervised position predicts the first answer token from the prompt.
    features = backend.layer_features(questions)
    first_indices = torch.tensor([0, counts[0], counts[0] + counts[1]])
    torch.testing.assert_close(backend.last_answer_query_inputs[first_indices], features)


def test_oracle_values_broadcast_by_example_and_writer_reader_address_gradients_flow():
    backend = make_backend()
    backend.mode = "vector_vera"
    supports = torch.randn(2, 5)
    backend.oracle_values = backend.vector_vera.encode_value(supports)
    loss = backend.batched_loss(["q", "much longer"], ["az", "b"])
    assert backend.last_answer_queries.shape == (5, 3)
    # An explicit addressing objective is required in an oracle curriculum:
    # answer loss cannot train which record to select when selection is forced.
    desired = torch.randn_like(backend.last_answer_queries)
    address_loss = (backend.last_answer_queries - desired).square().mean()
    (loss + address_loss).backward()
    assert backend.vector_vera.Wv.weight.grad.abs().sum() > 0
    assert backend.vector_vera.b.grad.abs().sum() > 0
    assert backend.vector_vera.Wq.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backend.model.parameters())
    assert backend.last_retrieval["indices"].shape[:2] == backend.model.model.calls[-1][0].shape
    assert backend.last_retrieval["indices"].numel() == 0
    # A batched oracle is not accidentally broadcast along the sequence axis.
    x = torch.randn(2, 7, 5)
    raw = torch.randn(2, 7, 5)
    actual = backend._inject(None, (x,), raw)
    expected = torch.stack([
        raw[row] + backend.vector_vera.delta_from_value(x[row], backend.oracle_values[row])
        for row in range(2)
    ])
    torch.testing.assert_close(actual, expected)


def test_real_retrieval_path_remains_differentiable_without_oracle():
    backend = make_backend()
    backend.mode = "vector_vera"
    supports = torch.randn(4, 5)
    backend.vector_keys = backend.vector_vera.encode_key(supports)
    backend.vector_values = backend.vector_vera.encode_value(supports)
    backend.batched_loss(["q", "long question"], ["xy", "a"]).backward()
    for name in ("Wq", "Wk", "Wv"):
        assert getattr(backend.vector_vera, name).weight.grad.abs().sum() > 0


def test_invalid_batches_fail_clearly_and_oracle_failure_cleans_capture_state():
    backend = make_backend()
    with pytest.raises(ValueError, match="same nonzero"):
        backend.batched_loss(["q"], [])
    with pytest.raises(ValueError, match="supervised token"):
        backend.batched_loss(["q"], [""], include_eos=False)
    with pytest.raises(TypeError, match="batches"):
        backend.batched_loss("q", "a")
    with pytest.raises(ValueError, match="positive integer"):
        backend.layer_features(["q"], batch_size=0)
    assert backend.layer_features([]).shape == (0, 5)
    backend.mode = "vector_vera"
    backend.oracle_values = torch.randn(1, 3)
    with pytest.raises(ValueError, match="batch, rank"):
        backend.batched_loss(["q", "qq"], ["a", "b"])
    assert backend._answer_capture_positions is None
    assert backend._answer_capture_batch_indices is None


def test_generation_trace_keeps_only_last_prefill_position_then_decode_indices():
    backend = make_backend()
    backend.mode = "vector_vera"
    backend.trace_retrieval = True
    backend.retrieval_trace = []
    backend.vector_vera.top_k = 1
    with torch.no_grad():
        backend.vector_vera.Wq.weight.zero_()
        backend.vector_vera.Wq.weight[:, :3].copy_(torch.eye(3))
    backend.vector_keys = torch.eye(3)
    backend.vector_values = torch.ones(3, 3)
    prefill = torch.tensor([[[1., 0, 0, 0, 0], [0, 1., 0, 0, 0]]])
    backend._inject(None, (prefill,), torch.ones_like(prefill))
    decode = torch.tensor([[[0, 0, 1., 0, 0]]])
    backend._inject(None, (decode,), torch.ones_like(decode))
    assert [row["phase"] for row in backend.retrieval_trace] == ["prefill", "decode"]
    assert [row["indices"].tolist() for row in backend.retrieval_trace] == [[[1]], [[2]]]
    assert all(set(row) == {"phase", "indices"} for row in backend.retrieval_trace)
    backend.oracle_values = torch.ones(1, 3)
    backend._inject(None, (decode,), torch.ones_like(decode))
    assert backend.retrieval_trace[-1]["indices"].shape == (1, 0)
    backend.trace_retrieval = False
    backend.batched_loss(["q"], ["a"])
    assert len(backend.retrieval_trace) == 3


def test_residual_ratio_measures_post_bfloat16_addition_without_detaching_training():
    backend = make_backend()
    backend.mode = "vector_vera"
    backend.oracle_values = torch.ones(1, 3)
    with torch.no_grad():
        backend.vector_vera.A.zero_()
        backend.vector_vera.A[:, :3].copy_(torch.eye(3))
        backend.vector_vera.B.zero_()
        backend.vector_vera.B[:3].copy_(torch.eye(3))
        backend.vector_vera.b.fill_(1e-4)
    x = torch.ones(1, 2, 5, dtype=torch.bfloat16)
    output = torch.ones_like(x)
    backend._answer_capture_batch_indices = torch.tensor([0])
    backend._answer_capture_positions = torch.tensor([1])
    small = backend._inject(None, (x,), output)
    # This residual exists mathematically but rounds away during BF16 addition.
    assert torch.equal(small, output)
    assert backend.last_residual_ratio.count_nonzero() == 0
    assert backend.last_answer_residual_ratio.shape == (1,)
    with torch.no_grad():
        backend.vector_vera.b.fill_(.5)
    result = backend._inject(None, (x,), output)
    expected_ratio = torch.tensor((3 * .5**2 / 5)**.5)
    torch.testing.assert_close(backend.last_residual_ratio, expected_ratio.expand(1, 2))
    assert not backend.last_residual_ratio.requires_grad
    assert not backend.last_answer_residual_ratio.requires_grad
    result.float().sum().backward()
    assert backend.vector_vera.b.grad.abs().sum() > 0
