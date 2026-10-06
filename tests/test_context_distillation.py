"""Small causal CPU model tests; no downloaded model or GPU is required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vera_mem.context_distillation import (
    ContextDistillationBackend, distillation_loss, hidden_alignment_loss,
)
from vera_mem.vector_vera import VectorVeRA


class TinyTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return messages[-1]["content"] + "|"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [2 + ord(char) % 21 for char in text]

    def decode(self, *args, **kwargs):
        raise AssertionError("Continuation IDs must never be decoded and re-encoded")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(23, 5)
        self.down_proj = nn.Linear(5, 5, bias=False)
        self.norm = nn.LayerNorm(5)
        self.calls = []
        self.fail = False

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        if self.fail:
            raise RuntimeError("fake model failure")
        x = (self.embedding(input_ids) * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        # Different padding-position hidden states make a mistaken [:, -1]
        # gather observable when short prompts share a right-padded batch.
        hidden = self.norm(self.down_proj(x)) * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(5, 23, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=1)


def make_backend():
    torch.manual_seed(113)
    backend = object.__new__(ContextDistillationBackend)
    backend.model = TinyModel().requires_grad_(False).eval()
    backend.tokenizer = TinyTokenizer()
    backend.max_input_tokens = 512
    backend.target = backend.model.model.down_proj
    backend.mode = "vector_vera"
    backend.capture_layer_input = False
    backend.captured_input = None
    backend.vector_vera = VectorVeRA(5, 5, rank=3, key_dim=3, top_k=3)
    backend.vector_vera.b.data.fill_(.5)
    supports = torch.randn(5, 5)
    backend.vector_keys = backend.vector_vera.encode_key(supports)
    backend.vector_values = backend.vector_vera.encode_value(supports)
    backend.vector_store = backend.vector_override = backend.oracle_values = None
    backend.last_retrieval = backend.prefill_retrieval = None
    backend.trace_retrieval = False
    backend.retrieval_trace = []
    backend._answer_capture_batch_indices = backend._answer_capture_positions = None
    backend.hook = backend.target.register_forward_hook(backend._inject)
    return backend


def reference(backend, question, tokens, context=None):
    prompt = backend.prompt_ids(question, context)
    ids = torch.tensor([prompt + tokens])
    output = backend.model.model(input_ids=ids, attention_mask=torch.ones_like(ids))
    hidden = output.last_hidden_state[0, len(prompt) - 1:len(prompt) - 1 + len(tokens)]
    return hidden, backend.model.lm_head(hidden).float()


def test_variable_length_student_positions_equal_independent_causal_forward():
    backend = make_backend()
    questions, tokens = ["q", "longer question"], [[12, 7], [5, 9, 10]]
    expected = [reference(backend, question, target) for question, target in zip(questions, tokens)]
    actual = backend.forward_sequences(questions, tokens)
    torch.testing.assert_close(actual["hidden"], torch.cat([item[0] for item in expected]))
    torch.testing.assert_close(actual["logits"], torch.cat([item[1] for item in expected]))
    assert actual["counts"].tolist() == [2, 3]
    assert actual["batch_indices"].tolist() == [0, 0, 1, 1, 1]
    assert actual["positions"].tolist() == [1, 2, 15, 16, 17]
    assert actual["query_inputs"].shape == (5, 5)
    assert actual["hidden"].requires_grad and actual["logits"].requires_grad
    # Hidden output is after final normalization, not the pre-adapter input.
    torch.testing.assert_close(actual["hidden"].mean(-1), torch.zeros(5), atol=1e-6, rtol=0)
    assert not torch.allclose(actual["hidden"], actual["query_inputs"])
    ids, mask = backend.model.model.calls[-1]
    assert ids[0, :4].tolist() == backend.prompt_ids("q") + tokens[0]
    assert mask.sum(1).tolist() == [4, 19]
    assert torch.all(ids[mask == 0] == 0)


def test_teacher_context_length_alignment_keeps_identical_continuation_ids():
    backend = make_backend()
    questions, tokens, contexts = ["q", "longer"], [[12, 7], [5]], ["small fact", "a longer memory observation"]
    with backend.disabled():
        expected = [reference(backend, q, target, context)
                    for q, target, context in zip(questions, tokens, contexts)]
    teacher = backend.forward_sequences(questions, tokens, contexts=contexts, teacher=True)
    student = backend.forward_sequences(questions, tokens)
    torch.testing.assert_close(teacher["hidden"], torch.cat([item[0] for item in expected]))
    torch.testing.assert_close(teacher["logits"], torch.cat([item[1] for item in expected]))
    assert torch.all(teacher["positions"] > student["positions"])
    assert torch.equal(teacher["batch_indices"], student["batch_indices"])
    assert torch.equal(teacher["counts"], student["counts"])
    assert all(not value.requires_grad for value in teacher.values())
    with pytest.raises(ValueError, match="must not receive"):
        backend.forward_sequences(questions, tokens, contexts=contexts)


def test_teacher_does_not_overwrite_student_graph_diagnostics_or_bank():
    backend = make_backend()
    student = backend.forward_sequences(["q"], [[12, 7]])
    backend.trace_retrieval = True
    backend.retrieval_trace.append({"sentinel": 13})
    fields = ContextDistillationBackend._TEACHER_TRANSIENT_FIELDS
    before = {name: getattr(backend, name) for name in fields if hasattr(backend, name)}
    keys, values = backend.vector_keys, backend.vector_values
    teacher = backend.forward_sequences(["q"], [[12, 7]], contexts=["a fact"], teacher=True)
    assert backend.mode == "vector_vera"
    assert backend.vector_keys is keys and backend.vector_values is values
    for name, value in before.items():
        assert getattr(backend, name) is value
    assert backend.retrieval_trace == [{"sentinel": 13}]
    loss = distillation_loss(student["logits"], teacher["logits"])
    loss = loss + hidden_alignment_loss(student["hidden"], teacher["hidden"])
    loss.backward()
    for name in ("Wq", "Wk", "Wv"):
        assert getattr(backend.vector_vera, name).weight.grad.abs().sum() > 0
    assert backend.vector_vera.b.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_teacher_failure_restores_state_and_student_capture_failure_cleans_up():
    backend = make_backend()
    backend.model.model.fail = True
    backend.last_answer_query_inputs = sentinel = torch.tensor([19.])
    with pytest.raises(RuntimeError, match="fake model failure"):
        backend.forward_sequences(["q"], [[3]], contexts=["fact"], teacher=True)
    assert backend.mode == "vector_vera"
    assert backend.last_answer_query_inputs is sentinel
    assert backend._answer_capture_positions is None
    assert backend._answer_capture_batch_indices is None
    with pytest.raises(RuntimeError, match="fake model failure"):
        backend.forward_sequences(["q"], [[3]])
    assert backend._answer_capture_positions is None
    assert backend._answer_capture_batch_indices is None


def test_first_prediction_cannot_see_first_continuation_token():
    backend = make_backend()
    left = backend.forward_sequences(["q"], [[3, 4]])
    right = backend.forward_sequences(["q"], [[9, 4]])
    torch.testing.assert_close(left["logits"][0], right["logits"][0])
    assert not torch.allclose(left["logits"][1], right["logits"][1])


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_full_vocab_kl_matches_definition_and_stops_teacher_gradient(direction):
    student = torch.tensor([[2., -1., .5], [.2, -.3, 1.1]], requires_grad=True)
    teacher = torch.tensor([[-2., .3, 1.], [1., -.5, .1]], requires_grad=True)
    temperature = 2.
    result = distillation_loss(student, teacher, direction=direction,
                               temperature=temperature, reduction="none")
    p, q = F.softmax(student / temperature, -1), F.softmax(teacher.detach() / temperature, -1)
    if direction == "forward":
        p, q = q, p
    expected = (p * (p.log() - q.log())).sum(-1) * temperature ** 2
    torch.testing.assert_close(result, expected)
    result.mean().backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None
    torch.testing.assert_close(distillation_loss(student, student, direction=direction), torch.tensor(0.))


def test_kl_extreme_logits_stay_float32_and_finite():
    student = torch.tensor([[1000., -1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[-1000., 1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    loss = distillation_loss(student, teacher, reduction="none")
    assert loss.dtype == torch.float32 and torch.isfinite(loss).all()
    assert loss.item() == 2000.
    loss.sum().backward()
    assert torch.isfinite(student.grad).all() and teacher.grad is None


def test_hidden_cosine_keeps_student_graph_and_detaches_teacher():
    student = torch.tensor([[1., 0], [0, 1.]], requires_grad=True)
    teacher = torch.tensor([[0., 1], [1., 1]], requires_grad=True)
    loss = hidden_alignment_loss(student, teacher, reduction="none")
    torch.testing.assert_close(loss, torch.tensor([1., 1 - 2 ** -.5]))
    loss.mean().backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None
    torch.testing.assert_close(hidden_alignment_loss(student, student), torch.tensor(0.))


class FixedHead(nn.Module):
    def __init__(self, token):
        super().__init__()
        self.token = token

    def forward(self, hidden):
        logits = torch.full((*hidden.shape[:-1], 23), -1000., device=hidden.device)
        logits[..., self.token] = 1000.
        return logits


def test_sampling_preserves_eos_and_truncation_without_synthetic_eos():
    backend = make_backend()
    backend.model.lm_head = FixedHead(1)
    assert backend.sample_student(["q", "longer"], max_new_tokens=4) == [[1], [1]]
    assert len(backend.model.model.calls) == 1
    backend.model.lm_head = FixedHead(7)
    assert backend.sample_student(["q"], max_new_tokens=3) == [[7, 7, 7]]
    # Every generated prefix is sent through the actual adapter hook.
    calls = backend.model.model.calls[-3:]
    prompt = backend.prompt_ids("q")
    assert [call[0][0].tolist() for call in calls] == [prompt, prompt + [7], prompt + [7, 7]]
    assert backend.last_retrieval["indices"].shape[-1] == 3
    assert all(parameter.grad is None for parameter in backend.vector_vera.parameters())


def test_sampling_is_seed_reproducible_and_rejects_oracle_or_cpu_bank():
    backend = make_backend()
    backend.model.generation_config.eos_token_id = None
    backend.tokenizer.eos_token_id = None
    left = backend.sample_student(["q", "qq"], generator=torch.Generator().manual_seed(4))
    right = backend.sample_student(["q", "qq"], generator=torch.Generator().manual_seed(4))
    assert left == right and all(len(tokens) == 4 for tokens in left)
    backend.oracle_values = torch.ones(2, 3)
    with pytest.raises(ValueError, match="oracle"):
        backend.sample_student(["q"])
    backend.oracle_values = None
    backend.vector_store = object()
    with pytest.raises(ValueError, match="GPU tensor bank"):
        backend.sample_student(["q"])


class PrefixScheduleHead(nn.Module):
    """Choose a token from the exact prefix to force distinct EOS times."""
    def __init__(self, backbone, prompts, schedules):
        super().__init__()
        self.backbone = backbone
        self.prompts = prompts
        self.schedules = schedules

    def forward(self, hidden):
        ids, mask = self.backbone.calls[-1]
        logits = torch.full((len(hidden), 23), -1000., device=hidden.device)
        for row, length in enumerate(mask.sum(-1).tolist()):
            prefix = ids[row, :length].tolist()
            matches = [index for index, prompt in enumerate(self.prompts)
                       if prefix[:len(prompt)] == prompt]
            assert len(matches) == 1
            index = matches[0]
            step = len(prefix) - len(self.prompts[index])
            assert prefix[len(self.prompts[index]):] == self.schedules[index][:step]
            logits[row, self.schedules[index][step]] = 1000.
        return logits


def test_batched_sampling_drops_eos_rows_and_keeps_original_order_and_padding_masks():
    backend = make_backend()
    questions = ["q", "a substantially longer prompt", "mid"]
    prompts = [backend.prompt_ids(question) for question in questions]
    schedules = [[1], [8, 9, 1], [7, 6, 5, 4]]
    backend.model.lm_head = PrefixScheduleHead(backend.model.model, prompts, schedules)
    keys, values = backend.vector_keys, backend.vector_values
    result = backend.sample_student(questions, max_new_tokens=4, temperature=1.)
    assert result == schedules
    assert backend.vector_keys is keys and backend.vector_values is values
    calls = backend.model.model.calls
    assert [len(ids) for ids, _ in calls] == [3, 2, 2, 1]
    active_rows = [[0, 1, 2], [1, 2], [1, 2], [2]]
    for step, ((ids, mask), active) in enumerate(zip(calls, active_rows)):
        for batch_row, original_row in enumerate(active):
            expected = prompts[original_row] + schedules[original_row][:step]
            length = int(mask[batch_row].sum())
            assert length == len(expected)
            assert ids[batch_row, :length].tolist() == expected
            assert mask[batch_row, :length].eq(1).all()
            assert mask[batch_row, length:].eq(0).all()
            assert ids[batch_row, length:].eq(backend.tokenizer.pad_token_id).all()


@torch.no_grad()
def old_sequential_greedy(backend, questions, max_new_tokens):
    """Independent reference: unpadded one-example forwards and final position."""
    result = []
    for question in questions:
        prompt = backend.prompt_ids(question)
        continuation = []
        for _ in range(max_new_tokens):
            ids = torch.tensor([prompt + continuation])
            hidden = backend.model.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
            token = int(backend.model.lm_head(hidden[0, -1]).argmax())
            continuation.append(token)
            if token == backend.tokenizer.eos_token_id:
                break
        result.append(continuation)
    return result


def test_batched_and_single_row_greedy_match_old_sequential_without_padding_predictions():
    backend = make_backend()
    questions = ["q", "a considerably longer question", "mid"]
    expected = old_sequential_greedy(backend, questions, max_new_tokens=4)
    batched = backend.sample_student(questions, max_new_tokens=4, temperature=0.)
    individual = [backend.sample_student([question], max_new_tokens=4, temperature=0.)[0]
                  for question in questions]
    assert batched == individual == expected


@pytest.mark.parametrize("kwargs", [
    {"questions": [], "continuations": []},
    {"questions": ["q"], "continuations": [[]]},
    {"questions": ["q"], "continuations": [[-1]]},
    {"questions": ["q"], "continuations": [[True]]},
    {"questions": ["q"], "continuations": [[2]], "teacher": True, "contexts": ["a", "b"]},
])
def test_invalid_sequence_inputs_fail_before_model_execution(kwargs):
    backend = make_backend()
    with pytest.raises((ValueError, TypeError)):
        backend.forward_sequences(**kwargs)
    assert not backend.model.model.calls
