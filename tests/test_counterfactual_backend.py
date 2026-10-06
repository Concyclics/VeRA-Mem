"""Each counterfactual query must read only its own single-intervention bank."""
import pytest
import torch

from vera_mem.context_distillation import ContextDistillationBackend, distillation_loss
from vera_mem.counterfactual_backend import BatchedStableVectorVeRA, CounterfactualBackend
from vera_mem.stable_vector_vera import StableVectorVeRA
from test_context_distillation import FixedHead, make_backend as make_plain_backend


def make_module(cls=BatchedStableVectorVeRA, top_k=3):
    torch.manual_seed(29)
    module = cls(5, 5, rank=3, key_dim=3, top_k=top_k, temperature=.7, seed=6)
    module.fit_statistics(torch.randn(23, 5), torch.randn(23, 5))
    with torch.no_grad():
        module.b.fill_(.5)
    return module


def make_backend():
    backend = make_plain_backend()
    backend.__class__ = CounterfactualBackend
    backend.vector_vera = make_module()
    supports = torch.randn(2, 5, 5)
    backend.vector_keys = backend.vector_vera.encode_key(supports)
    backend.vector_values = backend.vector_vera.encode_value(supports)
    return backend


def test_state_dict_is_strictly_compatible_and_shared_bank_path_is_unchanged():
    original, batched = make_module(StableVectorVeRA), make_module()
    assert set(original.state_dict()) == set(batched.state_dict())
    assert set(dict(original.named_parameters())) == set(dict(batched.named_parameters()))
    assert set(dict(original.named_buffers())) == set(dict(batched.named_buffers()))
    batched.load_state_dict(original.state_dict(), strict=True)
    original.load_state_dict(batched.state_dict(), strict=True)
    assert batched.configuration() == original.configuration()
    x, support = torch.randn(2, 4, 5), torch.randn(7, 5)
    keys, values = original.encode_key(support), original.encode_value(support)
    expected, info_expected = original(x, keys, values, return_info=True)
    actual, info_actual = batched(x, keys, values, return_info=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for field in info_expected:
        torch.testing.assert_close(info_actual[field], info_expected[field], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_independent_batched_retrieval_matches_individual_original_forwards(dtype):
    module = make_module()
    x, support = torch.randn(3, 4, 5).to(dtype), torch.randn(3, 7, 5)
    keys, values = module.encode_key(support), module.encode_value(support)
    actual, info = module(x, keys, values, return_info=True)
    reference = [module(x[row], keys[row], values[row], return_info=True) for row in range(3)]
    expected = torch.stack([item[0] for item in reference])
    torch.testing.assert_close(actual, expected, atol=.01 if dtype == torch.bfloat16 else 1e-6, rtol=1e-5)
    assert actual.dtype == dtype and actual.shape == (3, 4, 5)
    for field in info:
        torch.testing.assert_close(info[field], torch.stack([item[1][field] for item in reference]))
    torch.testing.assert_close(info["weights"].sum(-1), torch.ones(3, 4))


def test_bank_intervention_cannot_change_other_rows_outputs_or_selections():
    module = make_module()
    x, support = torch.randn(2, 3, 5), torch.randn(2, 6, 5)
    keys, values = module.encode_key(support), module.encode_value(support)
    before, before_info = module(x, keys, values, return_info=True)
    changed_keys, changed_values = keys.clone(), values.clone()
    changed_keys[1] = torch.randn_like(changed_keys[1])
    changed_values[1] += 10.
    after, after_info = module(x, changed_keys, changed_values, return_info=True)
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
    assert not torch.allclose(before[1], after[1])
    for field in before_info:
        torch.testing.assert_close(before_info[field][0], after_info[field][0], atol=0, rtol=0)


def test_row_loss_has_no_gradient_into_other_rows_keys_or_values():
    module = make_module()
    x = torch.randn(2, 3, 5, requires_grad=True)
    keys = torch.randn(2, 6, 3, requires_grad=True)
    values = torch.randn(2, 6, 3, requires_grad=True)
    output = module(x, keys, values)
    output[0].square().sum().backward()
    assert keys.grad[0].abs().sum() > 0 and values.grad[0].abs().sum() > 0
    assert x.grad[0].abs().sum() > 0
    assert keys.grad[1].eq(0).all() and values.grad[1].eq(0).all() and x.grad[1].eq(0).all()


def test_batched_encoder_gradients_match_individual_forward_accumulation():
    batched, individual = make_module(), make_module()
    x = torch.randn(2, 4, 5)
    support = torch.randn(2, 6, 5)
    batched_support = support.clone().requires_grad_(True)
    individual_support = support.clone().requires_grad_(True)
    result = batched(x, batched.encode_key(batched_support), batched.encode_value(batched_support))
    result.square().sum().backward()
    expected = torch.stack([individual(x[row], individual.encode_key(individual_support[row]),
                                       individual.encode_value(individual_support[row])) for row in range(2)])
    expected.square().sum().backward()
    for (name, actual), (expected_name, reference) in zip(batched.named_parameters(), individual.named_parameters()):
        assert name == expected_name and actual.grad.abs().sum() > 0
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(batched_support.grad, individual_support.grad, atol=1e-5, rtol=1e-4)


def test_empty_and_small_independent_banks_preserve_shapes_and_dtype():
    module = make_module(top_k=4)
    x = torch.randn(2, 3, 5, dtype=torch.bfloat16)
    residual, info = module(x, torch.empty(2, 0, 3), torch.empty(2, 0, 3), return_info=True)
    assert residual.dtype == x.dtype and residual.shape == (2, 3, 5)
    assert residual.count_nonzero() == 0
    assert info["indices"].shape == (2, 3, 0) and info["indices"].dtype == torch.long
    _, info = module(x, torch.randn(2, 1, 3), torch.randn(2, 1, 3), return_info=True)
    assert info["indices"].shape == (2, 3, 1)
    assert info["weights"].eq(1).all()


def test_counterfactual_forward_capture_and_teacher_scope_keep_independent_banks():
    backend = make_backend()
    keys, values = backend.vector_keys, backend.vector_values
    questions, tokens = ["q", "longer"], [[3, 1], [4, 5, 1]]
    student = backend.forward_sequences(questions, tokens)
    assert student["query_inputs"].shape == (5, 5)
    assert backend.last_retrieval["indices"].shape[0] == 2
    diagnostics = backend.last_retrieval
    teacher = backend.forward_sequences(questions, tokens, ["observation A", "observation B"], teacher=True)
    assert backend.vector_keys is keys and backend.vector_values is values
    assert backend.last_retrieval is diagnostics
    assert backend.mode == "vector_vera" and not teacher["logits"].requires_grad
    assert torch.equal(student["counts"], teacher["counts"])
    distillation_loss(student["logits"], teacher["logits"]).backward()
    for name, parameter in backend.vector_vera.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert all(parameter.grad is None for parameter in backend.model.parameters())


def test_independent_sampling_restores_bank_identity_on_eos_and_counts_sequential_cost():
    backend = make_backend()
    backend.model.lm_head = FixedHead(1)
    keys, values = backend.vector_keys, backend.vector_values
    questions = ["q", "longer"]
    assert backend.sample_student(questions, max_new_tokens=4) == [[1], [1]]
    assert backend.vector_keys is keys and backend.vector_values is values
    assert len(backend.model.model.calls) == 2
    cost = backend.last_sampling_cost
    assert cost["mode"] == "independent_banks_sequential"
    assert cost["sampled_tokens"] == cost["forward_calls"] == 2
    assert cost["unpadded_input_tokens"] == sum(len(backend.prompt_ids(q)) for q in questions)
    assert cost["padded_input_tokens"] == cost["unpadded_input_tokens"]


def test_independent_sampling_matches_manual_row_banks_and_preserves_truncation():
    backend = make_backend()
    keys, values = backend.vector_keys, backend.vector_values
    questions = ["q", "longer"]
    manual = []
    for row, question in enumerate(questions):
        backend.vector_keys, backend.vector_values = keys[row], values[row]
        manual.extend(ContextDistillationBackend.sample_student(backend, [question], max_new_tokens=4, temperature=0.))
    backend.vector_keys, backend.vector_values = keys, values
    assert backend.sample_student(questions, max_new_tokens=4, temperature=0.) == manual
    backend.model.lm_head = FixedHead(7)
    assert backend.sample_student(questions, max_new_tokens=3) == [[7, 7, 7], [7, 7, 7]]
    assert backend.vector_keys is keys and backend.vector_values is values
    assert backend.last_sampling_cost["forward_calls"] == 6


def test_independent_sampling_failure_restores_banks_even_after_one_completed_row():
    backend = make_backend()
    backend.model.lm_head = FixedHead(1)
    keys, values = backend.vector_keys, backend.vector_values
    original_prompt = backend.prompt_ids
    def failing_prompt(question, context=None):
        if question == "bad":
            raise RuntimeError("intentional second-row failure")
        return original_prompt(question, context)
    backend.prompt_ids = failing_prompt
    with pytest.raises(RuntimeError, match="second-row failure"):
        backend.sample_student(["q", "bad"])
    assert backend.vector_keys is keys and backend.vector_values is values
    assert len(backend.model.model.calls) == 1
    assert backend.last_sampling_cost is None
    assert backend._answer_capture_positions is None


def test_shared_bank_sampling_retains_parent_rng_behavior_and_records_padding_cost():
    backend = make_backend()
    backend.vector_keys, backend.vector_values = backend.vector_keys[0], backend.vector_values[0]
    questions = ["q", "longer"]
    expected = ContextDistillationBackend.sample_student(
        backend, questions, generator=torch.Generator().manual_seed(41))
    actual = backend.sample_student(questions, generator=torch.Generator().manual_seed(41))
    assert actual == expected
    assert backend.last_sampling_cost["mode"] == "shared_bank_active_batch"
    assert backend.last_sampling_cost["forward_calls"] == max(map(len, actual))
    assert backend.last_sampling_cost["padded_input_tokens"] >= backend.last_sampling_cost["unpadded_input_tokens"]


@pytest.mark.parametrize("xshape,kshape,vshape", [
    ((2, 5), (2, 3, 3), (2, 3, 3)),
    ((2, 4, 5), (3, 3, 3), (3, 3, 3)),
    ((2, 4, 5), (2, 3, 4), (2, 3, 3)),
    ((2, 4, 5), (2, 3, 3), (2, 4, 3)),
    ((2, 4, 5), (2, 3, 3), (2, 3, 4)),
])
def test_independent_shape_validation_prevents_accidental_broadcast(xshape, kshape, vshape):
    with pytest.raises(ValueError):
        make_module()(torch.randn(xshape), torch.randn(kshape), torch.randn(vshape))


def test_sampling_rejects_question_bank_mismatch_before_changing_references():
    backend = make_backend()
    keys, values = backend.vector_keys, backend.vector_values
    with pytest.raises(ValueError, match="batches must match"):
        backend.sample_student(["only one"])
    assert backend.vector_keys is keys and backend.vector_values is values
