"""Frozen-prefix reuse must match an ordinary hooked decoder forward."""
import pytest
import torch

from vera_mem.corpus_backend import CorpusBackbone, coarse_teacher_loss, answer_batch


def test_decomposition_matches_real_down_projection_hook_and_frozen_gradients():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=3, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=8,
                                      max_position_embeddings=128, use_cache=False)
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    backend = object.__new__(CorpusBackbone)
    backend.model, backend.layer = model, 1
    backend.target = model.model.layers[1].mlp.down_proj
    ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    delta = torch.randn(2, 4, 32, requires_grad=True)
    prefix = backend.prefix(ids, mask)
    actual = backend.suffix(prefix, delta)
    handle = backend.target.register_forward_hook(lambda module, inputs, output: output + delta)
    expected = model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False).last_hidden_state
    handle.remove()
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_contextual_topk_plus_tail_is_exact_coarsened_cross_entropy():
    logits = torch.tensor([[.2, .4, .7, -.5]], requires_grad=True)
    teacher = torch.tensor([[.1, .3, .4, .2]])
    ids = torch.tensor([[1, 2]])
    student = logits.softmax(-1)
    expected = -(.3 * student[0, 1].log() + .4 * student[0, 2].log()
                 + .3 * (student[0, 0] + student[0, 3]).log())
    actual = coarse_teacher_loss(logits, ids, teacher.gather(1, ids)).sum()
    assert torch.allclose(actual, expected)
    assert torch.allclose(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                          torch.autograd.grad(expected, logits)[0])


def test_prefix_reuse_preserves_incremental_kv_cache():
    transformers = pytest.importorskip("transformers")
    from transformers.cache_utils import DynamicCache
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=3, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    backend = object.__new__(CorpusBackbone)
    backend.model, backend.layer, backend.target = model, 1, model.model.layers[1].mlp.down_proj
    cache = DynamicCache(config=config)
    ids = torch.tensor([[1, 2, 3], [1, 5, 6]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        for step in range(4):
            prefix = backend.prefix(ids if step == 0 else ids[:, -1:], mask, past=cache)
            actual = backend.suffix(prefix)[:, -1]
            expected = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
            assert torch.allclose(actual, expected, atol=1e-6)
            ids = torch.cat((ids, torch.full((2, 1), 7 + step)), dim=1)
            mask = torch.ones_like(ids)


def test_teacher_and_student_targets_align_despite_different_prompt_lengths():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def encode(self, text, **kwargs):
            return [ord(c) + 2 for c in text]
        def apply_chat_template(self, messages, **kwargs):
            return self.encode(messages[0]["content"])
    episodes = [dict(title="A", anchor="Eight words", answer="one two three", support="A longer observation"),
                dict(title="B", anchor="Different words", answer="four five six", support="Another observation")]
    student = answer_batch(Tokenizer(), episodes, device="cpu")
    teacher = answer_batch(Tokenizer(), episodes, context=True, device="cpu")
    assert teacher[0].shape[1] > student[0].shape[1]
    assert torch.equal(student[1][student[1] != -100], teacher[1][teacher[1] != -100])
    for ids, labels, mask in (student, teacher):
        assert (labels[mask == 0] == -100).all()
        assert ((labels != -100).sum(1) == torch.tensor([14, 14])).all()
