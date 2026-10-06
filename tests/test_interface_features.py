"""Pooling includes only tokens wholly inside the actual support text."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vera_mem.interface_features import PREFIX, content_prompt, support_features


class OffsetTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        assert messages[1]["content"].startswith(PREFIX)
        return "<sys>" + messages[0]["content"] + "</sys><usr>" + messages[1]["content"] + "</usr><assistant>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        assert add_special_tokens is False and return_offsets_mapping is True
        return {"input_ids": [1] + [ord(c) + 2 for c in text],
                "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))]}


@pytest.mark.parametrize("support", ["fact=orange", "  value: red\n", "事实：蓝色", "PREFIX again: " + PREFIX + "blue"])
def test_content_prompt_mask_selects_raw_support_not_instruction_or_chat_tokens(support):
    tokenizer = OffsetTokenizer()
    ids, mask = content_prompt(tokenizer, support)
    selected = "".join(chr(token - 2) for token, keep in zip(ids, mask) if keep)
    assert selected == support and sum(mask) == len(support)
    assert len(ids) == len(mask) and mask[0] == 0
    assert mask[-1] == 0 and all(keep in (0, 1) for keep in mask)


def test_tokens_crossing_either_content_boundary_and_zero_width_specials_are_excluded():
    class CrossingTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            start = text.index(PREFIX) + len(PREFIX)
            end = start + len("abcdef")
            return {"input_ids": [11, 12, 13, 14, 15, 16],
                    "offset_mapping": [(start - 1, start + 1), (start + 1, start + 3),
                                       (start + 3, end), (end - 1, end + 1), (0, 0), (end, end)]}
    assert content_prompt(CrossingTokenizer(), "abcdef") == ([11, 12, 13, 14, 15, 16], [0, 1, 1, 0, 0, 0])


@pytest.mark.parametrize("support", ["", None, 9])
def test_empty_or_nontext_support_rejected(support):
    with pytest.raises(ValueError, match="Nonempty support"):
        content_prompt(OffsetTokenizer(), support)


def test_no_wholly_contained_content_token_fails_instead_of_pooling_wrapper():
    class NoContentTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}
    with pytest.raises(ValueError, match="Empty content mask"):
        content_prompt(NoContentTokenizer(), "x")


class FakeBackbone(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.calls = 0
    def forward(self, input_ids, attention_mask, use_cache=False):
        assert self.backend.mode == "disabled" and not use_cache and not torch.is_grad_enabled()
        self.calls += 1
        values = input_ids.float()
        features = torch.stack([values, values.square(), torch.ones_like(values)], -1)
        self.backend.target(features)
        return None


class FeatureBackend:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = OffsetTokenizer()
        self.target = nn.Identity()
        self.mode = "memory"
        self.max_input_tokens = 1000
        self.model = SimpleNamespace(model=FakeBackbone(self))
    def prompt_ids(self, text):
        assert text.startswith(PREFIX)
        return content_prompt(self.tokenizer, text[len(PREFIX):])[0]
    def _right_pad(self, sequences):
        # Large pad value catches an accidentally unmasked pooled mean.
        ids = torch.full((len(sequences), max(map(len, sequences))), 9000)
        mask = torch.zeros_like(ids)
        for row, tokens in enumerate(sequences):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
        return ids, mask
    @contextmanager
    def disabled(self):
        before, self.mode = self.mode, "disabled"
        try:
            yield
        finally:
            self.mode = before


def test_support_features_pool_only_content_and_last_is_final_real_prompt_token():
    backend = FeatureBackend()
    supports = ["blue", "a longer amber support", "世界"]
    last, pooled = support_features(backend, supports, batch_size=2)
    assert last.shape == pooled.shape == (3, 3)
    for row, text in enumerate(supports):
        support_ids = torch.tensor([ord(c) + 2 for c in text], dtype=torch.float32)
        expected = torch.tensor([support_ids.mean(), support_ids.square().mean(), 1.])
        torch.testing.assert_close(pooled[row], expected)
        final_id = backend.prompt_ids(PREFIX + text)[-1]
        torch.testing.assert_close(last[row], torch.tensor([float(final_id), float(final_id ** 2), 1.]))
    assert not last.requires_grad and not pooled.requires_grad
    assert backend.mode == "memory" and not backend.target._forward_pre_hooks
    assert backend.model.model.calls == 2


def test_support_feature_extraction_rejects_historical_prompt_divergence_or_truncation():
    backend = FeatureBackend()
    backend.prompt_ids = lambda _: [9]
    with pytest.raises(AssertionError, match="historical backend"):
        support_features(backend, ["fact"])
    assert backend.model.model.calls == 0
    backend = FeatureBackend()
    backend.max_input_tokens = 10
    with pytest.raises(ValueError, match="No silent truncation"):
        support_features(backend, ["fact"])
    assert backend.model.model.calls == 0


def test_extraction_failure_restores_disabled_scope_and_removes_capture_hook():
    backend = FeatureBackend()
    def fail(**kwargs):
        raise RuntimeError("intentional backbone failure")
    backend.model.model.forward = fail
    with pytest.raises(RuntimeError, match="intentional backbone failure"):
        support_features(backend, ["fact"])
    assert backend.mode == "memory" and not backend.target._forward_pre_hooks
