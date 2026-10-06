"""Regression tests for the historical observation-writer input contract."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from vera_mem import augmentation_data as historical
from vera_mem import counterfactual_data as data


# Deliberately independent of the implementation constant: dropping/changing
# the wrapper in production must not change the expected historical inputs.
HISTORICAL_PREFIX = "Remember this information: "
WIDTH = 9728


@pytest.fixture(scope="module")
def prepare():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_counterfactual.py"
    spec = importlib.util.spec_from_file_location("prepare_counterfactual", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def packet():
    return data.datasets(train_size=32)


def feature(text):
    # Text-sensitive deterministic vectors distinguish raw from wrapped facts,
    # every entity, value and style. No GPU/model download or RNG is involved.
    values = torch.tensor(list(hashlib.sha256(text.encode()).digest()), dtype=torch.float32)
    return (values - 127.5).repeat(WIDTH // len(values))


def original_cache(packet, prepare):
    pairs = packet["train"]
    styles = packet["template_ids"]["train"]["s"]
    supports = torch.stack([torch.stack([
        feature(HISTORICAL_PREFIX + data.render_support(pair, style)) for style in styles
    ]) for pair in pairs])
    return dict(protocol="generalization-v1-layer20", model_revision=prepare.REVISION,
                data_fingerprint=historical.protocol_fingerprint(),
                template_ids={"train": packet["template_ids"]["train"]},
                examples={"train": [asdict(pair.a) for pair in pairs]},
                features={"train": {"s": supports, "q": torch.ones(len(pairs), 8, WIDTH)}})


class FeatureBackend:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.teacher_contexts = []
        self.tokenizer = self

    def encode(self, word, add_special_tokens=False):
        index = data.MEMORY_WORDS.index(word)
        return [100 + index] + [200 + index] * (data.ANSWER_TOKEN_LENGTHS[word] - 1)

    def layer_features(self, texts, batch_size):
        assert batch_size == 32
        self.calls.append(list(texts))
        return torch.stack([feature(text) for text in texts])

    def prompt_ids(self, question, context=None):
        if context is not None:
            self.teacher_contexts.append(context)
        # Only the common prompt-length validation/hash is needed in these
        # preprocessing tests; real tokenizer validation belongs to GPU preflight.
        return list(hashlib.sha256((question + (context or "")).encode()).digest())


def test_prefix_check_reproduces_all_four_historical_training_styles(prepare, packet):
    old = original_cache(packet, prepare)
    backend = FeatureBackend()
    check = prepare.historical_prefix_feature_check(backend, packet, old)
    expected = [HISTORICAL_PREFIX + data.render_support(pair, style)
                for pair in packet["train"][:16] for style in packet["template_ids"]["train"]["s"]]
    assert backend.calls == [expected]
    assert check["passed"] and check["feature_pairs"] == 64
    assert check["split"] == "historical_train_only"
    assert check["writer_input_prefix"] == HISTORICAL_PREFIX
    assert check["max_relative_rmse"] == 0
    assert len(check["per_feature"]) == 64
    assert {row["id"] for row in check["per_feature"]} == {pair.id for pair in packet["train"][:16]}
    assert len(check["writer_text_sha256"]) == len(check["writer_prompt_token_ids_sha256"]) == 64


def test_missing_prefix_is_detected_independently_of_reference_generation(prepare, packet, monkeypatch):
    old = original_cache(packet, prepare)
    # Reproduce the actual v1 defect: encode raw observations despite the
    # inherited training cache using the historical instruction.
    monkeypatch.setattr(prepare, "writer_features",
                        lambda backend, supports, batch_size=32: backend.layer_features(supports, batch_size))
    check = prepare.historical_prefix_feature_check(FeatureBackend(), packet, old)
    assert not check["passed"]
    assert check["failed_feature_pairs"] == 64
    assert check["max_relative_rmse"] > check["max_relative_rmse_allowed"]


@pytest.mark.parametrize("scale,passes", [(1.001, True), (1.10, False)])
def test_prefix_check_tolerates_small_roundoff_but_rejects_same_direction_scale_error(prepare, packet, scale, passes):
    old = original_cache(packet, prepare)
    backend = FeatureBackend()
    raw = backend.layer_features
    backend.layer_features = lambda texts, batch_size: raw(texts, batch_size) * scale
    check = prepare.historical_prefix_feature_check(backend, packet, old)
    assert check["passed"] is passes
    assert check["min_cosine"] > .999
    assert check["max_relative_rmse"] == pytest.approx(abs(scale - 1), abs=1e-5)


def setup_main(tmp_path, monkeypatch, prepare, packet):
    old = original_cache(packet, prepare)
    cache = tmp_path / "historical.pt"
    torch.save(old, cache)
    model = tmp_path / "models/model"
    model.mkdir(parents=True)
    (model.parent / "manifest.json").write_text(json.dumps({"revision": prepare.REVISION}))
    backend = FeatureBackend()
    monkeypatch.setattr(prepare, "ContextDistillationBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(prepare.data, "datasets", lambda **kwargs: packet)
    monkeypatch.setattr(prepare.data, "protocol_fingerprint", lambda *args: "test-protocol-fingerprint")
    monkeypatch.setattr(prepare.data, "protocol_manifest", lambda *args: {"test_fixture": True})
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare_counterfactual.py", "--model", str(model),
                                    "--original-cache", str(cache), "--output", str(output)])
    return backend, output, cache, old


@pytest.mark.parametrize("smoke", [False, True])
def test_main_wraps_alternative_and_every_confirmation_world_but_not_teacher_context(
        prepare, packet, tmp_path, monkeypatch, smoke):
    backend, output, _, old = setup_main(tmp_path, monkeypatch, prepare, packet)
    if smoke:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--smoke"])
    prepare.main()
    train = torch.load(output / "train.pt", weights_only=True)
    evaluation = torch.load(output / "confirmation.pt", weights_only=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["historical_prefix_feature_check"]["passed"]
    for cached in (train, evaluation, manifest):
        assert cached["writer_input_prefix"] == HISTORICAL_PREFIX
        assert cached["historical_prefix_feature_check"]["feature_pairs"] == 64
    # A remains the original cached feature; B changes the fact, not the writer
    # instruction. P selects the same properly wrapped A/style feature axis.
    torch.testing.assert_close(train["s"][:, 0], old["features"]["train"]["s"])
    expected_b_inputs = [HISTORICAL_PREFIX + data.render_support(pair, style, world="B")
                         for pair in packet["train"] for style in train["template_ids"]["s"]]
    assert backend.calls[1] == expected_b_inputs
    for i, pair in enumerate(packet["train"]):
        for j, style in enumerate(train["template_ids"]["s"]):
            torch.testing.assert_close(train["s"][i, 1, j],
                                       feature(HISTORICAL_PREFIX + data.render_support(pair, style, world="B")))
    expected_confirmation_inputs = []
    raw_contexts = set()
    for phase in evaluation["phases"].values():
        for i, case in enumerate(phase["cases"]):
            assert not case["question"].startswith(HISTORICAL_PREFIX)
            for j, name in enumerate(("support_a", "support_b", "support_p")):
                raw = case[name]
                raw_contexts.add(raw)
                assert not raw.startswith(HISTORICAL_PREFIX)
                expected_confirmation_inputs.append(HISTORICAL_PREFIX + raw)
                torch.testing.assert_close(phase["supports"][i, j], feature(HISTORICAL_PREFIX + raw))
    assert backend.calls[2] == list(dict.fromkeys(expected_confirmation_inputs))
    assert all(text.startswith(HISTORICAL_PREFIX) and not text.startswith(" ")
               for call in backend.calls for text in call)
    assert backend.teacher_contexts and all(not text.startswith(HISTORICAL_PREFIX)
                                           for text in backend.teacher_contexts)
    assert raw_contexts.issuperset(backend.teacher_contexts[-2 * sum(len(p["cases"]) for p in evaluation["phases"].values()):])


def test_main_persists_failed_preflight_and_refuses_to_create_either_cache(
        prepare, packet, tmp_path, monkeypatch):
    backend, output, cache, old = setup_main(tmp_path, monkeypatch, prepare, packet)
    # A cache accidentally computed without the historical instruction must
    # also fail: neither the script's prefix nor provenance string alone suffices.
    old["features"]["train"]["s"] = torch.stack([torch.stack([
        feature(data.render_support(pair, style)) for style in packet["template_ids"]["train"]["s"]
    ]) for pair in packet["train"]])
    torch.save(old, cache)
    with pytest.raises(ValueError, match="Historical writer prefix feature check failed"):
        prepare.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["complete"]
    assert not manifest["historical_prefix_feature_check"]["passed"]
    assert manifest["historical_prefix_feature_check"]["failed_feature_pairs"] == 64
    assert not (output / "train.pt").exists() and not (output / "confirmation.pt").exists()
    assert len(backend.calls) == 1 and not backend.teacher_contexts
