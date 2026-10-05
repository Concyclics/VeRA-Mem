"""Value diagnostics must not apply tanh derivatives to RMS-normalized vectors."""

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_vectors.py"
spec = importlib.util.spec_from_file_location("diagnose_vectors_script", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def save_bank(tmp_path, values):
    bank = tmp_path / "bank.pt"
    torch.save({"keys": torch.eye(len(values)), "values": torch.tensor(values)}, bank)
    return bank


def assert_no_tanh_metrics(result):
    assert not ({"saturation_threshold", "saturation_fraction", "fully_saturated_rows",
                 "tanh_derivative_from_output"} & set(result["values"]))


@pytest.mark.parametrize("evidence", ["config", "buffers"])
def test_rms_values_above_one_skip_tanh_metrics(tmp_path, evidence):
    bank = save_bank(tmp_path, [[2., 0., 0., 0.], [0., 0., 0., -2.]])
    parameters = {"Wv.weight": torch.ones(4, 4)}
    config = {}
    if evidence == "config":
        config["variant"] = "stable"
    else:
        parameters.update(query_center=torch.zeros(4), support_center=torch.zeros(4),
                          statistics_fitted=torch.tensor(True))
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config, "module": parameters}, checkpoint)
    before = diagnostics.sha256(bank), diagnostics.sha256(checkpoint)
    result = diagnostics.diagnose(bank, checkpoint, .999, 16)
    assert result["value_activation"] == "rmsnorm"
    assert result["value_activation_provenance"]["source"] == "checkpoint"
    assert_no_tanh_metrics(result)
    assert result["values"]["row_rms"]["mean"] == 1.
    assert result["values"]["elements"]["max"] == 2.
    assert result["values"]["centered_spectrum"]["effective_rank"] == 1.
    assert before == (diagnostics.sha256(bank), diagnostics.sha256(checkpoint))


@pytest.mark.parametrize("legacy", [False, True])
def test_known_tanh_preserves_historical_saturation_fields(tmp_path, legacy):
    bank = save_bank(tmp_path, [[1., -1.], [.5, 0.]])
    config = {"variant": "raw"}
    parameters = {}
    if legacy:
        config = dict(rank=2, key_dim=2, offline_train_size=128,
                      offline_dev_size=32, offline_epochs=3)
        parameters = {name: torch.zeros(2, 2) for name in
                      ("A", "B", "Wq.weight", "Wk.weight", "Wv.weight", "b")}
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config, "module": parameters}, checkpoint)
    result = diagnostics.diagnose(bank, checkpoint, .999, 16)
    assert result["value_activation"] == "tanh"
    assert result["values"]["saturation_fraction"] == .5
    assert result["values"]["fully_saturated_rows"] == 1
    assert result["values"]["tanh_derivative_from_output"]["mean"] == .4375
    assert result["values"]["tanh_derivative_from_output"]["min"] == 0.


def test_unidentified_values_do_not_infer_tanh_from_range(tmp_path):
    bank = save_bank(tmp_path, [[.1, -.2], [.2, .3]])
    result = diagnostics.diagnose(bank, None, .999, 16)
    assert result["value_activation"] == "unknown"
    assert_no_tanh_metrics(result)
    assert result["value_activation_provenance"]["source"] == "unidentified"
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"variant": "unrecognized"}, "module": {}}, checkpoint)
    result = diagnostics.diagnose(bank, checkpoint, .999, 16)
    assert result["value_activation"] == "unknown"
    assert_no_tanh_metrics(result)


def test_explicit_activation_and_invalid_tanh_range(tmp_path):
    bank = save_bank(tmp_path, [[1.5, .1]])
    result = diagnostics.diagnose(bank, None, .999, 16, value_activation="rmsnorm")
    assert result["value_activation"] == "rmsnorm"
    assert result["value_activation_provenance"]["source"] == "explicit_cli_override"
    assert_no_tanh_metrics(result)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        diagnostics.diagnose(bank, None, .999, 16, value_activation="tanh")


def test_conflicting_checkpoint_evidence_is_unknown():
    checkpoint = {"config": {"variant": "raw"}, "module": {
        "query_center": torch.zeros(2), "support_center": torch.zeros(2),
        "statistics_fitted": torch.tensor(True),
    }}
    activation, provenance = diagnostics.resolve_activation(checkpoint, "auto")
    assert activation == "unknown"
    assert provenance["reason"] == "Conflicting activation evidence"
