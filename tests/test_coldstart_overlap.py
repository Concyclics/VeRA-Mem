"""Targeted raw/cache repair must neither reselect data nor touch other features."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/repair_coldstart_overlap.py"
spec = importlib.util.spec_from_file_location("repair_overlap", PATH)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def row(split="train", target=False, matched=False):
    entity = "entity-" + split
    a = "Ship launches Ship" if target else "Alpha " + split + " words"
    b = "Hubel described suppression" if target else "Beta " + split + " words"
    anchor = "References Sources 1916 Ship launches" if target else "A unique preceding natural text"
    passage = anchor + " " + a + (" launches" if target and not matched else "")
    supports_a = [f"Note {entity}.\n{passage}", f"BEGIN {entity}\n{passage}\nEND"]
    supports_b = [repair.replace_target_after_anchor(s, anchor, a, b) for s in supports_a]
    if target and not matched:
        supports_b = [s.replace(a, b, 1) for s in supports_a]
    return dict(id=repair.TARGET_ID if target else "id-" + split, entity=entity, relation="continuation",
        split=split, a=a, b=b, questions=["Question one", "Question two"],
        supports=[supports_a, supports_b], provenance=dict(anchor=anchor))


def test_raw_repair_changes_only_target_b_and_preserves_original(tmp_path, monkeypatch):
    source = tmp_path / "v1"; source.mkdir()
    manifest = dict(complete=True, files={})
    before = {}
    for domain in ("wikipedia", "matched", "synthetic"):
        (source / domain).mkdir(); manifest["files"][domain] = {}
        for split in ("train", "dev", "confirm"):
            r = row(split, target=split == "train" and domain != "synthetic", matched=domain == "matched")
            path = source / domain / (split + ".jsonl")
            path.write_text(json.dumps(r) + "\n")
            before[domain, split] = path.read_bytes()
            manifest["files"][domain][split] = dict(records=1, bytes=path.stat().st_size, sha256=repair.sha256(path))
    for name in ("source_README.md", "hub_convert_revision.json", "hub_source_revision.json"):
        (source / name).write_text("unchanged source attribution")
    repair.write_json(source / "manifest.json", manifest)
    monkeypatch.setattr(repair, "PARENT_MANIFEST_SHA", repair.sha256(source / "manifest.json"))
    monkeypatch.setattr(repair, "TARGET_INDEX", 0)
    result = repair.prepare_data(source, tmp_path / "v2")
    assert result["complete"] and len(result["repairs"]) == 1
    assert result["repairs"][0]["first_1024_update_target_steps"] == [940]
    for (domain, split), old in before.items():
        assert (source / domain / (split + ".jsonl")).read_bytes() == old
        new = (tmp_path / "v2" / domain / (split + ".jsonl")).read_bytes()
        if (domain, split) != ("wikipedia", "train"):
            assert old == new
        else:
            assert json.loads(new) == repair.corrected_record(json.loads(old))
    with pytest.raises(FileExistsError):
        repair.prepare_data(source, tmp_path / "v2")


def packet(monkeypatch):
    torch = pytest.importorskip("torch")
    from vera_mem.context_distillation_run import REVISION
    monkeypatch.setattr(repair, "TARGET_INDEX", 1)
    target = row(target=True)
    target["questions"] = target["questions"][:1]
    target["supports"] = [s[:1] for s in target["supports"]]
    target.update(a_tokens=[1, 2, 3], b_tokens=[4, 5, 6])
    rows = [row("first"), target, row("last")]
    return dict(protocol="dictionary-coldstart-v1", task="wikipedia", split="train", model_revision=REVISION,
        rows=rows, token_repairs=[dict(id="other", old_b="unchanged", new_b="unchanged too")],
        qids=["canonical"], sids=["canonical"], q=torch.arange(12).reshape(3, 1, 4).half(),
        last=torch.arange(24).reshape(3, 2, 1, 4).half(), pool=torch.arange(24).reshape(3, 2, 1, 4).half() + 1)


def test_feature_repair_encodes_only_b_canonical_and_hashes_every_other_element(monkeypatch):
    torch = pytest.importorskip("torch")
    source = packet(monkeypatch); original = copy.deepcopy(source)
    replacement = repair.corrected_record(row(target=True)); calls = []
    def encode(supports):
        calls.append(supports)
        return torch.full((1, 4), 41.), torch.full((1, 4), 42.)
    out, proof = repair.patch_packet(source, replacement, encode)
    assert calls == [[replacement["supports"][1][0]]]
    assert proof["reencoded_supports"] == 1
    assert proof["protected_tensors_bitwise_unchanged"] and proof["token_repairs_unchanged"]
    assert proof["protected_tensors_before"] == proof["protected_tensors_after"]
    assert torch.equal(out["q"], original["q"])
    for name in ("last", "pool"):
        changed = (out[name] != original[name]).any(-1)
        expected = torch.zeros((3, 2, 1), dtype=torch.bool); expected[1, 1, 0] = True
        assert torch.equal(changed, expected)
    assert out["rows"][0] == original["rows"][0] and out["rows"][2] == original["rows"][2]
    assert source["rows"] == original["rows"]  # only the output metadata is replaced
    assert out["rows"][1]["supports"][0] == original["rows"][1]["supports"][0]


@pytest.mark.parametrize("bad", ["wrong_split", "token_repaired_target", "wrong_raw", "heldout", "already_fixed"])
def test_feature_repair_rejects_scope_drift_before_encoder(monkeypatch, bad):
    source = packet(monkeypatch); replacement = repair.corrected_record(row(target=True)); calls = []
    if bad == "wrong_split": source["split"] = "dev"
    if bad == "token_repaired_target": source["rows"][1]["provenance"]["token_repair"] = {"id": repair.TARGET_ID}
    if bad == "wrong_raw": replacement["questions"][0] = "changed query"
    if bad == "heldout": source["sids"] = ["canonical", "heldout"]
    if bad == "already_fixed": source["rows"][1] = repair.corrected_record(source["rows"][1])
    with pytest.raises(ValueError): repair.patch_packet(source, replacement, lambda supports: calls.append(supports))
    assert calls == []


def test_invalid_encoder_shape_or_nonfinite_features_never_mutates_packet(monkeypatch):
    torch = pytest.importorskip("torch")
    for features in [(torch.ones(2, 4), torch.ones(1, 4)), (torch.ones(1, 4), torch.full((1, 4), float("nan")))]:
        source = packet(monkeypatch); before = copy.deepcopy(source)
        with pytest.raises(ValueError):
            repair.patch_packet(source, repair.corrected_record(row(target=True)), lambda supports: features)
        assert torch.equal(source["last"], before["last"]) and torch.equal(source["pool"], before["pool"])


def test_protected_digest_detects_mutation_outside_approved_b_slot(monkeypatch):
    source = packet(monkeypatch)
    before = repair.protected_digests(source, 1)
    source["last"][1, 1, 0, 0] += 1
    assert repair.protected_digests(source, 1) == before
    source["pool"][2, 1, 0, 0] += 1
    assert repair.protected_digests(source, 1) != before
