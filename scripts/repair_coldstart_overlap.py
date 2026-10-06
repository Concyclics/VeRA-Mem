"""Versioned repair of the single audited overlapping Wikipedia cloze span.

Create raw v2 on CPU with --source-data OLD --prepare-data NEW. Re-encode only
the affected training B support with --cache OLD/train.pt --data-dir NEW
--model PINNED_MODEL --output NEW_FEATURE_DIR. Originals are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vera_mem.coldstart_data import SPLITS, digest, replace_target_after_anchor, validate_records

TARGET_ID = "coldwiki-6184b88599664708a482acba"
TARGET_INDEX = 13896
PARENT_MANIFEST_SHA = "79a01d251cd492f64b3cdf87505aafc81cf69fe743eb1aa440bb1680e5520a76"
REPAIR_PROTOCOL = "coldstart-anchor-overlap-repair-v2"
REASON = ("The first literal occurrence of 'Ship launches Ship' overlaps the left anchor. "
          "Replace the exact span immediately following the unique anchor; keep all "
          "identities, answers, questions, A supports and other records unchanged.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def corrected_record(row):
    """Correct only B supports, without reselecting or retokenizing any data."""
    out = copy.deepcopy(row)
    anchor = row["provenance"]["anchor"]
    out["supports"][1] = [replace_target_after_anchor(s, anchor, row["a"], row["b"])
                           for s in row["supports"][0]]
    return out


def prepare_data(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent_sha = sha256(source / "manifest.json")
    if parent_sha != PARENT_MANIFEST_SHA:
        raise ValueError("Repair requires the audited immutable v1 manifest")
    parent = json.loads((source / "manifest.json").read_text())
    if not parent.get("complete"):
        raise ValueError("Incomplete source data")
    output.mkdir(parents=True)
    repaired, checked = [], {}
    manifest = copy.deepcopy(parent)
    manifest.update(complete=False, data_revision="v2", repair_protocol=REPAIR_PROTOCOL,
                    parent_manifest_sha256=parent_sha, parent_data_directory=str(source),
                    repair_started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "manifest.json", manifest)
    for domain in ("wikipedia", "matched", "synthetic"):
        (output / domain).mkdir()
        checked[domain] = {}
        all_rows = {}
        for split in SPLITS:
            old, new = source / domain / (split + ".jsonl"), output / domain / (split + ".jsonl")
            if sha256(old) != parent["files"][domain][split]["sha256"]:
                raise ValueError(f"Source checksum mismatch: {old}")
            lines = old.read_text().splitlines(keepends=True)
            rows = [json.loads(line) for line in lines]
            for index, row in enumerate(rows):
                if domain not in {"wikipedia", "matched"}:
                    continue
                fixed = corrected_record(row)
                if fixed != row:
                    if (domain, split, index, row["id"]) != ("wikipedia", "train", TARGET_INDEX, TARGET_ID):
                        raise ValueError(f"Unexpected correction scope: {domain}/{split}/{index}/{row['id']}")
                    if "token_repair" in row["provenance"]:
                        raise ValueError("Unexpected token repair on audited record")
                    repaired.append(dict(domain=domain, split=split, index=index, id=row["id"],
                        changed_fields=["supports[1][0]", "supports[1][1]"],
                        row_before_sha256=digest(row), row_after_sha256=digest(fixed),
                        supports_b_before=row["supports"][1], supports_b_after=fixed["supports"][1],
                        first_1024_update_target_steps=[940], seed=63042, batch_size=8,
                        statistics_first8192_affected=False, foundation_a_initialization_affected=False))
                    rows[index] = fixed
                    lines[index] = json.dumps(fixed, ensure_ascii=False, separators=(",", ":")) + "\n"
            new.write_text("".join(lines))
            all_rows[split] = rows
            checked[domain][split] = dict(records=len(rows), sha256=sha256(new),
                unchanged_bytes=sha256(new) == parent["files"][domain][split]["sha256"])
            manifest["files"][domain][split].update(bytes=new.stat().st_size, sha256=sha256(new))
        validate_records(all_rows)
    if len(repaired) != 1:
        raise AssertionError(f"Expected exactly one corrected record, got {len(repaired)}")
    for name in ("source_README.md", "hub_convert_revision.json", "hub_source_revision.json"):
        shutil.copy2(source / name, output / name)
    manifest.update(complete=True, repair_reason=REASON, repairs=repaired, repair_validation=checked,
        repair_finished_at=datetime.now(timezone.utc).isoformat(),
        repair_source_code_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
            for p in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src/vera_mem/coldstart_data.py")},
        source_shards_policy="Immutable parent shard paths/hashes retained; raw source is not duplicated.")
    write_json(output / "manifest.json", manifest)
    return manifest


def tensor_digest(tensor):
    import torch
    h = hashlib.sha256()
    h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    for chunk in tensor.detach().cpu().split(256):
        h.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def protected_digests(packet, index):
    """Hash every tensor element except exactly last/pool[index, B, view0]."""
    import torch
    result = {}
    for name, value in packet.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name in {"last", "pool"}:
            result[name] = dict(before=tensor_digest(value[:index]), after=tensor_digest(value[index + 1:]),
                                target_a=tensor_digest(value[index, 0]))
        else:
            result[name] = tensor_digest(value)
    return result


def patch_packet(packet, replacement, encode):
    """Mutate only permitted tensor slots in a newly loaded in-memory packet.

    Metadata is copied. Callers must not reuse this in-memory packet as an old
    reference; the original cache file itself is never opened for writing.
    """
    import torch
    from vera_mem.context_distillation_run import REVISION
    if (packet.get("protocol"), packet.get("task"), packet.get("split"), packet.get("model_revision")) != (
            "dictionary-coldstart-v1", "wikipedia", "train", REVISION):
        raise ValueError("Expected pinned Wikipedia training feature cache")
    if packet.get("qids") != ["canonical"] or packet.get("sids") != ["canonical"]:
        raise ValueError("Repair only supports canonical single-view training cache")
    indices = [i for i, row in enumerate(packet["rows"]) if row["id"] == TARGET_ID]
    if indices != [TARGET_INDEX]:
        raise ValueError("Unexpected target ID/index")
    index = indices[0]
    old = packet["rows"][index]
    if old.get("provenance", {}).get("token_repair") or any(r["id"] == TARGET_ID for r in packet["token_repairs"]):
        raise ValueError("Audited target must have no tokenizer repair")
    expected = corrected_record(old)
    reference = copy.deepcopy(replacement)
    reference["questions"] = reference["questions"][:1]
    reference["supports"] = [world[:1] for world in reference["supports"]]
    reference.update({key: old[key] for key in ("a_tokens", "b_tokens")})
    if expected != reference or old == expected:
        raise ValueError("Raw v2 target is inconsistent or feature cache is already repaired")
    count = len(packet["rows"])
    for name in ("last", "pool"):
        if packet[name].ndim != 4 or packet[name].shape[:3] != (count, 2, 1):
            raise ValueError(f"Unexpected {name} tensor shape")
    if packet["q"].shape[:2] != (count, 1):
        raise ValueError("Unexpected q tensor shape")
    before = protected_digests(packet, index)
    target_before = {name: tensor_digest(packet[name][index, 1, 0]) for name in ("last", "pool")}
    last, pool = encode([expected["supports"][1][0]])
    for name, feature in (("last", last), ("pool", pool)):
        if feature.shape != (1, packet[name].shape[-1]) or not bool(torch.isfinite(feature).all()):
            raise ValueError(f"Invalid newly encoded {name}")
    out = dict(packet)
    out["rows"] = list(packet["rows"])
    out["rows"][index] = expected
    for name, feature in (("last", last), ("pool", pool)):
        out[name][index, 1, 0].copy_(feature[0].to(out[name]))
    after = protected_digests(out, index)
    if before != after:
        raise AssertionError("An unapproved feature tensor changed")
    proof = dict(id=TARGET_ID, index=index, support_text=expected["supports"][1][0],
        reencoded_supports=1, tensor_slots=["last[13896,1,0,:]", "pool[13896,1,0,:]"],
        protected_tensors_before=before, protected_tensors_after=after,
        protected_tensors_bitwise_unchanged=True, target_before=target_before,
        target_after={name: tensor_digest(out[name][index, 1, 0]) for name in ("last", "pool")},
        token_repairs_unchanged=out["token_repairs"] == packet["token_repairs"],
        only_target_row_changed=all(a == b for i, (a, b) in enumerate(zip(packet["rows"], out["rows"])) if i != index))
    return out, proof


def repair_features(cache, data_dir, model, output):
    import torch
    from vera_mem.context_distillation_run import REVISION, parameter_digest
    from vera_mem.counterfactual_backend import CounterfactualBackend
    from vera_mem.interface_features import support_features
    cache, data_dir, model, output = map(Path, (cache, data_dir, model, output))
    if output.exists():
        raise FileExistsError(output)
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    if not data_manifest.get("complete") or data_manifest.get("repair_protocol") != REPAIR_PROTOCOL:
        raise ValueError("Expected completed v2 repair data manifest")
    if data_manifest["parent_manifest_sha256"] != PARENT_MANIFEST_SHA or len(data_manifest["repairs"]) != 1:
        raise ValueError("Unexpected repair lineage")
    if json.loads((model.parent / "manifest.json").read_text())["revision"] != REVISION:
        raise ValueError("Model revision mismatch")
    source_jsonl = data_dir / "wikipedia/train.jsonl"
    if sha256(source_jsonl) != data_manifest["files"]["wikipedia"]["train"]["sha256"]:
        raise ValueError("v2 raw train hash mismatch")
    with source_jsonl.open() as handle:
        replacement = next(json.loads(line) for index, line in enumerate(handle) if index == TARGET_INDEX)
    if digest(replacement) != data_manifest["repairs"][0]["row_after_sha256"]:
        raise ValueError("v2 repaired row hash mismatch")
    packet = torch.load(cache, map_location="cpu", weights_only=True)
    if packet["data_manifest_sha256"] != PARENT_MANIFEST_SHA:
        raise ValueError("Cache is not derived from audited v1 data")
    if packet["data_jsonl_sha256"] != data_manifest["repair_validation"]["wikipedia"]["train"].get("parent_sha256", "d15f0c0fb967ccf45e895e9621de7c0d964066f4b6419606522760c65dad171c"):
        raise ValueError("Unexpected parent train JSONL")
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = dict(protocol="dictionary-coldstart-v1", repair_protocol=REPAIR_PROTOCOL, stage="repair", complete=False,
        started_at=datetime.now(timezone.utc).isoformat(), configuration=dict(cache=str(cache), data_dir=str(data_dir),
        model=str(model), output=str(output)), cache_sha256=sha256(cache), model_revision=REVISION,
        data_manifest_sha256=sha256(data_dir / "manifest.json"), reason=REASON,
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p) for p in
                 [Path(__file__).resolve(), *sorted((Path(__file__).resolve().parents[1] / "src/vera_mem").glob("*.py"))]})
    write_json(output / "manifest.json", manifest)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(63042)
        backend = CounterfactualBackend(str(model), layer=20)
        model_before = parameter_digest(backend.model)
        def encode(supports):
            row = packet["rows"][TARGET_INDEX]
            for key in ("a", "b"):
                if backend.tokenizer.encode(row[key], add_special_tokens=False) != row[key + "_tokens"]:
                    raise ValueError("Pinned tokenizer target tokens changed")
            return support_features(backend, supports, batch_size=1)
        out, proof = patch_packet(packet, replacement, encode)
        if model_before != parameter_digest(backend.model) or any(p.grad is not None for p in backend.model.parameters()):
            raise AssertionError("Backbone changed")
        out["data_manifest_sha256"] = manifest["data_manifest_sha256"]
        out["data_jsonl_sha256"] = sha256(source_jsonl)
        out["feature_repair"] = dict(protocol=REPAIR_PROTOCOL, parent_cache_sha256=manifest["cache_sha256"],
            parent_data_manifest_sha256=PARENT_MANIFEST_SHA, id=TARGET_ID, index=TARGET_INDEX,
            reason=REASON, reencoded_supports=1, first_1024_update_target_steps=[940])
        torch.save(out, output / "train.pt")
        write_json(output / "train_repairs.json", out["token_repairs"])
        write_json(output / "preservation.json", proof)
        manifest.update(complete=True, backbone_unchanged=True, result=proof,
            output_train_sha256=sha256(output / "train.pt"), finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest["error"] = repr(error)
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--prepare-data", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source_data or args.prepare_data:
        if not (args.source_data and args.prepare_data) or any((args.cache, args.data_dir, args.model, args.output)):
            parser.error("Raw preparation requires only --source-data and --prepare-data")
        result = prepare_data(args.source_data, args.prepare_data)
        print(json.dumps(dict(complete=result["complete"], output=str(args.prepare_data),
                              manifest_sha256=sha256(args.prepare_data / "manifest.json"), repairs=result["repairs"])))
    else:
        if not all((args.cache, args.data_dir, args.model, args.output)):
            parser.error("Feature repair requires --cache --data-dir --model --output")
        result = repair_features(args.cache, args.data_dir, args.model, args.output)
        print(json.dumps(dict(complete=result["complete"], output=str(args.output),
                              output_train_sha256=result["output_train_sha256"], elapsed_seconds=result["elapsed_seconds"])))


if __name__ == "__main__":
    main()
