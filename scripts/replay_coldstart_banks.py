"""Read-only CPU replay of completed local cold-start evaluation banks.

python scripts/replay_coldstart_banks.py --runs-root ../runs/xtrah100 --output ../runs/local/coldstart_bank_replay.json
python scripts/replay_coldstart_banks.py --self-test

Reuses the interface audit for episodic snapshots, single-record B patches,
shuffles and raw read hashes. Adds checkpoint-backed foundation provenance,
compression arithmetic and real CPU-store reconstruction. No model/encoder,
feature cache, SSH or GPU is used. Incomplete evaluations are never inspected.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
import torch
from torch.nn import functional as F

import audit_interface_banks as episodic_audit
import summarize_coldstart as summary
from vera_mem.dictionary_vera import merge_bank
from vera_mem.vector_store import PersistentVectorDB

require, read_json, read_jsonl, file_hash = (
    summary.ia.require, summary.ia.read_json, summary.ia.read_jsonl, summary.ia.file_hash)
PROTOCOL = "coldstart-cpu-bank-replay-v1"
COMPRESSION_SEED = 92043


def experiment_scope(arm):
    """Label evidence without imposing a fixed arm count on this tensor audit."""
    if arm in {"wiki_teacher_warm", "wiki_teacher_repair"}:
        return "supplementary_teacher_feasibility"
    if arm in {"no_warm", "matched_warm", "wiki_warm", "matched_warm_transfer", "wiki_warm_transfer",
               "fixed", "learned_sparse", "dense_warm", "straight_through", "wiki_joint_warm", "wiki_joint"}:
        return "original_experiment"
    return "other_declared_run"


def scope_summary(records):
    scopes = defaultdict(lambda: dict(record_indices=[], run_dirs=[]))
    for index, record in enumerate(records):
        group = scopes[record["experiment_scope"]]
        group["record_indices"].append(index)
        group["run_dirs"].append(record["run_dir"])
    return {name: dict(value, observed_count=len(value["record_indices"])) for name, value in scopes.items()}


def analysis_exclusions(directory, runs_root):
    """Read ancestor exclusion markers only, before opening a run manifest.

    Exclusion affects analysis of that run, not the ability to resolve a
    declared parent checkpoint. Parent provenance is reported separately;
    this helper makes no semantic claim that an excluded source is pure.
    """
    directory, root = Path(directory).resolve(), Path(runs_root).resolve()
    require(directory.is_relative_to(root), "Exclusion lookup outside local runs root")
    markers = []
    current = directory
    while True:
        marker = current / "analysis_excluded.json"
        if marker.exists():
            require(marker.is_file(), "Analysis exclusion marker is not a file")
            metadata = read_json(marker)
            require(isinstance(metadata, dict) and isinstance(metadata.get("reason"), str)
                    and bool(metadata["reason"].strip()), "Invalid analysis exclusion reason")
            require("evidence" in metadata and "superseded_by" in metadata, "Incomplete analysis exclusion provenance")
            markers.append(dict(path=str(marker), sha256=file_hash(marker), metadata=metadata))
        if current == root:
            break
        current = current.parent
    return markers


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu", "Non-CPU/tensor module state")
        digest.update(key.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _close(actual, expected, name, *, exact=False):
    require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape, name + " shape mismatch")
    require(actual.device.type == "cpu" and bool(torch.isfinite(actual).all()), name + " must be finite CPU data")
    if exact:
        require(actual.dtype == expected.dtype and torch.equal(actual, expected), name + " differs")
    else:
        require(torch.allclose(actual.double(), expected.double(), rtol=2e-6, atol=2e-6), name + " differs")


def resolve_checkpoint(directory, manifest, runs_root):
    """Map only a declared remote /runs/... suffix into the local backup root."""
    root = Path(runs_root).resolve()
    raw = manifest.get("configuration", {}).get("checkpoint")
    require(isinstance(raw, str) and raw, "Missing declared source checkpoint")
    source = Path(raw)
    candidates = [source if source.is_absolute() else Path(directory) / source]
    parts = source.parts
    for i, part in enumerate(parts):
        if part == "runs" and i + 1 < len(parts):
            candidates.append(root.joinpath(*parts[i + 1:]))
    for candidate in candidates:
        local = candidate.resolve()
        if local.is_relative_to(root) and local.is_file():
            require(file_hash(local) == manifest.get("checkpoint_sha256"), "Source checkpoint SHA256 mismatch")
            return local
    raise ValueError("Source checkpoint is absent from local runs-root backup: " + raw)


def audit_foundation_tensor(directory, manifest, runs_root):
    directory = Path(directory)
    metrics, c = manifest["result"], manifest["result"]["coldstart"]
    path = directory / c["foundation_snapshot"]
    require(path.parent == directory and path.is_file(), "Unsafe/missing foundation snapshot")
    require(file_hash(path) == c["foundation_snapshot_sha256"], "Foundation snapshot SHA256 mismatch")
    packet = torch.load(path, map_location="cpu", weights_only=True)
    require(packet.get("protocol") == summary.FOUNDATION_PROTOCOL, "Wrong foundation snapshot protocol")
    source = resolve_checkpoint(directory, manifest, runs_root)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    require(checkpoint.get("protocol") == summary.RUN_PROTOCOL, "Wrong source checkpoint protocol")
    state, config = checkpoint["module"], checkpoint["architecture"]
    original_keys, original_values = state["base_keys"], state["base_values"]
    n, dimensions = original_keys.shape
    require(original_values.shape == (n, config["rank"]) and dimensions == config["key_dim"], "Checkpoint bank shape mismatch")
    require(n == config["base_size"] == packet["source_records"] == c["source_records"], "Source prototype count mismatch")
    _close(packet["original_keys"], original_keys, "Checkpoint source keys", exact=True)
    _close(packet["original_values"], original_values, "Checkpoint source values", exact=True)
    checkpoint_digest = tensor_state_digest(state)
    require(checkpoint_digest == packet["original_shared_weights"] == c["original_shared_weights_before"]
            == c["original_shared_weights_after"], "Original module digest differs from checkpoint")
    sparse_state = dict(state, dictionary_routing=torch.tensor(0, dtype=torch.int64))
    require(tensor_state_digest(sparse_state) == metrics["shared_weights_before"] == metrics["shared_weights_after"],
            "Inner sparse-evaluation module digest mismatch")
    require(float(state["dictionary_alpha"]) == config["alpha_base"] == packet["alpha_base"] == c["alpha_base"],
            "Foundation mixing alpha mismatch")
    require(config["base_top_k"] == packet["base_top_k"] == c["base_top_k"], "Foundation quota mismatch")
    require(config["top_k"] == c["episodic_top_k"], "Episodic quota mismatch")
    require(packet["mode"] == c["base_mode"] == manifest["configuration"].get("base_mode", "full"), "Base mode mismatch")
    require(packet["compression_seed"] == c["compression_seed"] == COMPRESSION_SEED, "Unexpected compression seed")
    require(packet["values_rms_recalibrated"] is False and packet["count_bias_applied"] is False,
            "Unregistered value normalization/count bias")
    mode, size = packet["mode"], packet["active_records"]
    require(size == c["active_records"] and type(size) is int and 0 <= size <= n, "Active prototype count mismatch")
    require(packet["keys"].shape == (size, dimensions) and packet["values"].shape == (size, config["rank"]),
            "Selected prototype shape mismatch")
    require(packet["merge_count"] == c["merge_count"], "Compression cardinality differs")
    requested = manifest["configuration"].get("merge_count")
    if mode in {"random256", "cluster256"}:
        require(size == (256 if requested is None else requested), "Compression differs from planned size")
    else:
        require(requested is None and packet["merge_count"] is None, "Unexpected full/off compression size")
    details = dict(mode=mode, source_records=n, active_records=size, source_checkpoint=str(source),
                   source_checkpoint_sha256=manifest["checkpoint_sha256"], checkpoint_state_digest=checkpoint_digest,
                   source_checkpoint_analysis_exclusions=analysis_exclusions(source.parent, runs_root),
                   mean_verification=None, compression_recomputed=False)
    if mode in {"full", "off", "random256"}:
        if mode == "full":
            indices = list(range(n))
        elif mode == "off":
            indices = []
        else:
            generator = torch.Generator(device="cpu").manual_seed(COMPRESSION_SEED)
            indices = sorted(torch.randperm(n, generator=generator)[:size].tolist())
        require(packet["source_indices"] == indices, "Random/full/off source indices differ")
        require(packet["members"] == [[i] for i in indices], "Subset member mapping differs")
        require(packet["ids"] == [f"base-{i}" for i in indices], "Subset IDs differ")
        _close(packet["keys"], original_keys[indices], "Subset keys", exact=True)
        _close(packet["values"], original_values[indices], "Subset values", exact=True)
        require(packet["exact_function_preserving"] == (mode == "full"), "Incorrect compression equivalence claim")
        details["compression_recomputed"] = True
    elif mode == "cluster256":
        require(packet["source_indices"] is None and packet["exact_function_preserving"] is False, "Invalid cluster provenance")
        merged = packet["merge"]
        labels = merged["assignments"]
        require(labels.dtype == torch.int64 and labels.shape == (n,) and bool(((labels >= 0) & (labels < size)).all()),
                "Invalid cluster assignments")
        counts = torch.bincount(labels, minlength=size)
        require(bool((counts > 0).all()), "Empty foundation cluster")
        _close(merged["counts"], counts, "Cluster counts", exact=True)
        _close(merged["weight_sums"], counts.float(), "Cluster prototype weights", exact=True)
        require(packet["members"] == [(labels == i).nonzero().flatten().tolist() for i in range(size)],
                "Cluster member mapping differs")
        require(packet["ids"] == [f"cluster-{i}" for i in range(size)], "Cluster IDs differ")
        require(merged["source_records"] == n and merged["num_clusters"] == size and merged["split"] == "train"
                and merged["seed"] == COMPRESSION_SEED, "Cluster metadata differs from learned training prototypes")
        require(merged["count_bias_applied"] is False and merged["exact_function_preserving"] is False, "Invalid merge transformation flags")
        expected_keys, expected_values, expected_variances = [], [], []
        normalized = F.normalize(original_keys.float(), dim=-1)
        for i in range(size):
            members = (labels == i).nonzero().flatten()
            key_mean = normalized[members].double().mean(0)
            if key_mean.norm() <= 1e-12:
                key_mean = normalized[members[0]].double()
            value_mean = original_values[members].double().mean(0)
            expected_keys.append(F.normalize(key_mean, dim=0))
            expected_values.append(value_mean)
            expected_variances.append((original_values[members].double() - value_mean).square().mean())
        for key, expected in (("keys", torch.stack(expected_keys)), ("values", torch.stack(expected_values)),
                              ("within_value_variance", torch.stack(expected_variances))):
            _close(merged[key], expected, "Independent cluster " + key)
        _close(packet["keys"], merged["keys"], "Selected cluster keys", exact=True)
        _close(packet["values"], merged["values"], "Selected cluster values", exact=True)
        # A deterministic algorithm rerun is meaningful only for the recorded
        # implementation. Arithmetic above is independently recomputed in FP64.
        declared = manifest.get("sources", {}).get("dictionary_vera.py")
        current = file_hash(_REPO / "src/vera_mem/dictionary_vera.py")
        require(declared == current, "Dictionary source changed; cannot claim deterministic compression replay")
        rerun = merge_bank(original_keys, original_values, size, split="train", seed=COMPRESSION_SEED)
        for key in ("keys", "values", "assignments", "counts", "weight_sums", "within_value_variance"):
            _close(merged[key], rerun[key], "Seeded cluster replay " + key, exact=True)
        details.update(compression_recomputed=True,
            mean_verification="Independent FP64 spherical key mean, value mean and within-value MSE; unit weight per learned prototype, not original Wikipedia cluster population.",
            max_value_mean_abs_error=float((merged["values"].double() - torch.stack(expected_values)).abs().max()),
            weighted_within_value_variance=float((merged["within_value_variance"] * counts).sum() / counts.sum()))
    else:
        raise ValueError("Unknown foundation mode")

    stored = episodic_audit.restore(packet["store"])
    rebuilt = PersistentVectorDB(config["key_dim"], config["rank"], config["temperature"], config["base_top_k"])
    for identity, key, value in zip(packet["ids"], packet["keys"], packet["values"]):
        rebuilt.write(identity, key, value, 0)
    require(rebuilt.hash() == stored.hash() == packet["bank_hash"] == c["bank_hash_before"] == c["bank_hash_after"],
            "Rebuilt foundation CPU bank hash mismatch")
    require(len(stored) == size and stored.ids == tuple(packet["ids"]), "Foundation snapshot identity mismatch")
    require(stored.timestamps == (0,) * size, "Foundation prototype timestamps differ")
    details.update(bank_hash=rebuilt.hash(), foundation_snapshot_sha256=file_hash(path), cpu_store_reconstructed=True)
    return details


def audit_run(directory, runs_root):
    directory = Path(directory).resolve()
    require(not analysis_exclusions(directory, runs_root), "Run is excluded from analysis")
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("protocol") == summary.RUN_PROTOCOL and manifest.get("complete") is True
            and manifest.get("stage") == manifest.get("configuration", {}).get("stage") == "eval", "Not a complete coldstart evaluation")
    require(manifest.get("backbone_unchanged") is True, "Backbone preservation missing")
    # Existing validator performs arithmetic/teacher-usage checks without torch;
    # the independent bank replay below fills its declared tensor-audit gap.
    arithmetic = summary.audit_foundation(directory, manifest, summary.ia.audit_evaluation(directory, manifest))
    files = [directory / name for name in ("manifest.json", "metrics.json", "assignments.json", "writes.jsonl", "predictions.jsonl",
                                          "foundation.pt", "foundation_usage.jsonl")]
    files += [directory / "banks" / (phase + ".pt") for phase in summary.ia.PHASES]
    before = {str(p.relative_to(directory)): file_hash(p) for p in files}
    episodic = episodic_audit.audit_run(directory)
    foundation = audit_foundation_tensor(directory, manifest, runs_root)
    predictions = read_jsonl(directory / "predictions.jsonl")
    teacher_count = 0
    for row in predictions:
        if row["method"] == "teacher":
            teacher_count += 1
            require(row["teacher_context"] and not row["selected_ids"] and not row["token_retrieval_sequence"],
                    "Teacher recorded an episodic route or lacks context")
            require(row["recall_at_1"] is None and row["recall_at_4"] is None, "Teacher claims episodic retrieval")
    after = {str(p.relative_to(directory)): file_hash(p) for p in files}
    require(before == after, "Audit changed an input artifact")
    bank_signature = hashlib.sha256(json.dumps({p: dict(base_hashes=r["base_hashes"], targets=r["targets"])
        for p, r in episodic["phases"].items()}, sort_keys=True).encode()).hexdigest()
    return dict(run_dir=str(directory), complete=True, split=manifest["result"]["split"],
                arm=manifest["configuration"].get("arm"), domain=manifest["configuration"].get("domain"),
                experiment_scope=experiment_scope(manifest["configuration"].get("arm")),
                evaluation_teacher_context="original source; training-only teacher intervention is not applied by evaluate_coldstart",
                checkpoint_sha256=manifest["checkpoint_sha256"], cache_sha256=manifest["cache_sha256"],
                comparison_key=arithmetic["comparison_key"], foundation=foundation, episodic=episodic,
                episodic_bank_signature=bank_signature, teacher_generations=teacher_count,
                teacher_foundation_zero_recorded_searches_verified=True,
                teacher_episodic_trace_empty_verified=True, artifact_hashes_before=before, artifact_hashes_after=after)


def audit_completed(runs_root):
    root = Path(runs_root).resolve()
    results, skipped, excluded = [], [], []
    for path in summary.discover(root):
        markers = analysis_exclusions(path.parent, root)
        if markers:
            excluded.append(dict(run_dir=str(path.parent), manifest_not_read=True, markers=markers))
            continue
        manifest = read_json(path)
        if manifest.get("protocol") != summary.RUN_PROTOCOL or manifest.get("stage") != "eval":
            continue
        if manifest.get("complete") is not True:
            skipped.append(dict(run_dir=str(path.parent), reason="Incomplete evaluation; raw files not opened"))
            continue
        suite_path = path.parent.parent / "suite.json"
        if suite_path.exists():
            suite = read_json(suite_path)
            require(suite.get("protocol") == "coldstart-experiment-v1", "Unexpected suite protocol")
            jobs = [job for job in suite["jobs"] if job["name"] == path.parent.name]
            require(len(jobs) == 1 and jobs[0]["status"] == "complete" and jobs[0]["exit_code"] == 0,
                    "Eval manifest complete but suite job not complete")
            require(jobs[0].get("manifest_sha256") == file_hash(path), "Suite/child manifest hash mismatch")
        result = audit_run(path.parent, root)
        result["scope"] = "smoke" if "smoke" in str(path.parent.relative_to(root)).lower() else "completed-local-evaluation"
        results.append(result)
        print(json.dumps(dict(audited=str(path.parent), base_mode=result["foundation"]["mode"],
            episodic_reads=result["episodic"]["verified_raw_reads"], foundation_hash=result["foundation"]["bank_hash"])), flush=True)
    groups = defaultdict(list)
    for result in results:
        groups[(result["checkpoint_sha256"], result["cache_sha256"], result["comparison_key"])].append(result)
    controls = []
    for identities, group in groups.items():
        require(len({r["episodic_bank_signature"] for r in group}) == 1,
                "Foundation control changed episodic banks or independent interventions")
        controls.append(dict(checkpoint_sha256=identities[0], cache_sha256=identities[1],
            modes=[r["foundation"]["mode"] for r in group], runs=[r["run_dir"] for r in group],
            episodic_snapshot_and_patch_identity_verified=True))
    return dict(protocol=PROTOCOL, audited_at=datetime.now(timezone.utc).isoformat(), audit_passed=True,
        complete=bool(results) and not skipped, partial=not results or bool(skipped), device="cpu",
        cuda_initialized=torch.cuda.is_initialized(), evaluations=results, skipped=skipped, excluded=excluded,
        scopes=scope_summary(results),
        cross_foundation_controls=controls,
        verified_raw_reads=sum(r["episodic"]["verified_raw_reads"] for r in results),
        verified_single_target_updates=sum(r["episodic"]["verified_single_target_updates"] for r in results),
        limitations=[
            "Reconstructs persisted CPU banks and source-prototype compression, not encoder features or model generations.",
            "Foundation usage saves aggregate indices/weights, not actual query vectors; retrieval logits/top-k cannot be independently recomputed.",
            "Teacher foundation calls are checked as zero; episodic bypass is supported by mode semantics and empty recorded routes, not a fresh model execution.",
            "Compression means weight each learned prototype equally. They do not reconstruct original Wikipedia observation counts or restore sparse softmax multiplicities.",
            "Ancestor analysis_excluded markers remove runs before their manifests are read. Excluded contains all discovered manifests in those suites, without inferring their stages. Parent checkpoint exclusions are recorded as provenance, not a claim of purity or a blanket ban on reuse.",
            "Supplementary teacher-feasibility evaluations are labeled separately from the original experiment. This artifact reports observed local runs, not a fixed-arm completeness gate or teacher-selection validation.",
            "Content hashes establish artifact consistency, not trusted-execution proof. No gradients or whole-sequence causal effect are inferred."])


def self_test():
    """Synthetic CPU evaluator fixtures only; pytest is needed by test fixtures."""
    import copy
    import tempfile
    import unittest
    sys.path.insert(0, str(_REPO / "tests"))
    from test_coldstart_eval import TinyDictionary, FoundationBackend, packet
    from vera_mem.coldstart_eval import evaluate_coldstart

    class Checks(unittest.TestCase):
        def setUp(self):
            self.temporary = tempfile.TemporaryDirectory()
            self.root = Path(self.temporary.name) / "coldstart_fixture"
            self.root.mkdir()
            self.module = TinyDictionary()
            with torch.no_grad():
                self.module.base_keys.copy_(torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]))
                self.module.base_values.copy_(torch.tensor([[1.], [3.], [10.], [14.]]))
            self.source = self.root / "checkpoint.pt"
            torch.save(dict(protocol=summary.RUN_PROTOCOL, architecture=self.module.configuration(),
                            module=self.module.state_dict()), self.source)

        def tearDown(self):
            self.temporary.cleanup()

        def fixture(self, mode="full"):
            output = self.root / mode
            count = 2 if mode in {"random256", "cluster256"} else None
            result = evaluate_coldstart(FoundationBackend(self.module), self.module, packet(), output,
                                       include_teacher=True, max_cases=2, base_mode=mode, merge_count=count)
            manifest = dict(protocol=summary.RUN_PROTOCOL, complete=True, stage="eval", backbone_unchanged=True,
                configuration=dict(stage="eval", checkpoint=str(self.source), base_mode=mode, merge_count=count,
                                   arm="test", model="fixed", domain="synthetic"),
                checkpoint_sha256=file_hash(self.source), cache_sha256="1" * 64, result=result,
                sources={"dictionary_vera.py": file_hash(_REPO / "src/vera_mem/dictionary_vera.py")})
            (output / "manifest.json").write_text(json.dumps(manifest))
            return output, manifest

        def edit_foundation(self, output, manifest, mutate):
            path = output / "foundation.pt"
            value = torch.load(path, weights_only=True)
            mutate(value)
            torch.save(value, path)
            manifest["result"]["coldstart"]["foundation_snapshot_sha256"] = file_hash(path)
            (output / "metrics.json").write_text(json.dumps(manifest["result"]))
            (output / "manifest.json").write_text(json.dumps(manifest))

        def test_all_modes_complete_and_preserve_episodic_banks(self):
            for mode in ("full", "off", "random256", "cluster256"):
                self.fixture(mode)
            result = audit_completed(self.root)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["evaluations"]), 4)
            self.assertEqual(result["verified_raw_reads"], 192)
            self.assertTrue(all(g["episodic_snapshot_and_patch_identity_verified"] for g in result["cross_foundation_controls"]))

        def test_original_prototypes_are_bound_to_checkpoint(self):
            output, manifest = self.fixture()
            self.edit_foundation(output, manifest, lambda p: p["original_values"][0].add_(1))
            with self.assertRaisesRegex(ValueError, "Checkpoint source values"):
                audit_run(output, self.root)

        def test_random_indices_and_cluster_members_are_independently_checked(self):
            output, manifest = self.fixture("random256")
            self.edit_foundation(output, manifest, lambda p: p["source_indices"].reverse())
            with self.assertRaisesRegex(ValueError, "source indices"):
                audit_run(output, self.root)
            output, manifest = self.fixture("cluster256")
            self.edit_foundation(output, manifest, lambda p: p["members"].reverse())
            with self.assertRaisesRegex(ValueError, "member mapping"):
                audit_run(output, self.root)

        def test_single_target_patch_corruption_is_rejected(self):
            output, _ = self.fixture()
            path = output / "banks/HC.pt"
            value = torch.load(path, weights_only=True)
            value["interventions"][0]["banks"]["real"]["value"].add_(1)
            torch.save(value, path)
            with self.assertRaisesRegex(ValueError, "hash"):
                audit_run(output, self.root)

        def test_teacher_foundation_read_cannot_hide_behind_updated_sidecar_hash(self):
            output, manifest = self.fixture()
            path = output / "foundation_usage.jsonl"
            rows = read_jsonl(path)
            next(r for r in rows if r["teacher"])["search_calls"] = 1
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            manifest["result"]["coldstart"]["foundation_usage_sha256"] = file_hash(path)
            (output / "metrics.json").write_text(json.dumps(manifest["result"]))
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Teacher read"):
                audit_run(output, self.root)

        def test_incomplete_evaluation_never_opens_raw_or_checkpoint(self):
            output = self.root / "incomplete"
            output.mkdir()
            (output / "manifest.json").write_text(json.dumps(dict(protocol=summary.RUN_PROTOCOL, stage="eval", complete=False)))
            (output / "predictions.jsonl").write_text("invalid, must never read")
            result = audit_completed(self.root)
            self.assertTrue(result["partial"])
            self.assertEqual(len(result["skipped"]), 1)
            self.assertFalse(result["evaluations"])

        def test_ancestor_exclusion_skips_even_unreadable_manifest(self):
            suite = self.root / "old_suite"
            output = suite / "nested" / "eval"
            output.mkdir(parents=True)
            (output / "manifest.json").write_text("invalid, must not be read")
            metadata = dict(reason="Overlapping B replacement corrupted anchor", evidence=["record-940"], superseded_by="new_suite")
            marker = suite / "analysis_excluded.json"
            marker.write_text(json.dumps(metadata))
            result = audit_completed(self.root)
            self.assertFalse(result["evaluations"])
            self.assertFalse(result["skipped"])
            self.assertEqual(result["excluded"][0]["markers"][0]["metadata"], metadata)
            self.assertEqual(result["excluded"][0]["markers"][0]["sha256"], file_hash(marker))
            with self.assertRaisesRegex(ValueError, "excluded from analysis"):
                audit_run(output, self.root)

        def test_excluded_parent_is_hash_bound_and_labeled_not_banned(self):
            output, manifest = self.fixture()
            excluded = self.root / "old_source"
            excluded.mkdir()
            parent = excluded / "checkpoint.pt"
            parent.write_bytes(self.source.read_bytes())
            marker = excluded / "analysis_excluded.json"
            marker.write_text(json.dumps(dict(reason="Old training excluded; source use requires explicit lineage", evidence="record-940", superseded_by="new_source")))
            manifest["configuration"]["checkpoint"] = str(parent)
            (output / "manifest.json").write_text(json.dumps(manifest))
            result = audit_run(output, self.root)
            self.assertEqual(result["foundation"]["source_checkpoint_analysis_exclusions"][0]["path"], str(marker))

        def test_supplementary_evaluation_is_independent_of_original_arm_count(self):
            output, manifest = self.fixture()
            manifest["configuration"]["arm"] = "wiki_teacher_repair"
            (output / "manifest.json").write_text(json.dumps(manifest))
            result = audit_completed(self.root)
            self.assertTrue(result["complete"])
            self.assertEqual(result["evaluations"][0]["experiment_scope"], "supplementary_teacher_feasibility")
            self.assertNotIn("original_experiment", result["scopes"])
            self.assertEqual(result["scopes"]["supplementary_teacher_feasibility"]["observed_count"], 1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
    if args.self_test:
        return self_test()
    if args.runs_root is None or args.output is None:
        parser.error("--runs-root and --output are required")
    require(not args.output.exists(), "Refusing to overwrite an audit artifact")
    started = time.perf_counter()
    result = audit_completed(args.runs_root)
    require(not torch.cuda.is_initialized(), "Audit unexpectedly initialized CUDA")
    result.update(elapsed_seconds=time.perf_counter() - started, audit_script_sha256=file_hash(Path(__file__)),
                  dependency_sha256={name: file_hash(Path(__file__).with_name(name))
                                     for name in ("audit_interface_banks.py", "summarize_coldstart.py", "summarize_interface.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=result["complete"], evaluations=len(result["evaluations"]),
                         reads=result["verified_raw_reads"], output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
