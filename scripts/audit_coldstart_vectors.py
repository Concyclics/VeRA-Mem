"""Read-only CPU audit of completed cold-start prototype updates and Adam state.

python scripts/audit_coldstart_vectors.py --self-test
python scripts/audit_coldstart_vectors.py --runs-root ../runs/xtrah100 --output ../runs/local/coldstart_vectors.json

Only completed training artifacts are opened. No feature cache, evaluation,
model forward, GPU, SSH, or checkpoint modification is needed.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
from replay_coldstart_banks import (torch, summary, require, read_json, read_jsonl,
                                    file_hash, resolve_checkpoint, analysis_exclusions, experiment_scope, scope_summary)

PROTOCOL = "coldstart-vector-update-audit-v1"
RELATIVE_EPSILON = 1e-12
MASS_RELATIVE_THRESHOLD = 1e-3


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def distribution(values):
    """Mass statistics, not an assertion of useful optimization or causality."""
    values = torch.as_tensor(values).detach().cpu().double().flatten()
    require(bool(torch.isfinite(values).all()) and bool((values >= 0).all()), "Invalid nonnegative slot mass")
    n, total = values.numel(), float(values.sum())
    require(n > 0, "Empty slot vector")
    nonzero = int((values > 0).sum())
    quantiles = torch.quantile(values, torch.tensor([0., .25, .5, .75, .9, .99, 1.], dtype=torch.float64))
    result = dict(slots=n, nonzero_slots=nonzero, total=total, mean=float(values.mean()),
                  quantiles=dict(zip(("min", "p25", "p50", "p75", "p90", "p99", "max"), quantiles.tolist())),
                  relative_to_max_threshold=MASS_RELATIVE_THRESHOLD)
    if total == 0:
        result.update(entropy_nats=0., normalized_entropy=0., effective_slots=0.,
                      top4_mass=0., slots_at_least_1e_3_of_max=0)
    else:
        p = values / total
        entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
        result.update(entropy_nats=entropy, normalized_entropy=entropy / math.log(n) if n > 1 else 0.,
                      effective_slots=math.exp(entropy), top4_mass=float(p.topk(min(4, n)).values.sum()),
                      slots_at_least_1e_3_of_max=int((values >= values.max() * MASS_RELATIVE_THRESHOLD).sum()))
    return result


def matrix_spectrum(matrix):
    """Small rank-64 CPU SVD; descriptive diversity, not downstream capacity."""
    matrix = matrix.detach().cpu().double()
    require(matrix.ndim == 2 and bool(torch.isfinite(matrix).all()), "Invalid spectrum matrix")
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = float(energy.sum())
    if total == 0:
        return dict(singular_values=singular.tolist(), frobenius_norm=0., stable_rank=0.,
                    energy_effective_rank=0., energy_participation_rank=0., top1_energy_fraction=0.,
                    top4_energy_fraction=0., rank_at_1e_3_max=0, rank_at_1e_6_max=0)
    p = energy / total
    positive = p[p > 0]
    return dict(singular_values=singular.tolist(), frobenius_norm=math.sqrt(total),
        stable_rank=total / float(energy[0]), energy_effective_rank=math.exp(float(-(positive*positive.log()).sum())),
        energy_participation_rank=1. / float(p.square().sum()), top1_energy_fraction=float(p[0]),
        top4_energy_fraction=float(p[:4].sum()), rank_at_1e_3_max=int((singular >= singular[0]*1e-3).sum()),
        rank_at_1e_6_max=int((singular >= singular[0]*1e-6).sum()))


def parameter_change(initial, final):
    require(initial.ndim == final.ndim == 2 and initial.shape == final.shape, "Prototype shapes differ")
    require(initial.dtype == final.dtype and bool(torch.isfinite(initial).all())
            and bool(torch.isfinite(final).all()), "Invalid prototype dtype/data")
    initial64, final64 = initial.double(), final.double()
    initial_norm, final_norm = initial64.norm(dim=1), final64.norm(dim=1)
    delta = (final64 - initial64).norm(dim=1)
    relative = delta / initial_norm.clamp_min(RELATIVE_EPSILON)
    valid_direction = (initial_norm > RELATIVE_EPSILON) & (final_norm > RELATIVE_EPSILON)
    initial_unit = initial64 / initial_norm.clamp_min(RELATIVE_EPSILON)[:, None]
    final_unit = final64 / final_norm.clamp_min(RELATIVE_EPSILON)[:, None]
    cosines = (initial_unit * final_unit).sum(dim=1).clamp(-1, 1)
    direction_delta = (final_unit - initial_unit).norm(dim=1)
    return dict(initial_sha256=tensor_hash(initial), final_sha256=tensor_hash(final),
                exactly_equal=torch.equal(initial, final), initial_norm=initial_norm.tolist(),
                final_norm=final_norm.tolist(), delta_l2=delta.tolist(), relative_delta_l2=relative.tolist(),
                cosine_similarity_to_parent=[float(v) if ok else None for v, ok in zip(cosines, valid_direction)],
                unit_direction_delta_l2=[float(v) if ok else None for v, ok in zip(direction_delta, valid_direction)],
                direction_definition="Unit-vector cosine/L2; null for norms <= 1e-12. Radial key updates alone do not change normalized-key retrieval.",
                relative_definition="L2(final-initial) / max(L2(initial), 1e-12), per slot",
                matrix_spectra={name:dict(raw=matrix_spectrum(value),
                    centered_across_slots=matrix_spectrum(value-value.mean(dim=0,keepdim=True)))
                    for name,value in (("initial",initial64),("final",final64),("delta",final64-initial64))},
                zero_initial_norm_slots=int((initial_norm == 0).sum()),
                delta_distribution=distribution(delta), relative_delta_distribution=distribution(relative),
                relative_delta_counts={str(t): int((relative >= t).sum()) for t in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, .1)})


def optimizer_source(directory, manifest):
    """Inspect a SHA-bound local snapshot as text; never import its code."""
    frozen = Path(directory).parent / "source/src/vera_mem/coldstart_run.py"
    source = frozen if frozen.is_file() else _REPO / "src/vera_mem/coldstart_run.py"
    declared = manifest.get("sources", {}).get("coldstart_run.py")
    require(file_hash(source) == declared, "Training source differs; cannot verify Adam parameter ordering")
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"]
    require(len(functions) == 1, "Missing unique training function for Adam group audit")
    statements = functions[0].body
    start = [i for i, node in enumerate(statements) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "groups" for target in node.targets)]
    require(len(start) == 1, "Ambiguous Adam parameter group construction")
    expected = ast.parse("""groups=[dict(params=list(module.Wv.parameters()),lr=1e-4),dict(params=[module.b],lr=.005),dict(params=list(module.Wq.parameters())+list(module.Wk.parameters()),lr=1e-5)]
if args.base_trainable: groups.append(dict(params=[module.base_keys,module.base_values],lr=1e-3))
optimizer=torch.optim.Adam(groups)
""").body
    actual = statements[start[0]:start[0]+3]
    require([ast.dump(n, include_attributes=False) for n in actual] ==
            [ast.dump(n, include_attributes=False) for n in expected], "Unsupported Adam parameter construction/order in recorded source")
    return dict(path=str(source), sha256=declared, frozen_snapshot=frozen.is_file(),
                group_construction_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=actual,type_ignores=[])).encode()).hexdigest())


def optimizer_statistics(checkpoint, manifest, directory):
    """Map Adam IDs using the source-bound group order and verify every shape."""
    config, architecture = manifest["configuration"], checkpoint["architecture"]
    require(not architecture.get("train_B") and not architecture.get("value_mlp_hidden"), "Unsupported optimizer architecture")
    trainable = config["base_trainable"]
    require(type(trainable) is bool and architecture["base_trainable"] == trainable, "Trainable foundation flag differs")
    optimizer = checkpoint.get("optimizer")
    require(isinstance(optimizer, dict), "Missing completed Adam state")
    names = [["Wv.weight"], ["b"], ["Wq.weight", "Wk.weight"]]
    rates = [1e-4, .005, 1e-5]
    if trainable:
        names.append(["base_keys", "base_values"])
        rates.append(1e-3)
    groups = optimizer["param_groups"]
    require(len(groups) == len(names), "Optimizer group count differs")
    # Shape alone cannot distinguish two identically shaped K/V parameters.
    # Bind the known group order to the runtime source recorded by the manifest.
    source = optimizer_source(directory, manifest)
    mapped, all_ids = {}, []
    for index, (group, group_names, rate) in enumerate(zip(groups, names, rates)):
        ids = group["params"]
        require(len(ids) == len(group_names) and group["lr"] == rate, "Optimizer group shape/rate differs")
        require(tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
                and group["weight_decay"] == 0 and not group["amsgrad"], "Unsupported Adam configuration")
        all_ids.extend(ids)
        for identity, name in zip(ids, group_names):
            state = optimizer["state"].get(identity)
            require(isinstance(state, dict), "Missing Adam parameter state: " + name)
            step = float(state["step"])
            require(step == checkpoint["step"] == config["updates"], "Adam step differs from completed budget")
            shape = checkpoint["module"][name].shape
            for field in ("exp_avg", "exp_avg_sq"):
                require(state[field].shape == shape and bool(torch.isfinite(state[field]).all()), "Adam tensor shape/data differs: " + name)
            require(bool((state["exp_avg_sq"] >= 0).all()), "Negative Adam second moment")
            if name.startswith("base_"):
                second = state["exp_avg_sq"].double()
                root_mean = second.mean(dim=1).sqrt()
                correction = 1. - group["betas"][1] ** step
                mapped[name] = dict(parameter_id=identity, group=index, shape=list(shape), step=int(step),
                    beta2=group["betas"][1], bias_correction=correction,
                    sqrt_mean_exp_avg_sq=root_mean.tolist(),
                    sqrt_mean_bias_corrected_exp_avg_sq=(root_mean / math.sqrt(correction)).tolist(),
                    mass_distribution=distribution(root_mean),
                    definition="Per-slot sqrt(mean(exp_avg_sq across coordinates)), final Adam EMA after group clipping; not raw gradient or full-trajectory gradient energy")
    require(len(all_ids) == len(set(all_ids)) and set(all_ids) == set(optimizer["state"]), "Duplicate/unmapped Adam parameter IDs")
    return dict(base_trainable=trainable, source_sha256=source["sha256"], source=source,
                mapping="Recorded training source group order, group cardinalities, learning rates, steps and all parameter/state shapes verified; equal-shaped K/V identity relies on source order",
                base_moments=mapped)


def usage_statistics(usage, status, base_size, top_k):
    result = {}
    for key in ("top1", "topk"):
        values = usage[key]
        require(values.shape == (base_size,) and values.dtype == torch.int64 and bool((values >= 0).all()), "Invalid usage counts")
        coverage = int((values > 0).sum())
        require(coverage == status["sampled_last_forward_" + key + "_coverage"], "Usage coverage differs from status")
        result[key] = dict(counts=values.tolist(), coverage=coverage, coverage_fraction=coverage / base_size,
                           count_mass_distribution=distribution(values))
    require(bool((usage["top1"] <= usage["topk"]).all()) and int(usage["topk"].sum()) == min(top_k, base_size) * int(usage["top1"].sum()),
            "Top-1/top-k usage totals disagree")
    for key, label in (("key_gradient", "key"), ("value_gradient", "value")):
        values = usage[key]
        require(values.shape == (base_size,) and values.dtype == torch.bool, "Invalid gradient-presence union")
        require(int(values.sum()) == status[label + "_gradient_coverage"], "Gradient union differs from status")
        result[key] = dict(nonzero_union_slots=int(values.sum()), mask=values.tolist())
    result["scope"] = status["usage_scope"]
    result["limitation"] = "Only last student forward per update, including padding; neither all-forward routing nor first-answer-only routing. Dense/ST top-k diagnostics are sparse subsets, not dense backward coverage."
    return result


def audit_run(directory, runs_root):
    directory = Path(directory).resolve()
    require(not analysis_exclusions(directory, runs_root), "Run is excluded from analysis")
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("protocol") == summary.RUN_PROTOCOL and manifest.get("complete") is True
            and manifest.get("stage") == manifest.get("configuration", {}).get("stage") == "train", "Not a completed coldstart training run")
    require(manifest.get("backbone_unchanged") is True, "Backbone preservation missing")
    files = [directory / name for name in ("manifest.json", "training_status.json", "training.jsonl", "last.pt", "usage.pt")]
    parent_path = resolve_checkpoint(directory, manifest, runs_root)
    files.append(parent_path)
    before = {str(path): file_hash(path) for path in files}
    arithmetic = summary.audit_training(directory, manifest)
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("protocol") == parent.get("protocol") == summary.RUN_PROTOCOL, "Wrong checkpoint protocol")
    config, status = manifest["configuration"], manifest["result"]
    require(checkpoint["step"] == config["updates"] and checkpoint["seed"] == config["seed"], "Final checkpoint step/seed differs")
    architecture = checkpoint["architecture"]
    require(architecture["base_size"] == parent["architecture"]["base_size"], "Parent foundation size differs")
    changes = {name: parameter_change(parent["module"][name], checkpoint["module"][name]) for name in ("base_keys", "base_values")}
    for name, width in (("base_keys", architecture["key_dim"]), ("base_values", architecture["rank"])):
        require(checkpoint["module"][name].shape == (architecture["base_size"], width), "Architecture prototype shape differs")
    adam = optimizer_statistics(checkpoint, manifest, directory)
    usage = usage_statistics(torch.load(directory / "usage.pt", map_location="cpu", weights_only=True),
                             status, architecture["base_size"], architecture["base_top_k"])
    if not config["base_trainable"]:
        require(all(x["exactly_equal"] for x in changes.values()), "Frozen foundation parameters changed")
        require(not usage["key_gradient"]["nonzero_union_slots"] and not usage["value_gradient"]["nonzero_union_slots"],
                "Frozen foundation claims gradients")
    parent_info = dict(path=str(parent_path), checkpoint_sha256=file_hash(parent_path), step=parent.get("step"),
                       analysis_exclusions=analysis_exclusions(parent_path.parent, runs_root),
                       base_keys_sha256=changes["base_keys"]["initial_sha256"], base_values_sha256=changes["base_values"]["initial_sha256"])
    parent_manifest = parent_path.parent / "manifest.json"
    if parent_manifest.is_file():
        upstream = read_json(parent_manifest)
        require(upstream.get("protocol") == summary.RUN_PROTOCOL and upstream.get("complete") is True, "Parent run is not complete")
        parent_info.update(manifest_sha256=file_hash(parent_manifest), stage=upstream.get("stage"),
                           arm=upstream.get("configuration", {}).get("arm"),
                           upstream_checkpoint=upstream.get("configuration", {}).get("checkpoint"),
                           upstream_checkpoint_sha256=upstream.get("checkpoint_sha256"),
                           cluster_initialization=upstream.get("configuration", {}).get("cluster_init"))
    else:
        parent_info["manifest_status"] = "Absent; checkpoint content is bound by the child manifest SHA256"
    rows = read_jsonl(directory / "training.jsonl")
    retained = [r["base_retained_dense_mass_mean"] for r in rows if "base_retained_dense_mass_mean" in r]
    require(all(math.isfinite(x) and 0 <= x <= 1 + 1e-6 for x in retained), "Invalid retained dense mass")
    after = {str(path): file_hash(path) for path in files}
    require(before == after, "Input artifact changed during audit")
    return dict(run_dir=str(directory), arm=config.get("arm"), domain=config.get("domain"), updates=config["updates"],
        experiment_scope=experiment_scope(config.get("arm")), teacher_strategy=config.get("teacher_strategy", "baseline"),
        routing=config["routing"], alpha_base=config["alpha_base"], base_trainable=config["base_trainable"],
        initialization=parent_info, final_checkpoint_sha256=file_hash(directory / "last.pt"), changes=changes,
        adam=adam, usage=usage, schedule_sha256=arithmetic["schedule_sha256"],
        retained_dense_mass=dict(observed_updates=len(retained), total_updates=len(rows),
            update_mean=sum(retained) / len(retained) if retained else None, minimum_update_mean=min(retained) if retained else None,
            scope="Unweighted mean of recorded last-forward means; not all tokens or all forwards"),
        artifact_hashes_before=before, artifact_hashes_after=after)


def audit_completed(runs_root):
    root = Path(runs_root).resolve()
    results, skipped, excluded = [], [], []
    for path in summary.discover(root):
        markers = analysis_exclusions(path.parent, root)
        if markers:
            excluded.append(dict(run_dir=str(path.parent), manifest_not_read=True, markers=markers))
            continue
        manifest = read_json(path)
        if manifest.get("protocol") != summary.RUN_PROTOCOL or manifest.get("stage") != "train":
            continue
        if manifest.get("complete") is not True:
            skipped.append(dict(run_dir=str(path.parent), reason="Incomplete training; tensors and raw logs not opened"))
            continue
        result = audit_run(path.parent, root)
        results.append(result)
        print(json.dumps(dict(audited=result["run_dir"], trainable=result["base_trainable"],
            changed_value_slots=result["changes"]["base_values"]["delta_distribution"]["nonzero_slots"])), flush=True)
    groups = defaultdict(list)
    for r in results:
        p = r["initialization"]
        groups[(p["base_keys_sha256"], p["base_values_sha256"])].append(r)
    return dict(protocol=PROTOCOL, audited_at=datetime.now(timezone.utc).isoformat(), audit_passed=True,
        complete=bool(results) and not skipped, partial=not results or bool(skipped), device="cpu", cuda_initialized=torch.cuda.is_initialized(),
        trainings=results, skipped=skipped, excluded=excluded, scopes=scope_summary(results),
        identical_initial_bank_groups=[dict(base_keys_sha256=key[0], base_values_sha256=key[1],
            runs=[r["run_dir"] for r in rows], arms=[r["arm"] for r in rows],
            same_parent_checkpoint=len({r["initialization"]["checkpoint_sha256"] for r in rows}) == 1)
            for key, rows in groups.items()],
        limitations=[
            "Parameter deltas are final minus parent checkpoint, not cumulative path length or a proof of useful learning; small net changes can hide cancelling updates.",
            "Adam statistics are the final exponentially decayed second moment of clipped gradients, not raw gradients, original loss sensitivities, or whole-training gradient energy. Bias correction changes scale, not concentration.",
            "Mass entropy uses per-slot sqrt(mean(exp_avg_sq)); a nonzero slot can carry negligible mass. Relative-to-maximum thresholds are descriptive, not scientific effectiveness cutoffs.",
            "Parameter and gradient norms depend on representation scale; compare concentration and relative updates jointly with downstream evaluation, without attributing quality to coverage alone.",
            "SVD describes raw prototype and net-update matrices, plus matrices with the across-slot mean removed. Its energy ranks use squared singular values and do not prove usable memory capacity, semantic diversity, or gradient rank throughout training.",
            "Usage is only the last student forward of each update and may include padding; it cannot prove all-token or first-answer routing coverage.",
            "Visits with alpha_base=0 have no direct foundation contribution. Base keys are normalized for lookup, so radial key changes can leave retrieval unchanged; cosine/unit-direction changes are also reported.",
            "Ancestor analysis_excluded markers remove runs before their manifests are read. Excluded lists all discovered manifests in those suites, without inferring stages. Parent checkpoint exclusions remain explicit provenance; this audit does not judge whether their data semantics are safe for reuse.",
            "Supplementary teacher-feasibility training is labeled separately and never changes the original experiment's observed records. Scope counts are descriptive; protocol coverage and selection-evidence gates belong to the independent experiment audit.",
            "Completeness covers discovered local train manifests only; it does not establish planned-arm or evaluation completeness. No evaluations, feature caches or models are read."])


def self_test():
    import tempfile
    import unittest
    from vera_mem.dictionary_vera import DictionaryVeRA

    class Checks(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.root = Path(self.temp.name) / "coldstart_fixture"
            self.root.mkdir()
        def tearDown(self):
            self.temp.cleanup()
        def fixture(self, trainable=True, name="train"):
            directory = self.root / name
            directory.mkdir()
            parent_dir = self.root / "init"
            parent_dir.mkdir(exist_ok=True)
            m = DictionaryVeRA(5, 4, rank=2, key_dim=3, base_size=8, base_top_k=2, base_trainable=False, seed=19)
            parent = parent_dir / "last.pt"
            if not parent.exists():
                torch.save(dict(protocol=summary.RUN_PROTOCOL, module=m.state_dict(), architecture=m.configuration(), step=0), parent)
                (parent_dir / "manifest.json").write_text(json.dumps(dict(protocol=summary.RUN_PROTOCOL, complete=True, stage="init", configuration=dict(arm="init", cluster_init=False))))
            m.set_base_trainable(trainable)
            groups = [dict(params=list(m.Wv.parameters()), lr=1e-4), dict(params=[m.b], lr=.005),
                      dict(params=list(m.Wq.parameters()) + list(m.Wk.parameters()), lr=1e-5)]
            if trainable:
                groups.append(dict(params=[m.base_keys, m.base_values], lr=1e-3))
            optimizer = torch.optim.Adam(groups)
            for _ in range(2):
                for group in groups:
                    for p in group["params"]:
                        p.grad = torch.full_like(p, .01)
                if trainable:
                    for p in (m.base_keys, m.base_values):
                        p.grad[1:] = 1e-18
                optimizer.step()
            torch.save(dict(protocol=summary.RUN_PROTOCOL, module=m.state_dict(), architecture=m.configuration(),
                            optimizer=optimizer.state_dict(), step=2, seed=42), directory / "last.pt")
            sample = dict(targets=[0, 1], episode=[0, 1, 2, 3], columns=[0, 1], qviews=[0, 0], sviews=[0]*4)
            rows = [dict(step=i, routing="straight_through", sample=sample, elapsed_seconds=float(i), target_tokens=6,
                         base_keys_gradient_slots=8 if trainable else 0, base_values_gradient_slots=8 if trainable else 0,
                         base_retained_dense_mass_mean=.99) for i in (1, 2)]
            schedule = hashlib.sha256()
            for row in rows:
                schedule.update(json.dumps(row["sample"], sort_keys=True).encode())
            status = dict(complete=True, updates=2, totals=dict(target_tokens=12), schedule_sha256=schedule.hexdigest(),
                target_unique_count=2, bank_unique_count=4, key_gradient_coverage=8 if trainable else 0,
                value_gradient_coverage=8 if trainable else 0, sampled_last_forward_top1_coverage=1,
                sampled_last_forward_topk_coverage=2, usage_scope="last student forward per update, includes padded positions; diagnostic only",
                elapsed_seconds=2., trainable_parameters=1)
            torch.save(dict(top1=torch.tensor([2, 0, 0, 0, 0, 0, 0, 0]), topk=torch.tensor([2, 2, 0, 0, 0, 0, 0, 0]),
                            key_gradient=torch.full((8,), trainable), value_gradient=torch.full((8,), trainable)), directory / "usage.pt")
            (directory / "training_status.json").write_text(json.dumps(status))
            (directory / "training.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            manifest = dict(protocol=summary.RUN_PROTOCOL, complete=True, stage="train", backbone_unchanged=True,
                configuration=dict(stage="train", checkpoint=str(parent), base_trainable=trainable, updates=2, seed=42,
                                   batch_size=2, routing="straight_through", arm=name, domain="synthetic", alpha_base=.25),
                checkpoint_sha256=file_hash(parent), cache_sha256="1"*64, result=status,
                sources={"coldstart_run.py": file_hash(_REPO / "src/vera_mem/coldstart_run.py")})
            (directory / "manifest.json").write_text(json.dumps(manifest))
            return directory
        def test_concentration_is_not_boolean_nonzero_coverage(self):
            x = distribution(torch.tensor([1., 1e-12, 1e-12, 1e-12, 1e-12]))
            self.assertEqual(x["nonzero_slots"], 5)
            self.assertEqual(x["slots_at_least_1e_3_of_max"], 1)
            self.assertAlmostEqual(x["effective_slots"], 1., places=8)
            self.assertEqual(distribution(torch.zeros(8))["effective_slots"], 0.)
            self.assertAlmostEqual(distribution(torch.ones(8))["normalized_entropy"], 1.)
        def test_tiny_dense_moments_do_not_imply_many_updates(self):
            result = audit_run(self.fixture(), self.root)
            for name in ("base_keys", "base_values"):
                stats = result["adam"]["base_moments"][name]["mass_distribution"]
                self.assertEqual(stats["nonzero_slots"], 8)
                self.assertEqual(stats["slots_at_least_1e_3_of_max"], 1)
                self.assertEqual(result["changes"][name]["delta_distribution"]["nonzero_slots"], 1)
            self.assertEqual(result["usage"]["top1"]["coverage"], 1)
            self.assertEqual(result["artifact_hashes_before"], result["artifact_hashes_after"])
        def test_shared_parent_fixed_zero_and_changed_fixed_rejected(self):
            fixed = self.fixture(False, "fixed")
            self.fixture(True, "learned")
            result = audit_completed(self.root)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["identical_initial_bank_groups"]), 1)
            self.assertTrue(result["identical_initial_bank_groups"][0]["same_parent_checkpoint"])
            packet = torch.load(fixed / "last.pt", weights_only=True)
            packet["module"]["base_values"][0, 0] += .1
            torch.save(packet, fixed / "last.pt")
            with self.assertRaisesRegex(ValueError, "Frozen foundation"):
                audit_run(fixed, self.root)
        def test_optimizer_shapes_steps_and_source_order_fail_closed(self):
            directory = self.fixture()
            packet = torch.load(directory / "last.pt", weights_only=True)
            identity = packet["optimizer"]["param_groups"][-1]["params"][0]
            packet["optimizer"]["state"][identity]["exp_avg_sq"] = torch.zeros(1, 3)
            torch.save(packet, directory / "last.pt")
            with self.assertRaisesRegex(ValueError, "Adam tensor shape"):
                audit_run(directory, self.root)
            directory = self.fixture(name="bad_step")
            packet = torch.load(directory / "last.pt", weights_only=True)
            next(iter(packet["optimizer"]["state"].values()))["step"] -= 1
            torch.save(packet, directory / "last.pt")
            with self.assertRaisesRegex(ValueError, "Adam step"):
                audit_run(directory, self.root)
            directory = self.fixture(name="bad_source")
            manifest = read_json(directory / "manifest.json")
            manifest["sources"]["coldstart_run.py"] = "0"*64
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "parameter ordering"):
                audit_run(directory, self.root)
        def test_relative_and_directional_changes_handle_radial_and_zero_vectors(self):
            stats = parameter_change(torch.tensor([[1., 0.], [0., 0.]]), torch.tensor([[2., 0.], [1., 0.]]))
            self.assertEqual(stats["relative_delta_l2"][0], 1.)
            self.assertEqual(stats["unit_direction_delta_l2"], [0., None])
            self.assertEqual(stats["cosine_similarity_to_parent"], [1., None])
        def test_spectrum_separates_full_rank_common_direction_and_no_update(self):
            full = matrix_spectrum(torch.eye(8))
            self.assertEqual(full["rank_at_1e_3_max"], 8)
            self.assertAlmostEqual(full["energy_effective_rank"], 8.)
            collapsed = matrix_spectrum(torch.ones(8, 3))
            self.assertEqual(collapsed["rank_at_1e_3_max"], 1)
            self.assertAlmostEqual(collapsed["top1_energy_fraction"], 1.)
            zero = parameter_change(torch.ones(8, 3), torch.ones(8, 3))["matrix_spectra"]
            self.assertEqual(zero["delta"]["raw"]["energy_effective_rank"], 0.)
            self.assertEqual(zero["final"]["centered_across_slots"]["rank_at_1e_3_max"], 0)
        def test_parent_sha_and_usage_counts_are_verified(self):
            directory = self.fixture()
            path = directory / "usage.pt"
            usage = torch.load(path, weights_only=True)
            usage["topk"][0] += 1
            torch.save(usage, path)
            with self.assertRaisesRegex(ValueError, "usage totals"):
                audit_run(directory, self.root)
            parent = self.root / "init/last.pt"
            with parent.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
                audit_run(directory, self.root)
        def test_incomplete_and_evaluation_raw_never_read(self):
            for name, stage in (("partial", "train"), ("evaluation", "eval")):
                directory = self.root / name
                directory.mkdir()
                (directory / "manifest.json").write_text(json.dumps(dict(protocol=summary.RUN_PROTOCOL, complete=False, stage=stage)))
                (directory / "last.pt").write_text("invalid, must not be opened")
            result = audit_completed(self.root)
            self.assertTrue(result["partial"])
            self.assertEqual(len(result["skipped"]), 1)
        def test_excluded_suite_is_not_read_or_used_in_training_statistics(self):
            suite = self.root / "old_suite"
            directory = suite / "nested/train"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("invalid, never read")
            metadata = dict(reason="Invalid B replacement", evidence=["record-940"], superseded_by="new_suite")
            (suite / "analysis_excluded.json").write_text(json.dumps(metadata))
            self.fixture(False, "valid")
            result = audit_completed(self.root)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["trainings"]), 1)
            self.assertEqual(result["excluded"][0]["markers"][0]["metadata"], metadata)
            with self.assertRaisesRegex(ValueError, "excluded from analysis"):
                audit_run(directory, self.root)
        def test_excluded_parent_keeps_hash_bound_lineage_without_semantic_gate(self):
            directory = self.fixture(False)
            parent_dir = self.root / "init"
            metadata = dict(reason="Old suite superseded", evidence="B repair", superseded_by="new_suite")
            (parent_dir / "analysis_excluded.json").write_text(json.dumps(metadata))
            upstream = read_json(parent_dir / "manifest.json")
            upstream["configuration"]["cluster_init"] = True
            (parent_dir / "manifest.json").write_text(json.dumps(upstream))
            result = audit_run(directory, self.root)
            self.assertTrue(result["initialization"]["cluster_initialization"])
            self.assertEqual(result["initialization"]["analysis_exclusions"][0]["metadata"], metadata)
        def test_frozen_training_source_can_differ_from_current_but_must_preserve_adam_order(self):
            directory = self.fixture()
            snapshot = directory.parent / "source/src/vera_mem/coldstart_run.py"
            snapshot.parent.mkdir(parents=True)
            text = (_REPO / "src/vera_mem/coldstart_run.py").read_text()
            snapshot.write_text(text + "\n# Frozen earlier experiment snapshot.\n")
            manifest = read_json(directory / "manifest.json")
            manifest["sources"]["coldstart_run.py"] = file_hash(snapshot)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = audit_run(directory, self.root)
            self.assertTrue(result["adam"]["source"]["frozen_snapshot"])
            self.assertEqual(result["adam"]["source_sha256"], file_hash(snapshot))
            swapped = text.replace("params=[module.base_keys,module.base_values]", "params=[module.base_values,module.base_keys]")
            self.assertNotEqual(swapped, text)
            snapshot.write_text(swapped)
            manifest["sources"]["coldstart_run.py"] = file_hash(snapshot)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "construction/order"):
                audit_run(directory, self.root)
        def test_supplementary_only_training_scope_has_no_fixed_arm_count(self):
            directory = self.fixture(name="wiki_teacher_warm")
            manifest = read_json(directory / "manifest.json")
            manifest["configuration"]["teacher_strategy"] = "gold_annotated"
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = audit_completed(self.root)
            self.assertTrue(result["complete"])
            self.assertEqual(result["trainings"][0]["teacher_strategy"], "gold_annotated")
            self.assertEqual(result["scopes"]["supplementary_teacher_feasibility"]["observed_count"], 1)
            self.assertNotIn("original_experiment", result["scopes"])
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
    result.update(elapsed_seconds=time.perf_counter()-started, audit_script_sha256=file_hash(Path(__file__)),
        dependency_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
                           ("replay_coldstart_banks.py", "summarize_coldstart.py", "summarize_interface.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=result["complete"], trainings=len(result["trainings"]), output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
