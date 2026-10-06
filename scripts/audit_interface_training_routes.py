"""CPU-only, train-only first-position routing proxy for completed experiments.

  /home/chenhan/miniconda3/envs/agent/bin/python scripts/audit_interface_training_routes.py \
      --runs-root ../runs/xtrah100 --output ../runs/local/interface_route_proxy.json

Waits for no process and launches nothing. All six step-3 jobs must already be
complete locally. Completed step-4 training jobs are included; incomplete ones
are listed without loading their cache/checkpoint. No dev/confirm files or
generation results are opened. Cache tensors are memory-mapped on CPU.

This replays the last 128 logged episodes under each FINAL checkpoint, using
cached frozen first-position query features. It is neither historical training
route reconstruction nor actual-model evaluation. In particular, both-world
target misses imply zero DIRECT target-value contribution at that one mixture,
not zero gradient through earlier prompt positions, later attention/decoding,
or shared writer parameters. There is no full-sequence gradient claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch.nn import functional as F


PROTOCOL = "interface-training-first-position-route-proxy-v1"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
REPO = Path(__file__).resolve().parents[1]
NOTES = [
    "Only completed train stages and their train cache are read; no confirmation predictions or scores are consulted.",
    "Final-checkpoint routing on the last 128 logged episodes is not the route realized at those historical updates.",
    "Frozen cached first-position inputs are a proxy: BF16 batching/rounding may change marginal top-k membership relative to actual forwards.",
    "Both-world target misses indicate no direct target-value edge at this single sparse mixture, not full-sequence gradient starvation.",
    "Earlier prompt positions may retrieve the target; later-layer causal attention can propagate those residuals to the prediction position. Later continuation positions can also retrieve it.",
    "Shared Wv can change through other selected values. Dense address supervision has gradients to candidate keys even when a candidate is outside sparse top-k.",
    "This is a routing diagnostic hypothesis, not evidence that a whole behavior/hidden objective has zero gradient or that routing causally explains final exact match.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class FrozenAddress:
    """CPU reproduction of the frozen StableVectorVeRA query/key contract.

    The audit verifies runtime vector_vera.py/stable_vector_vera.py against the
    local implementation and tests this calculation against their real encoders.
    Only Wq, Wk and their fixed centers are needed; no backbone is instantiated.
    """
    def __init__(self, architecture, state):
        self.width, self.key_dim, self.top_k = (architecture[k] for k in ("in_features", "key_dim", "top_k"))
        self.epsilon = float(architecture.get("input_epsilon", 1e-6))
        require(self.width > 0 and self.key_dim > 0 and self.top_k == 4 and math.isfinite(self.epsilon) and self.epsilon > 0,
                "Expected valid sparse top-4 address architecture")
        require(bool(state["statistics_fitted"].item()), "Address statistics are not fitted")
        self.state = {}
        for key, shape in (("Wq.weight", (self.key_dim, self.width)), ("Wk.weight", (self.key_dim, self.width)),
                           ("query_center", (self.width,)), ("support_center", (self.width,))):
            value = state[key]
            require(value.device.type == "cpu" and value.dtype == torch.float32 and tuple(value.shape) == shape
                    and bool(torch.isfinite(value).all()), "Unexpected address tensor: " + key)
            self.state[key] = value.detach()

    def encode(self, x, domain):
        require(domain in {"query", "support"} and x.device.type == "cpu" and x.shape[-1] == self.width,
                "CPU address features/domain mismatch")
        require(bool(torch.isfinite(x).all()), "Selected train features contain nonfinite values")
        normalized = F.normalize(x.float(), p=2, dim=-1, eps=1e-12) * math.sqrt(self.width)
        centered = normalized - self.state[domain + "_center"]
        centered = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.epsilon)
        weight = self.state["Wq.weight" if domain == "query" else "Wk.weight"]
        return F.normalize(F.linear(centered, weight), dim=-1, eps=1e-12)


def validate_packet(packet):
    require(packet.get("protocol") == "memory-interface-v1" and packet.get("split") == "train"
            and packet.get("task") == "extended" and packet.get("model_revision") == REVISION,
            "Only pinned extended TRAIN feature packets may be audited")
    require(packet.get("writer_prefix") == "Remember this information: ", "Wrong writer feature prefix")
    rows, q, last = packet["rows"], packet["q"], packet["last"]
    require(rows and q.device.type == last.device.type == "cpu" and q.ndim == 3 and last.ndim == 4
            and q.shape[0] == last.shape[0] == len(rows) and last.shape[1] == 2 and q.shape[2] == last.shape[3],
            "Invalid train query/support feature axes")
    require(q.shape[1] == len(packet["qids"]) and last.shape[2] == len(packet["sids"]), "Training template axes disagree")
    require(len({row["id"] for row in rows}) == len(rows), "Duplicate train fact IDs")
    require(all(len(row["questions"]) == q.shape[1] and len(row["supports"]) == 2
                and all(len(world) == last.shape[2] for world in row["supports"]) for row in rows),
            "Training text and feature view counts disagree")


def validate_sample(sample, packet):
    targets, episode, columns, qviews, sviews = [sample[k] for k in ("targets", "episode", "columns", "qviews", "sviews")]
    require(all(isinstance(x, list) and all(type(v) is int for v in x) for x in (targets, episode, columns, qviews, sviews)),
            "Episode indices must be integer lists")
    require(len(targets) == len(columns) == len(qviews) == 8 and len(episode) == len(sviews) >= 8,
            "Expected the actual batch-8 episode axes")
    require(len(set(targets)) == 8 and len(set(episode)) == len(episode), "Duplicate targets/episode facts")
    require(all(0 <= i < len(packet["rows"]) for i in targets + episode), "Train fact index out of bounds")
    require(all(0 <= column < len(episode) and episode[column] == target for target, column in zip(targets, columns)),
            "Target-to-episode column mapping is invalid")
    require(all(0 <= view < packet["q"].shape[1] for view in qviews)
            and all(0 <= view < packet["last"].shape[2] for view in sviews), "Training view index out of bounds")
    return targets, episode, columns, qviews, sviews


@torch.no_grad()
def episode_routes(reader, packet, log_row):
    targets, episode, columns, qviews, sviews = validate_sample(log_row["sample"], packet)
    batch, bank_size = len(targets), len(episode)
    require(packet["q"].shape[-1] == reader.width, "Checkpoint/cache feature widths differ")
    q = reader.encode(packet["q"][targets, qviews], "query")
    base_features = packet["last"][episode, 0, sviews]
    target_views = [sviews[column] for column in columns]
    alternate_features = packet["last"][targets, 1, target_views]
    a_keys = reader.encode(base_features, "support")
    b_target_keys = reader.encode(alternate_features, "support")
    a = a_keys.unsqueeze(0).expand(batch, -1, -1)
    b = a.clone()
    row_indices, column_indices = torch.arange(batch), torch.tensor(columns)
    b[row_indices, column_indices] = b_target_keys
    # Key equality is permitted at the intervened record; it is not proof of a
    # missing write. Only the target may differ, and its exact B source is known.
    non_target = torch.ones(batch, bank_size, dtype=torch.bool)
    non_target[row_indices, column_indices] = False
    require(torch.equal(a[non_target], b[non_target])
            and torch.equal(b[row_indices, column_indices], b_target_keys), "B bank is not a single-target replacement")
    results = []
    for world, bank in (("A", a), ("B", b)):
        similarities = torch.bmm(q.unsqueeze(1), F.normalize(bank, dim=-1, eps=1e-12).transpose(1, 2)).squeeze(1)
        scores, indices = similarities.topk(min(reader.top_k, bank_size), dim=-1)
        targets_scores = similarities[row_indices, column_indices]
        results.append(dict(indices=indices, scores=scores, target_scores=targets_scores))
    rows = []
    for row, (target, column) in enumerate(zip(targets, columns)):
        event = dict(step=log_row["step"], target_index=target, target_id=packet["rows"][target]["id"],
                     entity=packet["rows"][target]["entity"], relation=packet["rows"][target]["relation"],
                     column=column, query_view=qviews[row], support_view=target_views[row], bank_size=bank_size)
        for world, result in zip(("a", "b"), results):
            chosen = result["indices"][row].tolist()
            event.update({world + "_r1": int(chosen[0] == column), world + "_r4": int(column in chosen),
                          world + "_selected_ids": [packet["rows"][episode[index]]["id"] for index in chosen],
                          world + "_target_cosine": float(result["target_scores"][row]),
                          world + "_top4_boundary_margin": float(result["target_scores"][row] - result["scores"][row, -1])})
        event["both_miss4"] = int(not event["a_r4"] and not event["b_r4"])
        event["both_hit4"] = int(event["a_r4"] and event["b_r4"])
        event["either_hit4"] = int(event["a_r4"] or event["b_r4"])
        rows.append(event)
    return rows


def summarize(rows):
    require(rows, "No validated routing exposures")
    counts = {key: sum(row[key] for row in rows) for key in ("a_r1", "a_r4", "b_r1", "b_r4", "both_miss4", "both_hit4", "either_hit4")}
    result = dict(exposures=len(rows), unique_facts=len({row["target_id"] for row in rows}),
                  unique_entities=len({row["entity"] for row in rows}),
                  counts=counts, rates={key: value / len(rows) for key, value in counts.items()})
    for world in ("a", "b"):
        margins = torch.tensor([row[world + "_top4_boundary_margin"] for row in rows], dtype=torch.float64)
        result[world + "_top4_boundary_margin"] = dict(mean=float(margins.mean()),
            p05=float(torch.quantile(margins, .05)), p50=float(torch.quantile(margins, .5)), p95=float(torch.quantile(margins, .95)),
            note="Target cosine minus fourth-largest cosine. Near-zero values can be sensitive to BF16 numerical differences; no BF16 error bound is inferred.")
    result["by_relation"] = {relation: dict(exposures=sum(row["relation"] == relation for row in rows),
        both_miss4=sum(row["both_miss4"] for row in rows if row["relation"] == relation)) for relation in sorted({row["relation"] for row in rows})}
    return result


def discover(runs_root, tag):
    jobs, pending = [], []
    for directory in sorted(runs_root.glob(f"interface_step[34]_train_*_{tag}")):
        if not re.fullmatch(r"interface_step[34]_train_[abc]_" + re.escape(tag), directory.name):
            continue
        suite = read_json(directory / "suite.json")
        require(suite.get("protocol") == "interface-experiment-v1", "Unknown training suite protocol")
        plan = read_json(directory / "plan.json")
        require(len({entry["name"] for entry in plan}) == len(plan), "Duplicate planned training jobs")
        states = {entry["name"]: entry for entry in suite["jobs"]}
        for entry in plan:
            name = entry["name"]
            require(Path(name).name == name and name not in {".", ".."}, "Invalid run name")
            status = states.get(name, {})
            identity = directory.name + "/" + name
            if status.get("status") != "complete" or status.get("exit_code") != 0:
                pending.append(identity)
                continue
            manifest = read_json(directory / name / "manifest.json")
            require(manifest.get("complete") is True and manifest.get("protocol") == "memory-interface-v1"
                    and manifest.get("backbone_unchanged") is True, "Completed job lacks a valid frozen-backbone manifest")
            cfg = manifest["configuration"]
            require(cfg.get("stage") == "train" and cfg.get("task") == "extended", "Only extended training jobs are eligible")
            expected_suffix = f"/runs/interface_extended_prepare_{tag}/prepare/train.pt"
            require(cfg["cache"].endswith(expected_suffix), "Only the declared extended train cache is allowed")
            jobs.append((directory / name, manifest))
    warm = [name.parent.name + "/" + name.name for name, _ in jobs if "step3_train" in name.parent.name]
    required = {f"{arm}_{seed}" for arm in ("coupled", "decoupled") for seed in (42, 43, 44)}
    require(len(warm) == 6 and {name.rsplit("/", 1)[1] for name in warm} == required,
            "Wait until all six step-3 training jobs are completed and locally backed up")
    return jobs, pending


def audit_run(directory, manifest, packet, cache_hash, last_updates):
    cfg = manifest["configuration"]
    require(manifest["cache_sha256"] == cache_hash, "Checkpoint run and audited train cache differ")
    status = read_json(directory / "training_status.json")
    require(status.get("complete") is True and manifest.get("result") == status, "Training completion/status mismatch")
    start, updates = status["start_step"], status["updates"]
    if "step3_train" in directory.parent.name:
        require(start == 0 and updates == 512, "Step-3 must finish its fixed 512-update budget")
    else:
        require(start == 512 and updates in {256, 512}, "Unexpected completed step-4 continuation budget")
    require(cfg["updates"] == updates and updates >= last_updates, "Invalid final audit window")
    logs = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines() if line.strip()]
    require([row["step"] for row in logs] == list(range(start + 1, start + updates + 1)), "Incomplete training step sequence")
    digest = hashlib.sha256()
    for row in logs:
        validate_sample(row["sample"], packet)
        digest.update(json.dumps(row["sample"], sort_keys=True).encode())
    require(digest.hexdigest() == status["schedule_sha256"], "Logged sample schedule does not match training status")
    sources = {}
    for filename in ("vector_vera.py", "stable_vector_vera.py"):
        path = directory.parent / "source/src/vera_mem" / filename
        checksum = sha(path)
        require(checksum == manifest["sources"][filename] == sha(REPO / "src/vera_mem" / filename),
                "Address implementation differs from the verified local normalization contract")
        sources[filename] = checksum
    checkpoint_path = directory / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    require(checkpoint.get("protocol") == "memory-interface-v1" and checkpoint["step"] == start + updates
            and checkpoint["seed"] == cfg["seed"] and checkpoint["method"] == cfg["method"], "Wrong final checkpoint")
    reader = FrozenAddress(checkpoint["architecture"], checkpoint["module"])
    events = [event for row in logs[-last_updates:] for event in episode_routes(reader, packet, row)]
    return dict(run_id=directory.parent.name + "/" + directory.name, seed=cfg["seed"], method=cfg["method"],
                key_consistency=cfg["key_consistency"], final_checkpoint_step=checkpoint["step"],
                replayed_steps=[logs[-last_updates]["step"], logs[-1]["step"]], replayed_updates=last_updates,
                proxy=True, no_full_sequence_gradient_claim=True, independent_single_target_replacement_verified=True,
                summary=summarize(events), per_exposure=events, sample_schedule_sha256=digest.hexdigest(),
                source_sha256=sources, checkpoint_sha256=sha(checkpoint_path),
                manifest_sha256=sha(directory / "manifest.json"), training_log_sha256=sha(directory / "training.jsonl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPO.parent / "runs/xtrah100")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag", default="20261006")
    parser.add_argument("--last-updates", type=int, default=128)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output is None:
        parser.error("--output is required")
    require(re.fullmatch(r"[0-9]{8}", args.tag), "Invalid experiment tag")
    require(args.last_updates == 128, "This fixed diagnostic replays the last 128 updates only")
    torch.set_num_threads(4)
    root = args.runs_root.resolve(strict=True)
    jobs, pending = discover(root, args.tag)
    cache = root / f"interface_extended_prepare_{args.tag}/prepare/train.pt"
    cache_hash = sha(cache)
    packet = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    validate_packet(packet)
    results = []
    for directory, manifest in jobs:
        result = audit_run(directory, manifest, packet, cache_hash, args.last_updates)
        results.append(result)
        print(json.dumps(dict(run=result["run_id"], proxy=True, **result["summary"]["rates"])), flush=True)
    report = dict(protocol=PROTOCOL, proxy=True, train_only=True, no_full_sequence_gradient_claim=True,
                  completed_runs=len(results), pending_training_jobs=pending, cache=str(cache), cache_sha256=cache_hash,
                  audited_window_updates=128, runs=results, limitations=NOTES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


def self_test():
    import sys
    import unittest
    sys.path.insert(0, str(REPO / "src"))
    from vera_mem.stable_vector_vera import StableVectorVeRA

    class Checks(unittest.TestCase):
        def fixture(self):
            generator = torch.Generator().manual_seed(9)
            module = StableVectorVeRA(5, 6, rank=3, key_dim=4, top_k=4)
            module.fit_statistics(torch.randn(20, 5, generator=generator), torch.randn(20, 5, generator=generator))
            packet = dict(protocol="memory-interface-v1", split="train", task="extended", model_revision=REVISION,
                          writer_prefix="Remember this information: ", qids=["q0", "q1"], sids=["s0", "s1", "s2"],
                          rows=[dict(id=f"id{i}", entity=f"e{i//2}", relation=f"r{i%2}", questions=["a", "b"],
                                     supports=[["x", "y", "z"], ["X", "Y", "Z"]]) for i in range(12)],
                          q=torch.randn(12, 2, 5, generator=generator), last=torch.randn(12, 2, 3, 5, generator=generator))
            episode = [11, 2, 7, 0, 3, 1, 8, 4, 6, 9, 5, 10]
            targets = list(range(8))
            sample = dict(targets=targets, episode=episode, columns=[episode.index(i) for i in targets],
                          qviews=[i % 2 for i in targets], sviews=[i % 3 for i in episode])
            return module, packet, dict(step=512, sample=sample)

        def test_cpu_address_proxy_matches_real_stable_encoders(self):
            module, packet, _ = self.fixture()
            reader = FrozenAddress(module.configuration(), module.state_dict())
            validate_packet(packet)
            for domain, operation, features in (("query", module.encode_query, packet["q"]),
                                                ("support", module.encode_key, packet["last"])):
                torch.testing.assert_close(reader.encode(features, domain), operation(features), rtol=0, atol=0)

        def test_each_query_replaces_its_own_target_and_matches_literal_individual_banks(self):
            module, packet, log = self.fixture()
            reader = FrozenAddress(module.configuration(), module.state_dict())
            events = episode_routes(reader, packet, log)
            sample = log["sample"]
            for row, event in enumerate(events):
                target, column = sample["targets"][row], sample["columns"][row]
                query = module.encode_query(packet["q"][target, sample["qviews"][row]])
                base = packet["last"][sample["episode"], 0, sample["sviews"]]
                alternate = base.clone()
                alternate[column] = packet["last"][target, 1, sample["sviews"][column]]
                for world, bank in (("a", base), ("b", alternate)):
                    keys = F.normalize(module.encode_key(bank), dim=-1, eps=1e-12)
                    selected = (query @ keys.T).topk(4).indices.tolist()
                    self.assertEqual(event[world + "_selected_ids"], [packet["rows"][sample["episode"][i]]["id"] for i in selected])
                    self.assertEqual(event[world + "_r4"], int(column in selected))
            report = summarize(events)
            self.assertEqual(report["exposures"], 8)
            self.assertEqual(report["counts"]["both_miss4"] + report["counts"]["either_hit4"], 8)

        def test_wrong_mapping_and_nontrain_packets_are_rejected(self):
            module, packet, log = self.fixture()
            packet["split"] = "confirm"
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                validate_packet(packet)
            packet["split"] = "train"
            log["sample"]["columns"][0] = log["sample"]["columns"][1]
            with self.assertRaisesRegex(ValueError, "mapping"):
                episode_routes(FrozenAddress(module.configuration(), module.state_dict()), packet, log)

        def test_unselected_value_has_no_direct_local_mixture_gradient_only(self):
            # This deliberately has no transformer/earlier-token paths. It
            # establishes only the local sparse-mixture mechanism in the note.
            values = torch.randn(8, 3, requires_grad=True)
            scores = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.], requires_grad=True)
            top_scores, selected = scores.topk(4)
            mixture = (top_scores.softmax(-1).unsqueeze(-1) * values[selected]).sum(0)
            mixture.square().sum().backward()
            self.assertEqual(values.grad[7].abs().sum(), 0)
            self.assertGreater(values.grad[selected].abs().sum(), 0)
            dense = F.cross_entropy(scores.unsqueeze(0), torch.tensor([7]))
            gradient = torch.autograd.grad(dense, scores)[0]
            self.assertNotEqual(gradient[7].item(), 0)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
