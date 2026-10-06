"""Audit real evaluator artifacts with a CPU backend; no downloaded models.

The fixture deliberately uses evaluate_counterfactual for its predictions,
metrics, writes, and binary bank reconstruction packets. Only training logs
are synthesized, following counterfactual_run.train/paired_step verbatim.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vera_mem.counterfactual_data import datasets
from vera_mem.counterfactual_eval import evaluate_counterfactual
from vera_mem.data import MEMORY_WORDS
from vera_mem.run import tensor_digest


ROOT = Path(__file__).resolve().parents[1]
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ARMS = ("base", "behavior", "hidden", "mixed")
PHASES = ("canonical_support/canonical_query", "canonical_support/heldout_query",
          "heldout_support/canonical_query", "heldout_support/heldout_query")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def summary():
    return load_script("summarize_counterfactual")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def mutate_metrics(directory, change):
    """Keep duplicated metrics consistent; require independent raw auditing."""
    mutate_json(directory/"metrics.json", change)
    updated = json.loads((directory/"metrics.json").read_text())
    mutate_json(directory/"manifest.json", lambda value: value.__setitem__("evaluation", updated))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TinyMemory(nn.Module):
    """64-dimensional deterministic test writer matching formal bank sizes."""
    rank = key_dim = 64
    top_k, temperature = 4, .05

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def encode_key(self, features):
        return features[..., :64] * self.scale

    def encode_value(self, features):
        return features[..., 64:] * self.scale


class CpuBackend:
    device = torch.device("cpu")

    def __init__(self, module, evaluation):
        self.model = nn.Linear(1, 1).requires_grad_(False)
        self.vector_vera = module
        self.questions, self.contexts, self.answers = {}, {}, {}
        for packet in evaluation["phases"].values():
            for index, case in enumerate(packet["cases"]):
                self.questions[case["question"]] = index
                self.answers[index] = (case["answer_a"], case["answer_b"])
                for world, label in (("a", "a"), ("b", "b"), ("p", "a")):
                    self.contexts[case["support_"+world]] = case["answer_"+label]

    def generate(self, question, max_new_tokens, **kwargs):
        assert max_new_tokens == 4 and not torch.is_grad_enabled()
        if self.mode == "none":
            prediction = self.contexts[kwargs["context"]]
            index = self.questions[question]
            if index == 2 and prediction == self.answers[index][1]:
                prediction = "PRIVATE_TEACHER_WRONG_B"
        else:
            assert not kwargs
            query = torch.eye(64)[self.questions[question]].reshape(1, 1, 64)
            found = self.vector_store.search(query)
            if self.vector_override is None:
                selected = found["indices"][:, -1]
                self.retrieval_trace.extend([
                    {"phase": "prefill", "indices": selected},
                    {"phase": "decode", "indices": selected},
                ])
                value = found["mixed_value"].reshape(64)
            else:
                value = self.vector_override
            prediction = MEMORY_WORDS[int(value.argmax())]
            # Two controlled failures distinguish answer swaps from arbitrary
            # wrong-to-wrong changes. All branches still use real bank reads.
            index = self.questions[question]
            a, b = self.answers[index]
            if index == 0 and prediction in (a, b):
                prediction = b if prediction == a else a
            elif index == 1:
                prediction = "PRIVATE_WRONG_"+prediction
        return prediction, 2, .001

    def score(self, question, answer, **kwargs):
        return SimpleNamespace(nll_sum=2., tokens=2)


def evaluation_packet():
    packet = datasets()
    phases = load_script("prepare_counterfactual").make_evaluation_cases(packet)
    for phase in phases.values():
        features = torch.zeros(64, 3, 128)
        for index, case in enumerate(phase["cases"]):
            for view, label in enumerate(("a", "b", "a")):
                features[index, view, index] = 1.
                features[index, view, 64+MEMORY_WORDS.index(case["answer_"+label])] = 1.
        phase["supports"] = features
    return dict(phases=phases)


def training_artifacts(directory, method, start_step, module, checkpoint_hash, sources):
    directory.mkdir()
    cfg = dict(model="/private/model", cache="/private/train.pt", checkpoint="/private/checkpoint.pt",
               run_dir=str(directory), method=method, updates=4, start_step=start_step,
               batch_size=2, seed=42, evaluate_only=False, teacher=False)
    rows, trajectories, total = [], [], {}
    schedule = hashlib.sha256()
    for local_step in range(1, 5):
        step = start_step+local_step
        on = method == "mixed" and start_step > 0 and local_step % 4 == 0
        exposure = dict(target_ids=["PRIVATE_TRAIN_A", "PRIVATE_TRAIN_B"],
                        episode_ids=["PRIVATE_TRAIN_A", "PRIVATE_TRAIN_B", "PRIVATE_NEGATIVE"],
                        query_views=[step % 8, 0, 2], support_views=[step % 4, 0, 1], columns=[0, 1])
        schedule.update(json.dumps(exposure, sort_keys=True, separators=(",", ":")).encode())
        cost = dict(student_input_tokens=120, teacher_input_tokens=180, target_tokens=12,
                    sampled_tokens=4 if on else 0, rollout_input_tokens=80 if on else 0,
                    rollout_forward_calls=4 if on else 0, replay_input_tokens=40 if step % 4 == 0 else 0,
                    replay_target_tokens=4 if step % 4 == 0 else 0)
        row = dict(step=step, local_step=local_step, on_policy=on, **exposure, **cost,
                   gradient_norms=[.1, .1, .1], elapsed_seconds=float(local_step),
                   main_loss=1., total_loss=1., forward_kl=.5, first_token_ce=.5,
                   actual_address_loss=.5, style_address_loss=.5, behavior_loss=.5,
                   hidden_loss=.5, paraphrase_loss=.5, replay_loss=.5 if step % 4 == 0 else 0.,
                   behavior_valid_pairs=2, behavior_clipped_pairs=0,
                   teacher_behavior_delta_mean=2., student_behavior_delta_mean=1.,
                   teacher_first_token_joint_correct=2, student_first_token_joint_correct=1,
                   hidden_valid_pairs=2, teacher_delta_norm_mean=2., student_delta_norm_mean=1.)
        rows.append(row)
        for key, value in cost.items():
            total[key] = total.get(key, 0)+value
        for identity in exposure["target_ids"]:
            gold = [[101, 2], [102, 2], [101, 2]]
            continued = [[101, 2]]*3 if on else gold
            trajectories.append(dict(step=step, id=identity, on_policy=on,
                question="PRIVATE_TRAIN_PROMPT", contexts=["PRIVATE_A", "PRIVATE_B", "PRIVATE_P"],
                continuation_token_ids=continued, gold_token_ids=gold, sampled_world=0 if on else None))
    write_jsonl(directory/"training.jsonl", rows)
    write_jsonl(directory/"trajectories.jsonl", trajectories)
    status = dict(complete=True, step=start_step+4, updates=4, target_pair_exposures=8,
                  episode_schedule_sha256=schedule.hexdigest(), token_totals=total, elapsed_seconds=4.)
    write_json(directory/"training_status.json", status)
    torch.save(dict(protocol="counterfactual-context-v1", module=module.state_dict(), optimizer={},
                    step=start_step+4, configuration=cfg), directory/"last.pt")
    manifest = common_manifest(cfg, sources, module, checkpoint_hash, "train")
    manifest["training"] = status
    write_json(directory/"manifest.json", manifest)


def common_manifest(cfg, sources, module, checkpoint_hash, split):
    return dict(protocol="counterfactual-context-v1", complete=True, configuration=cfg,
        source_files_sha256=sources, cache_sha256=("1" if split == "train" else "2")*64,
        checkpoint_sha256=checkpoint_hash, model_revision=REVISION,
        teacher_parameter_sha256_before="3"*64, teacher_parameter_sha256_after="3"*64,
        teacher_parameters_unchanged=True, module_sha256_before=tensor_digest(module),
        module_sha256_after=tensor_digest(module), cache_split=split,
        started_at="2026-10-06T00:00:00Z", finished_at="2026-10-06T00:00:01Z")


@pytest.fixture(scope="module")
def original_fixture(tmp_path_factory):
    """Generate raw schema once, then isolate each corruption test by copying."""
    root = tmp_path_factory.mktemp("counterfactual_evidence")
    source = root/"source/src/vera_mem"
    shutil.copytree(ROOT/"src/vera_mem", source, ignore=shutil.ignore_patterns("__pycache__"))
    sources = {path.name: digest(path) for path in source.glob("*.py")}
    module, packet = TinyMemory(), evaluation_packet()
    training_artifacts(root/"warm", "base", 0, module, "4"*64, sources)
    for arm in ARMS:
        training_artifacts(root/(arm+"_train"), arm, 4, module, digest(root/"warm/last.pt"), sources)
    for arm in ("initial", "base"):
        directory = root/(arm+"_eval")
        result = evaluate_counterfactual(CpuBackend(module, packet), module, packet, directory,
                                        include_teacher=arm == "initial", max_new_tokens=4)
        cfg = dict(model="/private/model", cache="/private/confirmation.pt", checkpoint="/private/checkpoint.pt",
                   run_dir=str(directory), method="base", updates=4, start_step=4,
                   batch_size=2, seed=42, evaluate_only=True, teacher=arm == "initial")
        ckpt_hash = "4"*64 if arm == "initial" else digest(root/"base_train/last.pt")
        manifest = common_manifest(cfg, sources, module, ckpt_hash, "confirmation")
        manifest["evaluation"] = result
        write_json(directory/"manifest.json", manifest)
    for arm in ARMS[1:]:
        directory = root/(arm+"_eval")
        shutil.copytree(root/"base_eval", directory)
        def change(manifest):
            manifest["configuration"].update(method=arm, run_dir=str(directory))
            manifest["checkpoint_sha256"] = digest(root/(arm+"_train")/"last.pt")
        mutate_json(directory/"manifest.json", change)
    return root


@pytest.fixture
def evidence(tmp_path, original_fixture):
    destination = tmp_path/"PRIVATE_WORKSPACE"
    shutil.copytree(original_fixture, destination)
    return destination


def summarize(summary, root, samples=40):
    return summary.summarize(initial=root/"initial_eval", warm=root/"warm",
        trainings={arm: root/(arm+"_train") for arm in ARMS},
        evaluations={arm: root/(arm+"_eval") for arm in ARMS}, samples=samples, seed=123)


def test_real_evaluator_end_to_end_and_public_privacy(summary, evidence):
    protected = [evidence/"base_eval/predictions.jsonl", evidence/"base_eval/metrics.json",
                 evidence/"base_eval/banks/phase_00.pt", evidence/"base_train/training.jsonl"]
    before = {path: digest(path) for path in protected}
    result = summarize(summary, evidence)
    assert {path: digest(path) for path in protected} == before
    assert result["complete"]
    assert result["evaluation"]["facts"] == 64
    assert set(result["evaluation"]["phases"]) == set(PHASES)
    assert len(result["runs"]) == 5
    base = next(run for run in result["runs"] if run["method"] == "base")
    for phase in PHASES:
        actual = base["phases"][phase]
        assert actual["methods"]["real"]["paired_switch_em"] == 62/64
        assert actual["methods"]["real"]["changed_but_not_both_correct_rate"] == 2/64
        assert actual["methods"]["real"]["swapped_answer_count"] == 1
        assert actual["methods"]["real"]["swapped_answer_rate"] == 1/64
        # A swapped answer is also a both-wrong pair, not a disjoint category.
        assert actual["methods"]["real"]["wrong_wrong_count"] == 2
        assert actual["paraphrase"]["joint_em"] == 62/64
        assert actual["methods"]["empty"]["paired_switch_em"] == 0.
        eligible = actual["teacher_eligibility"]
        assert eligible["teacher_ab_eligible_count"] == 63
        assert eligible["teacher_ap_eligible_count"] == 64
        assert eligible["student_real_switch_on_teacher_ab_eligible_correct"] == 61
        assert eligible["student_real_switch_on_teacher_ab_eligible"] == 61/63
    assert base["clustered_real_minus_shuffled_switch"]["n_paired_facts"] == 64
    assert base["clustered_real_minus_shuffled_switch"]["phases_per_fact"] == 4
    serialized = json.dumps(result)+summary.report(result)
    for private in ("PRIVATE_", "/private/", str(evidence), "continuation_token_ids", "teacher_context"):
        assert private not in serialized
    first_row = read_jsonl(evidence/"base_eval/predictions.jsonl")[0]
    assert first_row["target_id"] not in serialized
    assert first_row["question"] not in serialized


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "false_em", "false_nll", "read_mutation", "fabricated_bank", "context_leak", "query_identity"])
def test_corrupt_raw_predictions_are_rejected(summary, evidence, mutation):
    path = evidence/"base_eval/predictions.jsonl"
    rows = read_jsonl(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "false_em":
        rows[0]["answer_results"]["A"]["em"] = 1-rows[0]["answer_results"]["A"]["em"]
    elif mutation == "false_nll":
        rows[0]["answer_results"]["A"]["nll_sum"] = -1.
    elif mutation == "read_mutation":
        rows[0]["bank_hash_after"] = "f"*64
    elif mutation == "fabricated_bank":
        rows[0]["bank_hash_before"] = rows[0]["bank_hash_after"] = "f"*64
    elif mutation == "context_leak":
        rows[0]["teacher_context"] = "PRIVATE_LEAKED_SUPPORT"
    else:
        rows[0]["query_id"] = "PRIVATE_WRONG_ENTITY"
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        summarize(summary, evidence)


@pytest.mark.parametrize("field", ["paired_switch_em", "changed_rate", "a_correct", "a_recall_at_4",
                                  "a_answer_token_nll", "a_decode_correct_residency"])
def test_reported_pair_statistics_are_recomputed(summary, evidence, field):
    mutate_metrics(evidence/"base_eval", lambda value: value["phases"][PHASES[0]]["methods"]["real"].__setitem__(field, -7.))
    with pytest.raises(ValueError):
        summarize(summary, evidence)


@pytest.mark.parametrize("where", ["paraphrase", "unrelated"])
def test_invariance_metrics_are_audited_not_only_answer_switch(summary, evidence, where):
    mutate_metrics(evidence/"base_eval", lambda value: value["phases"][PHASES[0]][where].__setitem__("joint_em", 0.))
    with pytest.raises(ValueError):
        summarize(summary, evidence)


@pytest.mark.parametrize("field,value", [("complete", False), ("teacher_parameters_unchanged", False),
    ("teacher_parameter_sha256_after", "e"*64), ("checkpoint_sha256", "e"*64),
    ("cache_sha256", "e"*64), ("cache_split", "train"), ("module_sha256_before", "e"*64)])
def test_eval_provenance_and_frozen_teacher_are_required(summary, evidence, field, value):
    mutate_json(evidence/"base_eval/manifest.json", lambda data: data.__setitem__(field, value))
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_changed_warm_checkpoint_breaks_every_continuation_link(summary, evidence):
    (evidence/"warm/last.pt").write_bytes(b"different warm checkpoint")
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_actual_four_fact_smoke_evaluation_cannot_enter_formal_summary(summary, evidence):
    directory = evidence/"initial_eval"
    manifest = json.loads((directory/"manifest.json").read_text())
    packet = evaluation_packet()
    for phase in packet["phases"].values():
        phase["cases"] = phase["cases"][:4]
        phase["supports"] = phase["supports"][:4]
    module = TinyMemory()
    shutil.rmtree(directory)
    manifest["evaluation"] = evaluate_counterfactual(CpuBackend(module, packet), module, packet,
        directory, include_teacher=True, max_new_tokens=4)
    write_json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_source_snapshot_cannot_change_after_recorded_hash(summary, evidence):
    with (evidence/"source/src/vera_mem/counterfactual_losses.py").open("a") as handle:
        handle.write("\n# changed after experiment\n")
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_independent_bank_intervention_hash_is_replayed(summary, evidence):
    path = evidence/"base_eval/banks/phase_00.pt"
    packet = torch.load(path, weights_only=True)
    packet["interventions"][0]["B"]["value"][0] += 1.
    torch.save(packet, path)
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_write_support_hash_must_match_evidence(summary, evidence):
    path = evidence/"base_eval/writes.jsonl"
    rows = read_jsonl(path)
    rows[0]["support"] += " corrupted"
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        summarize(summary, evidence)


@pytest.mark.parametrize("rehash", [False, True])
def test_changed_training_schedule_is_detected(summary, evidence, rehash):
    path = evidence/"hidden_train/training.jsonl"
    rows = read_jsonl(path)
    rows[0]["query_views"][0] = (rows[0]["query_views"][0]+1) % 8
    write_jsonl(path, rows)
    if rehash:
        schedule = hashlib.sha256()
        for row in rows:
            exposure = {key: row[key] for key in ("target_ids", "episode_ids", "query_views", "support_views", "columns")}
            schedule.update(json.dumps(exposure, sort_keys=True, separators=(",", ":")).encode())
        mutate_json(evidence/"hidden_train/training_status.json", lambda value: value.__setitem__("episode_schedule_sha256", schedule.hexdigest()))
        mutate_json(evidence/"hidden_train/manifest.json", lambda value: value["training"].__setitem__("episode_schedule_sha256", schedule.hexdigest()))
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_training_token_costs_are_recomputed_from_logs(summary, evidence):
    mutate_json(evidence/"mixed_train/training_status.json", lambda value: value["token_totals"].__setitem__("sampled_tokens", 0))
    mutate_json(evidence/"mixed_train/manifest.json", lambda value: value["training"]["token_totals"].__setitem__("sampled_tokens", 0))
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_on_policy_trajectories_must_share_prefix_across_worlds(summary, evidence):
    path = evidence/"mixed_train/trajectories.jsonl"
    rows = read_jsonl(path)
    row = next(row for row in rows if row["on_policy"])
    row["continuation_token_ids"][1][0] = 999
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        summarize(summary, evidence)


def test_cluster_bootstrap_uses_facts_not_phase_observations(summary):
    first = {"fact-a": [1., 0., 1., 0.], "fact-b": [0., 1., 0., 1.]}
    second = {key: [0.]*4 for key in first}
    result = summary.paired_cluster_bootstrap(first, second, samples=300, seed=7)
    assert result["n_paired_facts"] == 2
    assert result["phases_per_fact"] == 4
    assert result["delta"] == .5
    # Every fact mean is .5. Sampling phase rows independently would invent variance.
    assert result["ci95"] == [.5, .5]


@pytest.mark.parametrize("first,second", [
    ({"a": [1.]}, {"b": [1.]}),
    ({"a": [1., 0.]}, {"a": [1.]}),
    ({"a": [float("nan")]}, {"a": [1.]}),
])
def test_bootstrap_rejects_misaligned_or_nonfinite_facts(summary, first, second):
    with pytest.raises(ValueError):
        summary.paired_cluster_bootstrap(first, second, samples=10)
