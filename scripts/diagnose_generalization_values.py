"""CPU development-only addressing, writer, and answer-information diagnostics.

All fitting uses offline training facts. Only development features are scored;
the cache's confirmation features/records are never indexed. Supervised answer
probes and answer-class prototypes are diagnostics, not the VeRA model, VDB, or
evidence of generative memory accuracy. No checkpoint is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from vera_mem.augmentation_data import protocol_fingerprint
from vera_mem.generalization_run import PROTOCOL, REVISION
from vera_mem.stable_vector_vera import StableVectorVeRA
from diagnose_vectors import distribution, matrix_diagnostics
from probe_features import atomic_json, sha256


def normalized(features):
    return F.normalize(features.float(), dim=-1, eps=1e-12) * math.sqrt(features.shape[-1])


def centered(features, center):
    values = normalized(features) - center
    return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)


@torch.no_grad()
def encode(features, function, batch_size=256):
    return torch.cat([function(features[start:start + batch_size])
                      for start in range(0, len(features), batch_size)])


@torch.no_grad()
def retrieval(query, keys):
    scores = query @ keys.T
    truth = torch.arange(len(query))
    ordered = scores.argsort(dim=1, descending=True)
    ranks = (ordered == truth[:, None]).nonzero()[:, 1] + 1
    incorrect = scores.clone()
    incorrect[truth, truth] = -torch.inf
    gold = scores[truth, truth]
    return dict(count=len(query), correct_at_1=int((ranks == 1).sum()),
                recall_at_1=float((ranks == 1).float().mean()),
                recall_at_4=float((ranks <= 4).float().mean()),
                mrr=float(ranks.float().reciprocal().mean()),
                gold_cosine=distribution(gold),
                gold_minus_best_incorrect_cosine=distribution(gold - incorrect.max(1).values))


@torch.no_grad()
def prototypes(train_values, labels, dev_values, dev_labels):
    result = {}
    for use_center in (False, True):
        mean = train_values.mean(0) if use_center else torch.zeros(train_values.shape[-1])
        proto = torch.stack([(train_values[labels == label] - mean).mean(0)
                             for label in range(16)])
        proto = F.normalize(proto, dim=-1)
        scores = F.normalize(dev_values - mean, dim=-1) @ proto.T
        truth = scores[torch.arange(len(dev_labels)), dev_labels]
        incorrect = scores.clone()
        incorrect[torch.arange(len(dev_labels)), dev_labels] = -torch.inf
        result["train_centered" if use_center else "raw_direction"] = dict(
            correct=int((scores.argmax(1) == dev_labels).sum()), count=len(dev_labels),
            accuracy=float((scores.argmax(1) == dev_labels).float().mean()),
            correct_prototype_cosine=distribution(truth),
            correct_minus_best_other_cosine=distribution(truth - incorrect.max(1).values),
            prototype_geometry=matrix_diagnostics(proto, 16),
        )
    return result


@torch.no_grad()
def expression_variance(value_views, labels):
    """Balanced two-factor ANOVA geometry; no new model or parameters."""
    # Every entity contributes one row to every active expression. Since each
    # answer class has equal entity counts, template and answer main effects
    # are orthogonal; their residual includes entity noise and interactions.
    array = torch.stack(value_views, dim=1).double()
    grand = array.mean((0, 1), keepdim=True)
    template_effect = array.mean(0, keepdim=True) - grand
    answer_means = torch.stack([array[labels == label].mean((0, 1)) for label in range(16)])
    answer_effect = answer_means[labels, None, :] - grand
    residual = array - grand - template_effect - answer_effect
    total = (array - grand).square().sum()
    template_ss = template_effect.square().sum() * len(array)
    answer_ss = answer_effect.square().sum() * array.shape[1]
    residual_ss = residual.square().sum()
    if not torch.allclose(template_ss + answer_ss + residual_ss, total, atol=1e-7, rtol=1e-7):
        raise AssertionError("Unbalanced/nonorthogonal template-answer variance decomposition")
    pairwise = {}
    for first in range(array.shape[1]):
        for second in range(first + 1, array.shape[1]):
            pairwise[f"view{first}_view{second}"] = distribution(
                F.cosine_similarity(array[:, first], array[:, second]))
    return dict(total_centered_energy=float(total),
                template_main_effect_fraction=float(template_ss / total),
                answer_main_effect_fraction=float(answer_ss / total),
                residual_entity_and_interaction_fraction=float(residual_ss / total),
                same_fact_cross_template_cosine=pairwise,
                note="Balanced training-fact/template variance decomposition; residual includes answer-template interactions, not only noise")


def linear_probe(train_features, train_labels, dev_features, dev_labels, steps=256, seed=42):
    """Fixed-budget linear classification; no dev optimization/selection."""
    started = time.perf_counter()
    center = torch.zeros(train_features.shape[-1])
    for start in range(0, len(train_features), 256):
        center += normalized(train_features[start:start + 256]).sum(0)
    center /= len(train_features)
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[-1], 16)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        # Sampling with replacement keeps exactly the same optimization budget
        # for canonical-only and multi-template feature populations.
        indices = torch.randint(len(train_features), (128,), generator=generator)
        x = centered(train_features[indices], center)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x), train_labels[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite linear diagnostic loss")
        loss.backward()
        optimizer.step()
        if (step + 1) % 64 == 0 or step + 1 == steps:
            losses.append(dict(step=step + 1, loss=float(loss.detach())))
    head.eval()
    scores = {}
    with torch.no_grad():
        for name, features in dev_features.items():
            logits = head(centered(features, center))
            prediction = logits.argmax(-1)
            scores[name] = dict(count=len(dev_labels), correct=int((prediction == dev_labels).sum()),
                                accuracy=float((prediction == dev_labels).float().mean()),
                                cross_entropy=float(F.cross_entropy(logits, dev_labels)),
                                prediction_counts=torch.bincount(prediction, minlength=16).tolist())
    return dict(steps=steps, batch_size=128, example_exposures=steps * 128, seed=seed,
                train_feature_rows=len(train_features), feature_dimension=train_features.shape[-1],
                parameter_count=sum(parameter.numel() for parameter in head.parameters()),
                optimizer="Adam", learning_rate=1e-3,
                selection="fixed final step; development never optimizes/selects the diagnostic head",
                normalization="training-only RMS-normalized feature center then centered RMS",
                dev_scores=scores, loss_history=losses, seconds=time.perf_counter() - started)


def restore(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["module"]
    module = StableVectorVeRA(state["A"].shape[1], state["B"].shape[0],
                              rank=state["A"].shape[0], key_dim=state["Wq.weight"].shape[0],
                              top_k=4, temperature=0.05)
    module.load_state_dict(state)
    module.requires_grad_(False).eval()
    return module, checkpoint


def diagnose_checkpoint(path, features, labels, forms, probe_steps):
    module, checkpoint = restore(path)
    train_count = checkpoint["config"]["train_size"]
    condition = checkpoint["config"]["condition"]
    train_s = features["train"]["s"][:train_count]
    train_labels = labels["train"][:train_count]
    dev_labels = labels["dev"]
    qdev = [encode(features["dev"]["q"][:, view], module.encode_query) for view in range(3)]
    kdev = [encode(features["dev"]["s"][:, view], module.encode_key) for view in range(3)]
    vdev = [encode(features["dev"]["s"][:, view], module.encode_value) for view in range(3)]
    vtrain = [encode(train_s[:, view], module.encode_value) for view in range(4)]
    active_train_views = [0] if condition == "canonical" else list(range(4))
    training_values = torch.cat([vtrain[view] for view in active_train_views])
    training_labels = train_labels.repeat(len(active_train_views))
    record = dict(checkpoint_name=path.parent.name + "/" + path.name,
                  checkpoint_sha256=sha256(path), condition=condition, selected_step=checkpoint["step"],
                  train_size=train_count, active_training_support_templates=[forms["train"]["s"][i] for i in active_train_views],
                  addressing={}, development_values={}, training_values=matrix_diagnostics(training_values, 1024),
                  training_value_variance_decomposition=expression_variance(
                      [vtrain[view] for view in active_train_views], train_labels),
                  decoder_b_l2_norm=float(module.b.norm()))
    for qi, si in ((0, 0), (1, 0), (2, 0), (1, 1), (2, 2)):
        record["addressing"][f"q{qi}_s{si}"] = dict(
            query_template=forms["dev"]["q"][qi], support_template=forms["dev"]["s"][si],
            **retrieval(qdev[qi], kdev[si]))
    for view in range(3):
        record["development_values"][forms["dev"]["s"][view]] = dict(
            geometry=matrix_diagnostics(vdev[view], 64),
            same_fact_canonical_cosine=distribution(F.cosine_similarity(vdev[view], vdev[0])),
            canonical_support_class_prototypes=prototypes(vtrain[0], train_labels, vdev[view], dev_labels),
            active_training_class_prototypes=prototypes(training_values, training_labels, vdev[view], dev_labels),
        )
    record["value_output_linear_probe"] = linear_probe(
        training_values, training_labels,
        {forms["dev"]["s"][view]: vdev[view] for view in range(3)}, dev_labels, steps=probe_steps)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-steps", type=int, default=256)
    args = parser.parse_args()
    if args.probe_steps < 1:
        parser.error("--probe-steps must be positive")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # mmap avoids paging in confirmation feature tensors. Restrict indexing to
    # train/dev before dropping the container holding unused split references.
    packet = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    if (packet["protocol"], packet["model_revision"], packet["data_fingerprint"]) != (
            PROTOCOL, REVISION, protocol_fingerprint()):
        raise ValueError("Unexpected cache provenance")
    classes = sorted({row["answer"] for row in packet["examples"]["train"]})
    if len(classes) != 16:
        raise ValueError("Expected the 16-word output vocabulary")
    class_id = {word: i for i, word in enumerate(classes)}
    labels = {split: torch.tensor([class_id[row["answer"]] for row in packet["examples"][split]])
              for split in ("train", "dev")}
    features = {split: packet["features"][split] for split in ("train", "dev")}
    forms = {split: packet["template_ids"][split] for split in ("train", "dev")}
    del packet
    output = dict(diagnostic="offline-development addressing and value-information; not generative accuracy",
                  complete=False, started_at=datetime.now(timezone.utc).isoformat(),
                  cache_sha256=sha256(args.cache), source_sha256=sha256(Path(__file__)),
                  helper_source_sha256={name: sha256(Path(__file__).parent / name)
                                        for name in ("diagnose_vectors.py", "probe_features.py")},
                  protocol=PROTOCOL, model_revision=REVISION, device="cpu", threads=4,
                  accessed_feature_splits=["train", "dev"], confirmation_features_accessed=False,
                  class_labels=classes, chance_class_accuracy=1 / 16,
                  caveats=[
                      "Answer-class prototypes and linear heads use training labels only and are auxiliary diagnostics.",
                      "Writer-output separability does not prove the fixed VeRA reader can decode these vectors.",
                      "Support features contain revealed answers; query features contain no answer labels.",
                      "Development contains 64 facts, with correlated template views and one seed.",
                      "No confirmation values, predictions, or labels are used to choose the next mechanism.",
                  ], checkpoints=[], frozen_support_input_linear_probes={})
    atomic_json(args.output, output)
    for path in args.checkpoint:
        record = diagnose_checkpoint(path, features, labels, forms, args.probe_steps)
        output["checkpoints"].append(record)
        atomic_json(args.output, output)
        print(json.dumps(dict(checkpoint=record["checkpoint_name"], condition=record["condition"],
                              addressing={key: value["recall_at_1"] for key, value in record["addressing"].items()},
                              value_probe=record["value_output_linear_probe"]["dev_scores"])), flush=True)
    for condition, views in (("canonical_support_training", [0]), ("augmented_support_training", list(range(4)))):
        train = torch.cat([features["train"]["s"][:, view] for view in views])
        record = linear_probe(train, labels["train"].repeat(len(views)),
                              {forms["dev"]["s"][view]: features["dev"]["s"][:, view]
                               for view in range(3)}, labels["dev"], steps=args.probe_steps)
        output["frozen_support_input_linear_probes"][condition] = record
        atomic_json(args.output, output)
        print(json.dumps(dict(input_probe=condition, development=record["dev_scores"])), flush=True)
        del train
        gc.collect()
    output.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output, output)


if __name__ == "__main__":
    main()
