"""Training-only fact-centric addressing ablation, scored on development facts.

This probe does not run the language model or read confirmation features. All
arms inherit the same canonical checkpoint and all-view training centers. The
all-view arms expose more feature vectors per fact than the sampled baseline;
they match fact count and updates, not floating-point operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vera_mem.factcentric_losses import style_block_retrieval_loss, canonical_anchor_loss
from vera_mem.stable_vector_vera import StableVectorVeRA

METHODS = ("sampled_pair", "all_view", "anchored_all_view")
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


@torch.no_grad()
def transform(raw, center, device):
    """Cache the exact StableVectorVeRA domain-input transformation."""
    flat = raw.reshape(-1, raw.shape[-1])
    output = torch.empty_like(flat, dtype=torch.float32, device=device)
    center = center.to(device)
    for start in range(0, len(flat), 512):
        value = F.normalize(flat[start:start+512].to(device).float(), dim=-1, eps=1e-12)
        value = value * math.sqrt(raw.shape[-1]) - center
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite normalized features")
        output[start:start+len(value)] = value
    return output.reshape(raw.shape)


@torch.no_grad()
def training_center(raw):
    # Equal counts in every view: averaging the view means gives the same
    # mathematical estimator as flattening all views, without huge temporaries.
    means = []
    for view in range(raw.shape[1]):
        normalized = F.normalize(raw[:, view].float(), dim=-1, eps=1e-12)
        means.append((normalized * math.sqrt(raw.shape[-1])).mean(0))
    center = torch.stack(means).mean(0)
    if not torch.isfinite(center).all():
        raise ValueError("Invalid training center")
    return center


def balanced_indices(labels, batch_size, rng):
    classes = sorted(set(labels))
    if batch_size % len(classes):
        raise ValueError("Batch size must divide into balanced answer groups")
    groups = {label: [i for i, item in enumerate(labels) if item == label] for label in classes}
    selected = [i for group in groups.values() for i in rng.sample(group, batch_size // len(classes))]
    rng.shuffle(selected)
    return selected


def encode(module, query, support):
    q = F.normalize(module.Wq(query), dim=-1, eps=1e-12)
    k = F.normalize(module.Wk(support), dim=-1, eps=1e-12)
    v = module.Wv(support)
    v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + module.value_epsilon)
    return q, k, v


def paired_consistency(q1, q2, k1, k2, v1, v2):
    return ((1 - (q1 * q2).sum(-1)).mean()
            + (1 - (k1 * k2).sum(-1)).mean() + F.mse_loss(v1, v2))


def retrieval_metrics(query, key):
    similarities = query @ key.T
    truth = torch.arange(len(query), device=query.device)
    hit1 = int((similarities.argmax(-1) == truth).sum())
    hit4 = int((similarities.topk(min(4, len(key)), -1).indices == truth[:, None]).any(-1).sum())
    return dict(count=len(truth), correct_at_1=hit1, correct_at_4=hit4,
                recall_at_1=hit1 / len(truth), recall_at_4=hit4 / len(truth),
                contrastive_nll=float(F.cross_entropy(similarities / .1, truth)))


@torch.no_grad()
def evaluate(module, dev, names, anchors, prototypes, labels):
    q, k, v = encode(module, dev["q"], dev["s"])
    retrieval = {}
    query_anchor = {}
    writer = {}
    for qi, qname in enumerate(names["q"]):
        query_anchor[qname] = retrieval_metrics(q[:, qi], anchors["k"])
        query_anchor[qname]["canonical_teacher_query_cosine"] = float((q[:, qi] * anchors["q"]).sum(-1).mean())
        for si, sname in enumerate(names["s"]):
            retrieval[qname + "/" + sname] = retrieval_metrics(q[:, qi], k[:, si])
    for si, sname in enumerate(names["s"]):
        value = F.normalize(v[:, si], dim=-1, eps=1e-12)
        prediction = (value @ prototypes.T).argmax(-1)
        writer[sname] = dict(
            count=len(v), correct_at_1=int((prediction == labels).sum()),
            teacher_value_prototype_accuracy=float((prediction == labels).float().mean()),
            canonical_teacher_value_cosine=float(F.cosine_similarity(v[:, si], anchors["v"], dim=-1).mean()),
            canonical_student_value_cosine=float(F.cosine_similarity(v[:, si], v[:, 0], dim=-1).mean()),
            key_teacher_cosine=float((k[:, si] * anchors["k"]).sum(-1).mean()),
            teacher_query_to_student_key=retrieval_metrics(anchors["q"], k[:, si]),
        )
    conditions = list(retrieval.values())
    return dict(retrieval=retrieval, query_to_canonical_teacher_bank=query_anchor,
                writer_compatibility=writer,
                macro_recall_at_1=sum(row["recall_at_1"] for row in conditions) / len(conditions),
                macro_recall_at_4=sum(row["recall_at_4"] for row in conditions) / len(conditions))


def make_student(teacher_state, centers, config, device):
    module = StableVectorVeRA(9728, 2560, rank=64, key_dim=64, top_k=4,
                             temperature=.05, seed=config["seed"])
    module.load_state_dict(teacher_state)
    with torch.no_grad():
        module.query_center.copy_(centers["q"])
        module.support_center.copy_(centers["s"])
    module.b.requires_grad_(False)
    return module.to(device)


def run(method, training, development, names, labels, teacher_state, centers,
        anchors, prototypes, dev_labels, config, output):
    started = time.perf_counter()
    module = make_student(teacher_state, centers, config, config["device"])
    parameters = [*module.Wq.parameters(), *module.Wk.parameters(), *module.Wv.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    fact_rng, view_rng = random.Random(config["seed"]+10), random.Random(config["seed"]+20)
    history = []
    exposure_hash = hashlib.sha256()
    for step in range(config["steps"]):
        indices = balanced_indices(labels, config["batch_size"], fact_rng)
        exposure_hash.update(json.dumps(indices, separators=(",", ":")).encode())
        index = torch.tensor(indices, device=config["device"])
        qviews = [[view_rng.randrange(training["q"].shape[1]) for _ in indices] for _ in range(2)]
        sviews = [[view_rng.randrange(training["s"].shape[1]) for _ in indices] for _ in range(2)]
        rows = torch.arange(len(indices), device=index.device)
        if method == "sampled_pair":
            # Each main-column key is a different fact, including other facts
            # with the same answer; query encoders are never given answer IDs.
            q1,k1,v1 = encode(module, training["q"][index,qviews[0]], training["s"][index,sviews[0]])
            q2,k2,v2 = encode(module, training["q"][index,qviews[1]], training["s"][index,sviews[1]])
            logits = q1 @ k1.T / .1
            retrieval = (F.cross_entropy(logits, rows) + F.cross_entropy(logits.T, rows)) / 2
            consistency = paired_consistency(q1,q2,k1,k2,v1,v2)
            anchor = retrieval.new_zeros(())
        else:
            q,k,v = encode(module, training["q"][index], training["s"][index])
            retrieval = style_block_retrieval_loss(q,k,temperature=.1)["retrieval"]
            consistency = paired_consistency(q[rows,qviews[0]],q[rows,qviews[1]],
                                             k[rows,sviews[0]],k[rows,sviews[1]],
                                             v[rows,sviews[0]],v[rows,sviews[1]])
            anchor = retrieval.new_zeros(())
            if method == "anchored_all_view":
                anchor = (canonical_anchor_loss(q,anchors["train"]["q"][index],metric="cosine")
                          + canonical_anchor_loss(k,anchors["train"]["k"][index],metric="cosine")
                          + F.mse_loss(v,anchors["train"]["v"][index,None,:].expand_as(v)))
        loss = retrieval + config["consistency_weight"] * consistency + anchor
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == config["steps"]:
            row = dict(step=step+1, loss=float(loss.detach()), retrieval_loss=float(retrieval.detach()),
                       consistency_loss=float(consistency.detach()), anchor_loss=float(anchor.detach()),
                       elapsed_seconds=time.perf_counter()-started)
            history.append(row)
            print(json.dumps(dict(method=method, training=row)), flush=True)
    metrics = evaluate(module,development,names,anchors["dev"],prototypes,dev_labels)
    weights = output.parent / (method + ".pt")
    torch.save(dict(module=module.cpu().state_dict(), config=config, method=method,
                    step=config["steps"], teacher_checkpoint_sha256=config["teacher_checkpoint_sha256"],
                    center_fit="all training query/support views, separate domains"),weights)
    result = dict(method=method,development=metrics,loss_history=history,
                  fixed_final_step=config["steps"], training_fact_exposures=config["steps"]*config["batch_size"],
                  encoded_query_vectors=config["steps"]*config["batch_size"]*(2 if method=="sampled_pair" else 8),
                  encoded_support_vectors=config["steps"]*config["batch_size"]*(2 if method=="sampled_pair" else 4),
                  fact_exposure_sha256=exposure_hash.hexdigest(), elapsed_seconds=time.perf_counter()-started,
                  checkpoint_filename=weights.name, checkpoint_sha256=digest(weights))
    print(json.dumps(dict(method=method,development=metrics)),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=1600)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--learning-rate",type=float,default=1e-4)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args()
    if args.steps<1 or args.batch_size<16 or args.batch_size>4096 or args.batch_size%16:
        parser.error("Use positive steps and a batch size between16 and4096 divisible by16")
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error("Learning rate must be finite and positive")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a probe result")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    packet = torch.load(args.cache,map_location="cpu",weights_only=True)
    if packet["protocol"]!="generalization-v1-layer20" or packet["model_revision"]!=REVISION:
        raise ValueError("Feature cache provenance mismatch")
    training,development = [packet["features"][split] for split in ("train","dev")]
    for domain,views in (("q",8),("s",4)):
        if tuple(training[domain].shape)!=(4096,views,9728) or tuple(development[domain].shape)!=(64,3,9728):
            raise ValueError("Unexpected training/development feature shape")
    names = packet["template_ids"]["dev"]
    labels = [row["answer"] for row in packet["examples"]["train"]]
    classes = sorted(set(labels))
    if len(classes)!=16:
        raise ValueError("Expected the declared 16-word training vocabulary")
    dev_labels = torch.tensor([classes.index(row["answer"]) for row in packet["examples"]["dev"]],device=args.device)
    checkpoint = torch.load(args.teacher,map_location="cpu",weights_only=True)
    if checkpoint["config"]["condition"]!="canonical" or checkpoint["config"]["train_size"]!=4096:
        raise ValueError("Teacher must be the canonical4096 checkpoint")
    teacher = StableVectorVeRA(9728,2560,rank=64,key_dim=64,top_k=4,temperature=.05,seed=42)
    teacher.load_state_dict(checkpoint["module"])
    teacher = teacher.to(args.device).requires_grad_(False).eval()
    anchors = {}
    with torch.no_grad():
        for split,raw in (("train",training),("dev",development)):
            anchors[split] = {
                "q":teacher.encode_query(raw["q"][:,0].to(args.device)),
                "k":teacher.encode_key(raw["s"][:,0].to(args.device)),
                "v":teacher.encode_value(raw["s"][:,0].to(args.device)),
            }
        prototypes = F.normalize(torch.stack([anchors["train"]["v"][[i for i,label in enumerate(labels) if label==word]].mean(0)
                                              for word in classes]),dim=-1,eps=1e-12)
    centers = {domain:training_center(training[domain]) for domain in ("q","s")}
    transformed = {split:{domain:transform(raw[domain],centers[domain],args.device) for domain in ("q","s")}
                   for split,raw in (("train",training),("dev",development))}
    config = dict(steps=args.steps,batch_size=args.batch_size,seed=args.seed,learning_rate=args.learning_rate,
                  device=args.device,training_facts=4096,query_views=8,support_views=4,
                  consistency_weight=.1,query_anchor_weight=1.,key_anchor_weight=1.,value_anchor_weight=1.,
                  contrastive_temperature=.1,clipping_norm=1.,optimizer="Adam",
                  all_view_objective="symmetric entity CE averaged over every query-style/support-style bank pair",
                  teacher_checkpoint_sha256=digest(args.teacher),teacher_selected_step=checkpoint["step"],
                  value_anchor_metric="MSE of RMS-normalized64-dimensional vectors",
                  balance="128 distinct facts:8 facts for each of16 answers; answer labels only choose negatives")
    root = Path(__file__).resolve().parents[1]
    result = dict(diagnostic="offline fact-centric addressing/writer compatibility; NOT language-model accuracy",
                  complete=False,started_at=datetime.now(timezone.utc).isoformat(),
                  cache_sha256=digest(args.cache),model_revision=packet["model_revision"],data_fingerprint=packet["data_fingerprint"],
                  source_script_sha256=digest(Path(__file__)),
                  module_source_sha256={name:digest(root/"src"/"vera_mem"/name) for name in
                                        ("factcentric_losses.py","stable_vector_vera.py","vector_vera.py")},
                  torch_version=str(torch.__version__),python_version=platform.python_version(),
                  device_name=torch.cuda.get_device_name(0) if args.device=="cuda" else platform.processor(),
                  torch_threads=4,config=config,eval_splits=["dev"],confirmation_used=False,dev_selection=False,
                  teacher_canonical_development=retrieval_metrics(anchors["dev"]["q"],anchors["dev"]["k"]),
                  limitations=[
                      "All arms use the same1536-update canonical LM-trained teacher initialization; no LM is optimized here.",
                      "Student all-view centers replace teacher canonical centers equally in every arm.",
                      "Same final updates and target facts do not match FLOPs: all-view arms encode8 query/4 support views versus2/2.",
                      "Only the two all-view arms match all encoded views; their difference isolates fixed canonical teacher anchoring.",
                      "Teacher targets for training use only canonical representations of training facts; development anchors are diagnostics only.",
                      "Balanced same-answer negatives encourage identity discrimination; this is not proof of natural-language semantic understanding.",
                      "Only prefill layer inputs are probed; no per-token decoding, residual readout, online writes, or answer EM is tested.",
                      "Development64 entities across9 crossed styles reuse entities; cells are not independent samples.",
                      "Existing development styles are reused; confirmation features/examples are never selected or evaluated.",
                  ],runs=[])
    del packet,training,development,teacher
    initial = make_student(checkpoint["module"],centers,config,args.device)
    result["shared_initial_development"] = evaluate(initial,transformed["dev"],names,anchors["dev"],prototypes,dev_labels)
    del initial
    write_json(args.output,result)
    for method in METHODS:
        result["runs"].append(run(method,transformed["train"],transformed["dev"],names,labels,
                                  checkpoint["module"],centers,anchors,prototypes,dev_labels,config,args.output))
        write_json(args.output,result)
    if len({run["fact_exposure_sha256"] for run in result["runs"]})!=1:
        raise AssertionError("Mismatched training fact exposures")
    result.update(complete=True,completed_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output,result)
    print("PROBE_COMPLETE "+str(args.output),flush=True)


if __name__=="__main__":
    main()
