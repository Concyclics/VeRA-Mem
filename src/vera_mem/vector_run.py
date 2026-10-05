"""Layer-input-addressed VDB values as dynamic VeRA scaling parameters.

Shared writer/query/reader weights are learned OFFLINE on separate entities.
ONLINE: each observed support produces a key/value write, with no optimizer
step. Every forward token queries the CPU vector store at the adapted layer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .backend import QwenBackend
from .data import synthetic_dataset, prepare_medmcqa
from .metrics import exact_match, token_f1, save_examples
from .run import aggregate, json_write, digest_file, tensor_digest
from .vector_vera import VectorVeRA
from .vector_store import PersistentVectorDB

METHODS = {"vdb_real", "vdb_oracle", "vdb_shuffled", "vdb_empty"}


def support_input(example):
    return "Remember this information: " + example.support


def collect_features(backend, examples):
    questions = torch.stack([backend.layer_feature(e.question) for e in examples]).to("cuda")
    supports = torch.stack([backend.layer_feature(support_input(e)) for e in examples]).to("cuda")
    return questions, supports


def new_module(backend, cfg):
    return VectorVeRA(backend.target.in_features, backend.target.out_features,
        rank=cfg["rank"], key_dim=cfg["key_dim"], top_k=cfg["top_k"],
        temperature=cfg["temperature"], seed=cfg["seed"],
        value_centering=cfg.get("value_centering", False)).to("cuda")


def train_offline(backend, cfg, output):
    module = new_module(backend, cfg)
    backend.vector_vera = module
    train = synthetic_dataset(cfg["seed"] + 1000, cfg["offline_train_size"], 0)["stream"]
    dev = synthetic_dataset(cfg["seed"] + 2000, cfg["offline_dev_size"], 0)["stream"]
    save_examples(output / "offline_train.jsonl", train)
    save_examples(output / "offline_dev.jsonl", dev)
    qt, st = collect_features(backend, train)
    qd, sd = collect_features(backend, dev)
    if cfg.get("value_centering", False):
        module.fit_value_center(st)
    # Addressing warm-up is trained solely on disjoint offline entities.
    optimizer = torch.optim.Adam(module.parameters(), lr=cfg["learning_rate"])
    alignment_log = []
    for step in range(cfg["alignment_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = module.contrastive_loss(qt, st, temperature=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        if (step+1) % 50 == 0:
            with torch.no_grad():
                scores = module.encode_query(qd) @ module.encode_key(sd).T
                hit = (scores.argmax(-1) == torch.arange(len(dev), device="cuda")).float().mean()
            row = {"step": step+1, "train_alignment_loss": float(loss.detach()), "dev_retrieval_at_1": float(hit)}
            alignment_log.append(row)
            print(json.dumps(row), flush=True)
    # b starts at zero; higher b LR avoids a near-zero residual throughout the
    # entire smoke run. Both learning rates are fixed before the held-out test.
    if cfg.get("freeze_addressing_after_alignment", False):
        module.Wq.requires_grad_(False)
        module.Wk.requires_grad_(False)
        module.zero_grad(set_to_none=True)
    other = [p for name, p in module.named_parameters() if name != "b" and p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": other, "lr": cfg["learning_rate"]},
        {"params": [module.b], "lr": cfg["output_scale_learning_rate"]}])
    rng = random.Random(cfg["seed"])
    best = float("inf")
    training_log = []
    for epoch in range(cfg["offline_epochs"]):
        order = list(range(len(train)))
        rng.shuffle(order)
        losses, alignment = [], []
        for target in order:
            others = rng.sample([i for i in range(len(train)) if i != target], min(cfg["episode_size"]-1, len(train)-1))
            episode = [target] + others
            rng.shuffle(episode)
            backend.mode = "vector_vera"
            backend.vector_store = None
            read_episode = [target] if cfg.get("oracle_reader_training", False) else episode
            backend.vector_keys = module.encode_key(st[read_episode])
            backend.vector_values = module.encode_value(st[read_episode])
            optimizer.zero_grad(set_to_none=True)
            lm_loss, _, _ = backend.loss(train[target].question, train[target].answer)
            address_loss = module.contrastive_loss(qt[episode], st[episode], temperature=0.1)
            total = lm_loss + cfg["alignment_weight"] * address_loss
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite offline objective")
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(lm_loss.detach()))
            alignment.append(float(address_loss.detach()))
        nll, tokens, oracle_nll = 0., 0, 0.
        with torch.no_grad():
            backend.vector_keys = module.encode_key(sd)
            backend.vector_values = module.encode_value(sd)
            for index, ex in enumerate(dev):
                score = backend.score(ex.question, ex.answer)
                nll += score.nll_sum
                tokens += score.tokens
                if cfg.get("oracle_reader_training", False):
                    all_keys, all_values = backend.vector_keys, backend.vector_values
                    backend.vector_keys, backend.vector_values = all_keys[index:index+1], all_values[index:index+1]
                    oracle_nll += backend.score(ex.question, ex.answer).nll_sum
                    backend.vector_keys, backend.vector_values = all_keys, all_values
            scores = module.encode_query(qd) @ backend.vector_keys.T
            hit1 = (scores.argmax(-1) == torch.arange(len(dev), device="cuda")).float().mean()
            hitk = (scores.topk(min(cfg["top_k"], len(dev)), dim=-1).indices == torch.arange(len(dev), device="cuda")[:,None]).any(-1).float().mean()
        row = {"epoch": epoch+1, "train_lm_loss_with_eos": float(np.mean(losses)),
            "train_alignment_loss": float(np.mean(alignment)), "dev_answer_token_nll": nll/tokens,
            "dev_retrieval_at_1": float(hit1), "dev_retrieval_at_k": float(hitk),
            "b_absolute_mean": float(module.b.detach().abs().mean())}
        if cfg.get("oracle_reader_training", False):
            row["dev_oracle_answer_token_nll"] = oracle_nll/tokens
        print(json.dumps({"offline_epoch": row}), flush=True)
        training_log.append(row)
        selection_metric = "dev_oracle_answer_token_nll" if cfg.get("oracle_reader_training", False) else "dev_answer_token_nll"
        if row[selection_metric] < best:
            best = row[selection_metric]
            torch.save({"module": module.state_dict(), "config": cfg, "selected_epoch": epoch+1}, output / "vector_vera.pt")
        json_write(output / "training.json", {"alignment": alignment_log, "epochs": training_log, "selection_metric": selection_metric})
    checkpoint = torch.load(output / "vector_vera.pt", weights_only=True, map_location="cuda")
    module.load_state_dict(checkpoint["module"])
    module.requires_grad_(False).eval()
    backend.mode, backend.vector_keys, backend.vector_values = "none", None, None
    return module


class OnlineExperiment:
    def __init__(self, backend, module, config, output):
        self.backend, self.module, self.cfg, self.output = backend, module, config, output

    def write(self, db, example, timestamp):
        # Full observation contains the newly revealed fact. No memory-modified
        # hidden states are used to prevent feedback contamination of new keys.
        start = time.perf_counter()
        hidden = self.backend.layer_feature(support_input(example)).to("cuda")
        with torch.no_grad():
            key = self.module.encode_key(hidden).cpu()
            value = self.module.encode_value(hidden).cpu()
        db.write(example.id, key, value, timestamp)
        return time.perf_counter() - start

    def evaluate(self, method, db, examples, phase, paraphrase=False):
        b = self.backend
        module_hash = tensor_digest(self.module)
        database_hash = db.hash()
        rows = []
        ids = list(db.ids)
        for ex in examples:
            b.mode, b.vector_vera, b.vector_store = "vector_vera", self.module, db
            b.vector_override = None
            if method == "vdb_empty":
                b.vector_override = torch.zeros(self.cfg["rank"], device="cuda")
            elif method == "vdb_oracle":
                b.vector_override = db.values[ids.index(ex.id)].to("cuda") if ex.id in ids else torch.zeros(self.cfg["rank"], device="cuda")
            elif method == "vdb_shuffled":
                # Address unchanged, values permuted by one row: wrong value
                # under the right key; the only altered variable is content.
                b.vector_store = shuffled_store(db)
            question = ex.paraphrase if paraphrase else ex.question
            b.prefill_retrieval = None
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, generated_tokens, generation_seconds = b.generate(question, max_new_tokens=self.cfg["max_new_tokens"])
            torch.cuda.synchronize()
            read_seconds = time.perf_counter() - started
            retrieval = b.prefill_retrieval
            selected = []
            if retrieval and "indices" in retrieval:
                index = retrieval["indices"]
                if index.numel():
                    selected = [ids[i] for i in index[0,-1].tolist()]
            score = b.score(question, ex.answer)
            row = {"id": ex.id, "phase": phase, "answer": ex.answer, "prediction": prediction,
                "em": float(exact_match(prediction, ex.answer)), "f1": float(token_f1(prediction, ex.answer)),
                "nll_sum": score.nll_sum, "answer_tokens": score.tokens,
                "generated_tokens": generated_tokens, "generation_seconds": generation_seconds,
                "read_seconds": read_seconds, "retrieved_ids_at_final_prompt_token": selected,
                "retrieved_id": (selected[0] if selected else (ex.id if method=="vdb_oracle" and ex.id in ids else None)),
                "expected_id": ex.id if ex.id in ids else None,
                "retrieval_hit_at_k": float(ex.id in selected) if ex.id in ids and method in ("vdb_real", "vdb_shuffled") else None}
            if ex.choices:
                choice, scores = b.choose(question)
                row.update(choice=choice, choice_nll=scores, choice_correct=float(choice == ex.answer))
            rows.append(row)
        if tensor_digest(self.module) != module_hash or db.hash() != database_hash:
            raise AssertionError("Read-only evaluation changed shared parameters or VDB")
        with (self.output / method / "predictions.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return rows

    def run(self, method, priming, stream, control, do_paraphrase=True):
        out = self.output / method
        out.mkdir()
        db = PersistentVectorDB(self.cfg["key_dim"], self.cfg["rank"],
            temperature=self.cfg["temperature"], top_k=self.cfg["top_k"])
        writes = []
        for i, ex in enumerate(priming):
            writes.append({"id": ex.id, "phase": "priming", "seconds": self.write(db, ex, i)})
        initial = self.evaluate(method, db, control, "control_before")
        prime_before = self.evaluate(method, db, priming, "priming_before")
        pre, immediate, blocks = [], [], []
        for i, ex in enumerate(stream):
            pre += self.evaluate(method, db, [ex], "pre_write")
            writes.append({"id": ex.id, "phase": "online", "seconds": self.write(db, ex, len(priming)+i)})
            immediate += self.evaluate(method, db, [ex], "immediate")
            if (i+1) % self.cfg["block_size"] == 0 or i == len(stream)-1:
                final = self.evaluate(method, db, stream[:i+1], f"block_{i+1}")
                block = {"after_writes": i+1, "aggregate": aggregate_vector(final),
                    "by_block": [aggregate_vector(final[j:j+self.cfg["block_size"]]) for j in range(0,i+1,self.cfg["block_size"])]}
                blocks.append(block)
                print(json.dumps({"method": method, **block}), flush=True)
                db.save(out / "vdb.pt")
        control_after = self.evaluate(method, db, control, "control_after")
        prime_after = self.evaluate(method, db, priming, "priming_after")
        para = self.evaluate(method, db, stream, "paraphrase", True) if do_paraphrase else []
        metrics = {"method": method, "seed": self.cfg["seed"],
            "training_seed": self.cfg["seed"], "evaluation_seed": self.cfg.get("eval_seed", self.cfg["seed"]),
            "pre_write": aggregate_vector(pre), "immediate": aggregate_vector(immediate),
            "final": aggregate_vector(final), "paraphrase": aggregate_vector(para),
            "priming_before": aggregate_vector(prime_before), "priming_after": aggregate_vector(prime_after),
            "control_before": aggregate_vector(initial), "control_after": aggregate_vector(control_after),
            "retention_matrix": blocks, "write_seconds": sum(x["seconds"] for x in writes),
            "offline_trainable_parameters": sum(p.numel() for p in self.module.parameters()),
            "online_gradient_steps": 0, "shared_parameters_frozen_online": True,
            "random_projection_buffer_bytes": sum(x.numel()*x.element_size() for x in (self.module.A, self.module.B)),
            "writer_statistics_buffer_bytes": self.module.value_center.numel()*self.module.value_center.element_size() if self.cfg.get("value_centering", False) else 0,
            "vdb_numeric_resident_bytes": db.resident_bytes(), "vdb_records": len(db.ids),
            "vdb_snapshot_sha256": digest_file(out / "vdb.pt"),
            "peak_gpu_allocated_bytes_process": torch.cuda.max_memory_allocated(),
            "read_query": "every token from actual down_proj input; CPU exact top-k VDB; copied sparse mixed value back to GPU",
            "timing_note": "single-device pilot; write feature extraction may be cached across controls; not production latency"}
        json_write(out / "metrics.json", metrics)
        json_write(out / "writes.json", writes)
        self.backend.mode, self.backend.vector_store = "none", None
        return metrics


def shuffled_store(db):
    result = PersistentVectorDB(db.key_dim, db.value_dim, temperature=db.temperature, top_k=db.top_k)
    values = db.values.roll(1, dims=0)
    for i, (identifier, key, value) in enumerate(zip(db.ids, db.keys, values)):
        result.write(identifier, key, value, i)
    return result


def aggregate_vector(rows):
    result = aggregate(rows)
    valid = [x["retrieval_hit_at_k"] for x in rows if x.get("retrieval_hit_at_k") is not None]
    if valid:
        result["retrieval_hit_at_k"] = sum(valid)/len(valid)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--dataset", choices=["synthetic", "medmcqa"], default="synthetic")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--methods", nargs="+")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.methods:
        cfg["methods"] = a.methods
    if not cfg["methods"] or set(cfg["methods"]) - METHODS:
        raise ValueError(f"Methods must be drawn from {sorted(METHODS)}")
    if len(cfg["methods"]) != len(set(cfg["methods"])):
        raise ValueError("Duplicate methods are not allowed")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    a.output.mkdir(parents=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    json_write(a.output / "config.json", cfg)
    if a.dataset == "synthetic":
        eval_seed = cfg.get("eval_seed", cfg["seed"])
        data = synthetic_dataset(eval_seed, cfg["stream_size"], cfg["control_size"])
        priming = synthetic_dataset(eval_seed+3000, cfg["priming_size"], 0)["stream"]
        groups = [data["stream"], data["control"], priming,
                  synthetic_dataset(cfg["seed"]+1000, cfg["offline_train_size"], 0)["stream"],
                  synthetic_dataset(cfg["seed"]+2000, cfg["offline_dev_size"], 0)["stream"]]
        all_ids = [ex.id for group in groups for ex in group]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Overlapping entity IDs between offline/online splits")
    else:
        data = prepare_medmcqa(a.data, cfg["seed"], cfg["priming_size"]+cfg["stream_size"], cfg["control_size"])
        priming, data["stream"] = data["stream"][:cfg["priming_size"]], data["stream"][cfg["priming_size"]:]
    for name, rows in dict(data, priming=priming).items():
        save_examples(a.output / (name + ".jsonl"), rows)
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "dataset": a.dataset,
        "training_seed": cfg["seed"], "evaluation_seed": cfg.get("eval_seed", cfg["seed"]),
        "model_id": cfg["model_id"], "hardware": torch.cuda.get_device_name(0),
        "packages": {x: importlib.metadata.version(x) for x in ["torch", "transformers", "numpy"]},
        "script_sha256": digest_file(__file__), "data_sha256": {n:digest_file(a.output/(n+'.jsonl')) for n in ['priming','stream','control']},
        "source_checkpoint": str(a.checkpoint) if a.checkpoint else "trained in this run",
        "protocol": "offline supervised support-to-key/value training; frozen shared weights online; observed support writes only"}
    metadata["source_files_sha256"] = {p.name: digest_file(p) for p in Path(__file__).parent.glob("*.py")}
    model_manifest = Path(a.model).parent / "manifest.json"
    if model_manifest.exists():
        metadata["model_revision"] = json.loads(model_manifest.read_text())["revision"]
    json_write(a.output / "manifest.json", metadata)
    b = QwenBackend(a.model, cfg["layer"], cfg["max_input_tokens"])
    if a.checkpoint:
        module = new_module(b, cfg)
        checkpoint = torch.load(a.checkpoint, map_location="cuda", weights_only=True)
        for field in ("model_id", "layer", "rank", "key_dim"):
            if checkpoint["config"][field] != cfg[field]:
                raise ValueError(f"Checkpoint {field} mismatch")
        if checkpoint["config"].get("value_centering", False) != cfg.get("value_centering", False):
            raise ValueError("Checkpoint value_centering mismatch")
        metadata["checkpoint_sha256"] = digest_file(a.checkpoint)
        module.load_state_dict(checkpoint["module"])
        module.requires_grad_(False).eval()
    else:
        module = train_offline(b, cfg, a.output)
    experiment = OnlineExperiment(b, module, cfg, a.output)
    summary = []
    for method in cfg["methods"]:
        print("METHOD_START " + method, flush=True)
        summary.append(experiment.run(method, priming, data["stream"], data["control"], a.dataset=="synthetic"))
        json_write(a.output / "summary.json", summary)
    metadata.update(completed_at=datetime.now(timezone.utc).isoformat(), complete=True)
    json_write(a.output / "manifest.json", metadata)
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
