"""End-to-end small-budget, supervised read-before-write memory experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from .backend import QwenBackend
from .data import synthetic_dataset, prepare_medmcqa
from .metrics import exact_match, token_f1, save_examples, load_examples
from .memory import LoRABank, LatentResidual, ExactMemory, FrozenRouter, LexicalMemory


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def aggregate(rows):
    if not rows:
        return {"n": 0}
    count = len(rows)
    total_tokens = sum(x["answer_tokens"] for x in rows)
    token_nll = sum(x["nll_sum"] for x in rows) / max(total_tokens, 1)
    result = {"n": count, "em": sum(x["em"] for x in rows) / count,
        "f1": sum(x["f1"] for x in rows) / count, "answer_token_nll": token_nll,
        "answer_token_ppl": math.exp(min(token_nll, 700)),
        "read_seconds_median": float(np.median([x["read_seconds"] for x in rows])),
        "read_seconds_p95": float(np.quantile([x["read_seconds"] for x in rows], .95)),
        "generated_tokens_per_second": sum(x["generated_tokens"] for x in rows) / max(sum(x["generation_seconds"] for x in rows), 1e-9)}
    candidate = [x for x in rows if "choice_correct" in x]
    if candidate:
        result["choice_accuracy"] = sum(x["choice_correct"] for x in candidate) / len(candidate)
        result["invalid_letter_rate"] = sum(x["prediction"].strip().upper() not in list("ABCD") for x in candidate) / len(candidate)
    retrieval = [x for x in rows if x.get("expected_id") is not None]
    if retrieval:
        result["retrieval_hit_at_1"] = sum(x["retrieved_id"] == x["expected_id"] for x in retrieval) / len(retrieval)
    return result


class Experiment:
    def __init__(self, backend, cfg, output, dataset, reader=None):
        self.backend = backend
        self.cfg = cfg
        self.output = output
        self.dataset = dataset
        self.reader = reader
        self.features = {}

    def feature(self, question):
        return self.backend.feature(question)

    def prepare_read(self, method, example, bank, router, memory, lexical, observed, paraphrase=False):
        b = self.backend
        b.mode, b.value = "none", None
        question = example.paraphrase if paraphrase else example.question
        context, retrieved_id = None, None
        if method.startswith("lora"):
            b.bank = bank
            b.slot = router.route(self.feature(question)) if router else 0
            b.mode = "lora"
        elif method == "text_oracle":
            # Label/id-based lookup is explicitly oracle; never called a real router.
            if example.id in observed:
                context, retrieved_id = observed[example.id].support, example.id
        elif method == "text_tfidf":
            found = lexical.retrieve(question, top_k=1)
            if found:
                context, retrieved_id = found[0]["text"], found[0]["id"]
        elif method.startswith("latent"):
            b.reader, b.mode = self.reader, "latent"
            if method == "latent_oracle" and example.id in observed:
                retrieved_id = example.id
            elif method == "latent_shuffled" and observed:
                # Deterministic WRONG memory when >=2 available, no accidental self.
                candidates = sorted(x for x in observed if x != example.id)
                retrieved_id = candidates[0] if candidates else None
            elif method == "latent_knn":
                found = memory.retrieve(self.feature(question), top_k=1)
                if found:
                    b.value = found[0]["value"].to("cuda")
                    retrieved_id = found[0]["id"]
            if retrieved_id is not None and method in ("latent_oracle", "latent_shuffled"):
                b.value = self.values[retrieved_id].to("cuda")
        return question, context, retrieved_id

    def evaluate(self, method, examples, phase, bank, router, memory, lexical, observed, paraphrase=False):
        b = self.backend
        before = tensor_digest(bank) if bank is not None else None
        reader_before = tensor_digest(self.reader) if self.reader is not None else None
        rows = []
        for ex in examples:
            torch.cuda.synchronize()
            start = time.perf_counter()
            question, context, retrieved = self.prepare_read(method, ex, bank, router, memory, lexical, observed, paraphrase)
            prediction, tokens, generation_seconds = b.generate(question, context, self.cfg["max_new_tokens"])
            # Read time includes routing / retrieval plus generation, excludes
            # teacher-forced scoring, which is an offline evaluator operation.
            torch.cuda.synchronize()
            read_seconds = time.perf_counter() - start
            score = b.score(question, ex.answer, context)
            row = {"id": ex.id, "phase": phase, "answer": ex.answer, "prediction": prediction,
                "em": float(exact_match(prediction, ex.answer)), "f1": float(token_f1(prediction, ex.answer)),
                "nll_sum": score.nll_sum, "answer_tokens": score.tokens,
                "generated_tokens": tokens, "generation_seconds": generation_seconds, "read_seconds": read_seconds,
                "retrieved_id": retrieved, "expected_id": ex.id if ex.id in observed and not method.startswith("lora") and method != "frozen" else None}
            if ex.choices:
                choice, scores = b.choose(question, context)
                row.update(choice=choice, choice_nll=scores, choice_correct=float(choice == ex.answer))
            rows.append(row)
        if bank is not None and before != tensor_digest(bank):
            raise AssertionError("Evaluation mutated adapter parameters")
        if self.reader is not None and reader_before != tensor_digest(self.reader):
            raise AssertionError("Evaluation mutated the frozen reader")
        with (self.output / method / "predictions.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows

    def write_lora(self, ex, bank, router, optimizers):
        b = self.backend
        slot = router.route(self.feature(ex.question)) if router else 0
        b.bank, b.slot, b.mode = bank, slot, "lora"
        optimizer = optimizers[slot]
        losses = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.cfg["write_steps"]):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = b.loss(ex.question, ex.answer, include_eos=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite supervised write loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters_for_slot(slot), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        return {"id": ex.id, "slot": slot, "losses": losses, "seconds": time.perf_counter() - start}

    def run_method(self, method, priming, stream, control):
        out = self.output / method
        out.mkdir()
        cfg, b = self.cfg, self.backend
        torch.manual_seed(cfg["seed"])
        bank, router, optimizers = None, None, None
        memory, lexical, observed = ExactMemory(), LexicalMemory(), {}
        self.values = {}
        if method.startswith("lora"):
            slots = int(method.split("_")[0].removeprefix("lora"))
            rank = int(method.split("_r")[1])
            bank = LoRABank(b.target.in_features, b.target.out_features, rank=rank, slots=slots, seed=cfg["seed"]).to("cuda")
            if slots > 1:
                router = FrozenRouter(slots=slots, seed=cfg["seed"])
                router.fit(torch.stack([self.feature(x.question) for x in priming]))
            optimizers = [torch.optim.Adam(bank.parameters_for_slot(s), lr=cfg["learning_rate"]) for s in range(slots)]

        def write(ex, timestamp):
            observed[ex.id] = ex
            if bank is not None:
                return self.write_lora(ex, bank, router, optimizers)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if method == "text_tfidf":
                lexical.write(ex.id, ex.support, timestamp)
            elif method.startswith("latent") and method != "latent_empty":
                with torch.no_grad():
                    value = self.reader.encode(self.feature("Remember this information: " + ex.support).to("cuda")).detach().cpu()
                self.values[ex.id] = value
                memory.write(ex.id, self.feature(ex.question), value, timestamp)
            torch.cuda.synchronize()
            return {"id": ex.id, "seconds": time.perf_counter() - started}

        updates = []
        for i, ex in enumerate(priming):
            if method != "frozen":
                updates.append(dict(phase="priming", **write(ex, i)))
        eval_args = (bank, router, memory, lexical, observed)
        initial = self.evaluate(method, control, "control_before", *eval_args)
        priming_before = self.evaluate(method, priming, "priming_before", *eval_args)
        pre_rows, immediate_rows, block_metrics = [], [], []
        for i, ex in enumerate(stream):
            pre_rows += self.evaluate(method, [ex], "pre_write", *eval_args)
            if method != "frozen":
                updates.append(dict(phase="online", **write(ex, len(priming) + i)))
            immediate_rows += self.evaluate(method, [ex], "immediate", *eval_args)
            if (i + 1) % cfg["block_size"] == 0 or i == len(stream) - 1:
                rows = self.evaluate(method, stream[:i+1], f"block_{i+1}", *eval_args)
                by_id = {r["id"]: r for r in rows}
                block_metrics.append({"after_writes": i + 1, "aggregate": aggregate(rows),
                    "by_block": [aggregate([by_id[x.id] for x in stream[k:k+cfg["block_size"]]])
                                 for k in range(0, i+1, cfg["block_size"])]})
                print(json.dumps({"method": method, "writes": i+1, **aggregate(rows)}), flush=True)
        final_rows = rows
        final_control = self.evaluate(method, control, "control_after", *eval_args)
        priming_final = self.evaluate(method, priming, "priming_after", *eval_args)
        paraphrases = self.evaluate(method, stream, "paraphrase", *eval_args, paraphrase=True) if self.dataset == "synthetic" else []
        metrics = {"method": method, "dataset": self.dataset, "seed": cfg["seed"],
            "control_before": aggregate(initial), "control_after": aggregate(final_control),
            "priming_before": aggregate(priming_before), "priming_after": aggregate(priming_final),
            "pre_write": aggregate(pre_rows), "immediate": aggregate(immediate_rows),
            "final": aggregate(final_rows), "paraphrase": aggregate(paraphrases),
            "retention_matrix": block_metrics,
            "trainable_parameters": sum(p.numel() for p in bank.parameters()) if bank is not None else
                                      (sum(p.numel() for p in self.reader.parameters()) if method.startswith("latent") else 0),
            "online_gradient_steps": cfg["write_steps"] * len(stream) if bank is not None else 0,
            "write_seconds": sum(x["seconds"] for x in updates),
            "peak_gpu_allocated_bytes_process": torch.cuda.max_memory_allocated(),
            "evaluation_parameters_immutable": True,
            "timing_note": "routing features are cached across repeats/methods; warm pilot timings, not deployment benchmark"}
        if bank is not None:
            metrics["active_adapter_parameters"] = sum(p.numel() for p in bank.parameters_for_slot(0))
            metrics["route_counts"] = {str(s): sum(x.get("slot") == s for x in updates if x["phase"] == "online") for s in range(slots)}
            if router is not None:
                metrics["paraphrase_route_agreement"] = sum(router.route(self.feature(x.question)) == router.route(self.feature(x.paraphrase)) for x in stream) / len(stream)
            torch.save({"bank": bank.state_dict(), "router": router.state_dict() if hasattr(router, "state_dict") else None,
                        "optimizers": [o.state_dict() for o in optimizers], "config": cfg}, out / "adapter.pt")
        elif method.startswith("latent"):
            torch.save(memory.state_dict(), out / "memory.pt")
            metrics["external_value_bytes"] = sum(v.numel() * v.element_size() for v in self.values.values())
            metrics["external_key_bytes"] = len(self.values) * b.model.config.hidden_size * 4
        json_write(out / "metrics.json", metrics)
        json_write(out / "updates.json", updates)
        b.mode, b.bank, b.value = "none", None, None
        del bank, optimizers
        torch.cuda.empty_cache()
        return metrics


def train_reader(backend, cfg, output):
    """Offline supervised episodic training on disjoint synthetic facts.

    The writer sees an observed support; the reader sees a separate question.
    Online inference then freezes both modules and only appends encoded values.
    """
    train = synthetic_dataset(cfg["seed"] + 1000, cfg["latent_train_size"], 0)["stream"]
    dev = synthetic_dataset(cfg["seed"] + 2000, cfg["latent_dev_size"], 0)["stream"]
    save_examples(output / "reader_train.jsonl", train)
    save_examples(output / "reader_dev.jsonl", dev)
    reader = LatentResidual(backend.model.config.hidden_size, rank=cfg["latent_rank"], seed=cfg["seed"]).to("cuda")
    backend.reader = reader
    optimizer = torch.optim.Adam(reader.parameters(), lr=cfg["latent_learning_rate"])
    supports = {x.id: backend.feature("Remember this information: " + x.support).to("cuda") for x in train + dev}
    best, history = float("inf"), []
    rng = random.Random(cfg["seed"])
    for epoch in range(cfg["latent_epochs"]):
        order = list(train)
        rng.shuffle(order)
        train_losses = []
        for ex in order:
            backend.mode = "latent"
            backend.value = reader.encode(supports[ex.id])
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = backend.loss(ex.question, ex.answer)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite offline reader loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        nll, tokens = 0.0, 0
        with torch.no_grad():
            for ex in dev:
                backend.value = reader.encode(supports[ex.id])
                score = backend.score(ex.question, ex.answer)
                nll += score.nll_sum
                tokens += score.tokens
        dev_nll = nll / tokens
        history.append({"epoch": epoch+1, "train_mean_loss_with_eos": float(np.mean(train_losses)), "dev_answer_token_nll": dev_nll})
        print(json.dumps({"reader_training": history[-1]}), flush=True)
        if dev_nll < best:
            best = dev_nll
            torch.save({"reader": reader.state_dict(), "config": cfg, "epoch": epoch+1}, output / "reader.pt")
    checkpoint = torch.load(output / "reader.pt", map_location="cuda", weights_only=True)
    reader.load_state_dict(checkpoint["reader"])
    reader.requires_grad_(False).eval()
    json_write(output / "reader_training.json", {"history": history, "selected_epoch": checkpoint["epoch"], "train_facts": len(train), "dev_facts": len(dev), "selection_metric": "dev_answer_token_nll"})
    backend.mode, backend.value = "none", None
    return reader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=["synthetic", "medmcqa"], default="synthetic")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--reader-checkpoint", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.methods:
        cfg["methods"] = a.methods
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {a.output}")
    a.output.mkdir(parents=True)
    a.data.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    json_write(a.output / "config.json", cfg)
    if a.dataset == "synthetic":
        data = synthetic_dataset(cfg["seed"], cfg["stream_size"], cfg["control_size"])
        priming = synthetic_dataset(cfg["seed"] + 3000, cfg["priming_size"], 0)["stream"]
    else:
        data = prepare_medmcqa(a.data, cfg["seed"], cfg["priming_size"] + cfg["stream_size"], cfg["control_size"])
        priming, data["stream"] = data["stream"][:cfg["priming_size"]], data["stream"][cfg["priming_size"]:]
    for name, examples in dict(data, priming=priming).items():
        save_examples(a.output / f"{name}.jsonl", examples)
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "dataset": a.dataset,
        "model_id": cfg["model_id"], "data_sha256": {name: digest_file(a.output / f"{name}.jsonl") for name in ["stream", "control", "priming"]},
        "packages": {x: importlib.metadata.version(x) for x in ["torch", "transformers", "numpy", "huggingface_hub"]},
        "hardware": torch.cuda.get_device_name(0), "device_visible_count": torch.cuda.device_count(),
        "script_sha256": digest_file(__file__), "answer_nll_includes_eos": False,
        "train_loss_includes_eos": True, "protocol": "supervised observation; query-before-write; immutable delayed evaluation"}
    model_manifest = Path(a.model).parent / "manifest.json"
    if model_manifest.exists():
        metadata["model_revision"] = json.loads(model_manifest.read_text())["revision"]
    json_write(a.output / "manifest.json", metadata)
    backend = QwenBackend(a.model, layer=cfg["layer"], max_input_tokens=cfg["max_input_tokens"])
    reader = None
    if any(x.startswith("latent") for x in cfg["methods"]):
        if a.reader_checkpoint:
            checkpoint = torch.load(a.reader_checkpoint, map_location="cuda", weights_only=True)
            reader = LatentResidual(backend.model.config.hidden_size, cfg["latent_rank"], seed=cfg["seed"]).to("cuda")
            reader.load_state_dict(checkpoint["reader"])
            reader.requires_grad_(False).eval()
        else:
            reader = train_reader(backend, cfg, a.output)
    experiment = Experiment(backend, cfg, a.output, a.dataset, reader)
    results = []
    for method in cfg["methods"]:
        print(f"METHOD_START {method}", flush=True)
        results.append(experiment.run_method(method, priming, data["stream"], data["control"]))
        json_write(a.output / "summary.json", results)
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["complete"] = True
    json_write(a.output / "manifest.json", metadata)
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
