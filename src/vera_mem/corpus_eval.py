"""Static language modeling and frozen-weight continual writes are separate protocols."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .corpus_backend import CorpusBackbone, CorpusLoRA, adapter_delta
from .corpus_data import Corpus, atomic_json, sha256
from .corpus_episodes import answer_text, prompt_ids
from .corpus_features import SupportEncoder
from .corpus_index import CandidateIndex, coarse_keys
from .corpus_reader import CorpusReader, FeatureBank, FrozenBank, MODES


def digest_parameters(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def score_answer(prediction, answer):
    predicted, gold = answer_text(prediction).split(), answer_text(answer).split()
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    return {"em": float(predicted == gold),
            "f1": 2 * overlap / (len(predicted) + len(gold)) if predicted or gold else 1.,
            "first_three_em": float(predicted[:3] == gold)}


@torch.no_grad()
def generate(backbone, module, bank, index, prompts, *, positive=None, oracle=False,
             empty=False, shuffle=False, max_new_tokens=32):
    from transformers.cache_utils import DynamicCache
    batch = len(prompts)
    length = max(map(len, prompts))
    ids = torch.full((batch, length), backbone.tokenizer.pad_token_id, device="cuda", dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = torch.tensor(prompt, device="cuda")
        mask[row, -len(prompt):] = 1
    cache = DynamicCache(config=backbone.model.config)
    done = torch.zeros(batch, dtype=torch.bool, device="cuda")
    generated = [[] for _ in prompts]
    diagnostics = [{} for _ in prompts]
    t0 = time.monotonic()
    for step in range(max_new_tokens):
        prefix = backbone.prefix(ids, mask, past=cache)
        delta, aux = adapter_delta(module, prefix, bank, index, positive=positive, oracle=oracle,
                                    empty=empty, shuffle_values=shuffle)
        hidden = backbone.suffix(prefix, delta)
        logits = backbone.model.lm_head(hidden[:, -1]).float()
        next_id = logits.argmax(-1)
        if step == 0 and "selected" in aux:
            selected = aux["selected"][:, -1].cpu().tolist()
            candidates = aux["candidate_ids"].reshape(batch, ids.shape[1], -1)[:, -1]
            for row in range(batch):
                diagnostics[row].update(selected=selected[row],
                                        candidate_recall=float(positive[row] in candidates[row]) if positive is not None else None,
                                        block_recall=float(positive[row] in selected[row]) if positive is not None else None)
        values = next_id.cpu().tolist()
        finished = done.cpu().tolist()
        for row, token in enumerate(values):
            if not finished[row] and token != backbone.tokenizer.eos_token_id:
                generated[row].append(token)
        done |= next_id == backbone.tokenizer.eos_token_id
        if bool(done.all()):
            break
        ids = next_id[:, None]
        mask = torch.cat((mask, torch.ones((batch, 1), device="cuda", dtype=mask.dtype)), dim=1)
    elapsed = time.monotonic() - t0
    for row in range(batch):
        diagnostics[row].update(generated_tokens=len(generated[row]), batch_seconds=elapsed,
                                batch_size=batch, reached_token_limit=len(generated[row]) == max_new_tokens)
    return [backbone.tokenizer.decode(row, skip_special_tokens=True) for row in generated], diagnostics


@torch.no_grad()
def static_ppl(backbone, modules, banks, index, corpus, output, *, batch_size=2, shuffled=False):
    output = Path(output)
    if output.exists():
        return json.loads(output.read_text())
    table = corpus.window_table(1024, 512)
    sums = {name: np.zeros((len(corpus.offsets) - 1, 2)) for name in modules}
    t0 = time.monotonic()
    for start in range(0, len(table), batch_size):
        records = table[start:start + batch_size]
        ids, labels, mask = [torch.from_numpy(t).to("cuda") for t in corpus.batch(records, corpus.manifest["pad_token_id"])]
        prefix = backbone.prefix(ids, mask)
        for name, module in modules.items():
            delta, _ = adapter_delta(module, prefix, banks.get(name), index, shuffle_values=shuffled)
            hidden = backbone.suffix(prefix, delta)
            for row, record in enumerate(records):
                valid = labels[row] != -100
                chosen = hidden[row, valid]
                target = labels[row, valid]
                loss = 0.
                for offset in range(0, len(chosen), 128):
                    logits = backbone.model.lm_head(chosen[offset:offset + 128]).float()
                    loss += float(F.cross_entropy(logits, target[offset:offset + 128], reduction="sum"))
                sums[name][record[0]] += [loss, len(target)]
        if start % 100 == 0:
            print(json.dumps({"stage": "static_ppl", "split": corpus.split,
                              "shuffled": shuffled, "windows_done": start + len(records), "windows": len(table)}), flush=True)
    excluded = sorted({r["id"] for r in corpus.manifest["cross_split_overlaps"] if r["split"] == corpus.split})
    clean = np.ones(len(corpus.offsets) - 1, dtype=bool)
    clean[excluded] = False
    result = {"split": corpus.split, "condition": "shuffled_values" if shuffled else "train_only_bank",
              "seconds": time.monotonic() - t0, "excluded_primary_articles": excluded,
              "score_rule": "1024 input positions, stride 512; every target exactly once including initial/tail windows",
              "methods": {}}
    for name, values in sums.items():
        primary = values[clean].sum(0)
        full = values.sum(0)
        if int(full[1]) != corpus.manifest["splits"][corpus.split]["lm_target_tokens"]:
            raise RuntimeError("Static evaluation omitted or duplicated targets")
        result["methods"][name] = {"nll": float(primary[0] / primary[1]), "ppl": math.exp(primary[0] / primary[1]),
                                    "target_tokens": int(primary[1]), "official_all_ppl": math.exp(full[0] / full[1]),
                                    "official_all_target_tokens": int(full[1]), "article_nll_tokens": values.tolist()}
    atomic_json(output, result)
    return result


def episode_features(args, rows, split):
    """Encode only observed A/B text. Gold answer fields are not passed to the encoder."""
    output = Path(args.output) / f"observations_{split}.npz"
    if output.exists():
        return np.load(output)
    encoder = SupportEncoder(args.model)
    corpus = Corpus(args.data, split)
    arrays = {}
    for variant, field in (("a", "support_ids"), ("b", "edited_support_ids")):
        chunks = []
        for start in range(0, len(rows), 96):
            k, v, m, _ = encoder.encode([row[field] for row in rows[start:start + 96]],
                                         corpus.manifest["boundary_token_id"], corpus.manifest["pad_token_id"])
            chunks.append((k, v, m))
        for i, field_name in enumerate(("keys", "values", "masks")):
            arrays[variant + "_" + field_name] = np.concatenate([chunk[i] for chunk in chunks])
    np.savez(output, **arrays)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(output)


def evaluate_condition(backbone, module, bank, index, rows, output, *, positive=None,
                       edited=False, context=False, oracle=False, shuffle=False,
                       heldout=True, batch_size=16, extra_contexts=None):
    output = Path(output)
    if output.exists():
        return [json.loads(line) for line in output.open()]
    records = []
    for style in (0, 1):
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = []
            for offset, row in enumerate(batch):
                context_text = (row["edited_support"] if edited else row["support"]) if context else None
                if extra_contexts is not None:
                    context_text = extra_contexts[str(style)][start + offset]
                prompts.append(prompt_ids(backbone.tokenizer, row, style=style, heldout=heldout, context=context_text))
            observed = None if positive is None else np.asarray(positive[start:start + len(batch)])
            predictions, diagnostics = generate(backbone, module, bank, index, prompts, positive=observed,
                                                 oracle=oracle, shuffle=shuffle)
            for row, prediction, diagnostic in zip(batch, predictions, diagnostics):
                answer = row["edited_answer"] if edited else row["answer"]
                records.append({"id": row["id"], "article": row["article"], "style": style,
                                "prediction": prediction, "answer": answer,
                                **score_answer(prediction, answer), **diagnostic})
    temporary = output.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
    temporary.replace(output)
    return records


@torch.no_grad()
def retrieved_contexts(backbone, reader, bank, index, rows, text_for_id):
    result = {"0": [], "1": []}
    for style in (0, 1):
        for row in rows:
            prompt = prompt_ids(backbone.tokenizer, row, heldout=True, style=style)
            ids = torch.tensor([prompt], device="cuda")
            prefix = backbone.prefix(ids, torch.ones_like(ids))
            one = dict(prefix, x=prefix["x"][:, -1:])
            _, aux = adapter_delta(reader, one, bank, index)
            selected = aux["selected"][0, -1].cpu().tolist()
            result[str(style)].append("\n\n".join(text_for_id(i) for i in selected))
    return result


@torch.no_grad()
def audit_learned_ann(backbone, modules, index, corpus, output):
    """Check index approximation separately from semantic memory relevance."""
    import faiss
    output = Path(output)
    if output.exists():
        return
    table = corpus.window_table()
    selected = np.random.default_rng(20261013).choice(len(table), 8, replace=False)
    features = []
    for record in table[selected]:
        ids, labels, mask = [torch.from_numpy(t).to("cuda") for t in corpus.batch([record], corpus.manifest["pad_token_id"])]
        prefix = backbone.prefix(ids, mask)
        x = prefix["x"][0]
        positions = torch.linspace(0, len(x) - 1, 32, device="cuda").long()
        features.append(modules["block_outer"].query_features(x[positions]))
    features = torch.cat(features)
    exact = faiss.IndexFlatIP(index.vectors.shape[1])
    exact.add(index.vectors)
    result = {"split": corpus.split, "queries": len(features), "fixed_budget": True, "methods": {}}
    for name in MODES:
        queries = modules[name].route_queries(features).float().cpu().numpy()
        _, truth = exact.search(queries, 32)
        began = time.monotonic()
        actual = index.search(queries, 32)
        elapsed = time.monotonic() - began
        result["methods"][name] = {"recall_at_32": float(np.mean([
            len(set(a.tolist()) & set(b.tolist())) / 32 for a, b in zip(actual, truth)])),
            "seconds": elapsed, "queries_per_second": len(queries) / elapsed}
    atomic_json(output, result)


def summarize_records(records):
    result = {"queries": len(records), "articles": len(set(r["article"] for r in records))}
    for metric in ("em", "f1", "first_three_em", "candidate_recall", "block_recall"):
        numbers = [r[metric] for r in records if r.get(metric) is not None]
        if numbers:
            result[metric] = float(np.mean(numbers))
    return result


def continual(args, backbone, modules, banks, training_index, rows, observations, split):
    output = Path(args.output) / f"continual_{split}"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "complete.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    before = {name: digest_parameters(module) for name, module in modules.items() if module is not None}
    index = training_index
    feature_mean = modules["block_outer"].key_mean.cpu().numpy()
    summary = {"units": len(rows), "articles": len(set(row["article"] for row in rows)),
               "shared_weights_frozen": True, "groups": {}}
    log = []
    def condition(name, phase, selected_rows, *, active_index=index, ids=None, **kwargs):
        path = output / f"{name}_{phase}.jsonl"
        records = evaluate_condition(backbone, modules.get(name), banks.get(name), active_index,
                                      selected_rows, path, positive=ids, heldout=True, **kwargs)
        summary["groups"][name + "/" + phase] = summarize_records(records)
        atomic_json(output / "progress.json", summary)
        print(json.dumps({"stage": "continual", "method": name, "phase": phase,
                          **summary["groups"][name + "/" + phase]}), flush=True)
        return records
    condition("base", "empty", rows)
    condition("lora", "prewrite", rows)
    condition("base", "context_a", rows, context=True)
    condition("base", "context_b", rows, context=True, edited=True)
    for name in MODES:
        condition(name, "prewrite", rows)
    original_ids = np.empty(len(rows), dtype=np.int64)
    text_payload = {}
    train_corpus = Corpus(args.data, "train")
    def text_for_id(physical):
        if physical in text_payload:
            return text_payload[physical]
        return backbone.tokenizer.decode(train_corpus.chunk(physical))
    def write(start, end, variant):
        keys, values, masks = [observations[variant + "_" + key][start:end] for key in ("keys", "values", "masks")]
        t0 = time.monotonic()
        expected = np.arange(index.index.ntotal, index.index.ntotal + end - start)
        for bank in banks.values():
            actual = bank.append(keys, values, masks)
            if not np.array_equal(actual, expected):
                raise RuntimeError("VDB payload and physical index IDs diverged")
        coarse = coarse_keys(keys, masks, feature_mean)
        ids = index.upsert_many([row["id"] for row in rows[start:end]], coarse,
                                [-1 - row["article"] for row in rows[start:end]])
        if not np.array_equal(ids, expected):
            raise RuntimeError("Unexpected physical observation IDs")
        for physical, row in zip(ids, rows[start:end]):
            text_payload[int(physical)] = row["support"] if variant == "a" else row["edited_support"]
        log.append({"variant": variant, "start": start, "end": end, "seconds_all_readers": time.monotonic() - t0,
                    "physical_start": int(ids[0]), "physical_end": int(ids[-1]),
                    "active_records": int(index.active.sum()), "physical_records": index.index.ntotal})
        atomic_json(output / "write_log.json", log)
        return ids
    # Sequential observations; no later support is searchable in an earlier query.
    for start in range(0, len(rows), 100):
        end = min(len(rows), start + 100)
        ids = write(start, end, "a")
        original_ids[start:end] = ids
        for name in MODES:
            condition(name, f"write_a_{start:04d}", rows[start:end], ids=ids)
        if not args.development:
            contexts_path = output / f"rag_contexts_{start:04d}.json"
            if contexts_path.exists():
                contexts = json.loads(contexts_path.read_text())
            else:
                contexts = retrieved_contexts(backbone, modules["block_outer"], banks["block_outer"], index,
                                               rows[start:end], text_for_id)
                atomic_json(contexts_path, contexts)
            condition("base", f"retrieved_context_{start:04d}", rows[start:end], extra_contexts=contexts)
        if start:
            for name in MODES:
                condition(name, f"retention_after_{end:04d}", rows[:100], ids=original_ids[:100])
    # Controls read the final A bank. Main A accuracy above uses the actual stream.
    for name in MODES:
        condition(name, "final_a", rows, ids=original_ids)
        condition(name, "oracle_a", rows, ids=original_ids, oracle=True)
        condition(name, "shuffled_a", rows, ids=original_ids, shuffle=True)
    # A stale snapshot is an explicit intervention, not the live edited store.
    stale = index.clone()
    edited_ids = np.empty(len(rows), dtype=np.int64)
    for start in range(0, len(rows), 100):
        end = min(len(rows), start + 100)
        ids = write(start, end, "b")
        edited_ids[start:end] = ids
        for name in MODES:
            condition(name, f"write_b_{start:04d}", rows[start:end], ids=ids, edited=True)
            condition(name, f"stale_b_{start:04d}", rows[start:end], ids=original_ids[start:end],
                      edited=True, active_index=stale)
    del stale
    for name in MODES:
        condition(name, "oracle_b", rows, ids=edited_ids, oracle=True, edited=True)
    for start in range(0, len(rows), 100):
        end = min(len(rows), start + 100)
        ids = write(start, end, "a")
        for name in MODES:
            condition(name, f"restore_a_{start:04d}", rows[start:end], ids=ids)
    after = {name: digest_parameters(module) for name, module in modules.items() if module is not None}
    if before != after:
        raise RuntimeError("Online evaluation changed model weights or statistics")
    summary.update(parameter_digests_before=before, parameter_digests_after=after,
                   write_events=len(log), final_active_records=int(index.active.sum()),
                   final_physical_records=index.index.ntotal)
    if not args.development:
        index_path = output / "restored_index"
        if not index_path.exists():
            index.save(index_path)
        for name, bank in banks.items():
            path = output / f"{name}_new_observations.pt"
            # Base rows reconstruct exactly from the pinned feature cache/checkpoint.
            begin = len(train_corpus.chunks)
            torch.save({"keys": bank.keys[begin:bank.size].cpu(), "values": bank.values[begin:bank.size].cpu(),
                        "masks": bank.masks[begin:bank.size].cpu(), "physical_start": begin,
                        "base_feature_manifest_sha256": sha256(Path(args.features) / "progress.json"),
                        "checkpoint_sha256": sha256(args.checkpoint)}, path)
    atomic_json(state_path, summary)
    return summary


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not args.development and not checkpoint["state"]["complete"]:
        raise ValueError("Confirmation evaluation requires a complete full-pass checkpoint")
    split = "validation" if args.development else "test"
    rows = [json.loads(line) for line in (Path(args.episodes) / f"{split}.jsonl").open()]
    if args.development:
        rows = rows[:64]
    manifest = {"checkpoint_sha256": sha256(args.checkpoint), "development": args.development,
                "units": len(rows), "split": split, "seed": 20261013,
                "data_manifest_sha256": sha256(Path(args.data) / "manifest.json"),
                "episode_manifest_sha256": sha256(Path(args.episodes) / "manifest.json"),
                "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                  ("corpus_eval.py", "corpus_backend.py", "corpus_reader.py", "corpus_index.py",
                                   "corpus_data.py", "corpus_episodes.py", "corpus_features.py")},
                "empty_memory": "Exactly the frozen backbone by zero-injection construction",
                "inference_payload": "8 rows of 64-dimensional K and V; no labels or original writer features",
                "generation": {"greedy": True, "max_new_tokens": 32, "styles": 2}}
    registration = output / "evaluation.json"
    if registration.exists() and json.loads(registration.read_text()) != manifest:
        raise ValueError("Evaluation inputs or code changed; use a fresh evaluation directory")
    atomic_json(registration, manifest)
    # This prefix encoder exits before the full reader backbone is allocated.
    observations = episode_features(args, rows, split)
    backbone = CorpusBackbone(args.model)
    backbone.qualify_prefix()
    modules = {"base": None}
    for name, config in checkpoint["configurations"].items():
        module = CorpusLoRA(**config) if name == "lora" else CorpusReader(**config)
        module.load_state_dict(checkpoint["modules"][name])
        modules[name] = module.to("cuda").eval().requires_grad_(False)
    feature_bank = FeatureBank(args.features)
    banks = {name: FrozenBank(modules[name], feature_bank) for name in MODES}
    del feature_bank, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    index = CandidateIndex.load(args.index, threads=8)
    audit_learned_ann(backbone, modules, index, Corpus(args.data, "validation"), output / "ann_validation.json")
    static_ppl(backbone, modules, banks, index, Corpus(args.data, "validation"), output / "static_validation.json")
    static_ppl(backbone, {n: modules[n] for n in MODES}, banks, index, Corpus(args.data, "validation"),
                output / "shuffled_validation.json", shuffled=True)
    if not args.development:
        static_ppl(backbone, modules, banks, index, Corpus(args.data, "test"), output / "static_test.json")
    continual(args, backbone, modules, banks, index, rows, observations, split)
    atomic_json(output / "complete.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "data", "episodes", "features", "index", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--development", action="store_true")
    main(parser.parse_args())
