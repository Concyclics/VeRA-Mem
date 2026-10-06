"""Prepare distinct train and confirmation caches without evaluating a model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from vera_mem import counterfactual_data as data
from vera_mem.context_distillation import ContextDistillationBackend
from vera_mem.context_distillation_run import REVISION, sha256
from vera_mem.counterfactual_run import PROTOCOL
from vera_mem.run import json_write


# This is the historical writer interface in generalization_run.prepare_features.
# It belongs to feature extraction, not the fact text shown to the teacher.
WRITER_INPUT_PREFIX = "Remember this information: "
PREFIX_CHECK_FACTS = 16
PREFIX_CHECK_MIN_COSINE = .999
PREFIX_CHECK_MAX_RELATIVE_RMSE = .05


def writer_features(backend, supports, batch_size=32):
    """Encode observations using exactly the historical writer instruction."""
    if isinstance(supports, str):
        raise TypeError("supports must be a sequence of observation strings")
    return backend.layer_features([WRITER_INPUT_PREFIX + text for text in supports],
                                  batch_size=batch_size)


def historical_prefix_feature_check(backend, packet, original):
    """Reproduce 16 historical training facts in all four support styles.

    Fixed thresholds accommodate BF16 batch/padding roundoff, not a different
    writer input distribution. The check uses no confirmation facts, teacher
    responses, model predictions, or task scores. A failing report is persisted
    by main before it aborts cache creation.
    """
    pairs = packet["train"][:PREFIX_CHECK_FACTS]
    styles = packet["template_ids"]["train"]["s"]
    if len(pairs) != PREFIX_CHECK_FACTS or len(styles) != 4:
        raise ValueError("Prefix check requires 16 historical training facts and four styles")
    if styles != original["template_ids"]["train"]["s"]:
        raise ValueError("Historical support feature template axes differ")
    texts = [data.render_support(pair, style) for pair in pairs for style in styles]
    inputs = [WRITER_INPUT_PREFIX + text for text in texts]
    actual = writer_features(backend, texts).detach().float().cpu()
    expected = original["features"]["train"]["s"][:len(pairs)].detach().float().cpu()
    expected = expected.flatten(0, 1)
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Historical prefix feature check has invalid shape or nonfinite values")
    norms = expected.norm(dim=-1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("Historical prefix feature reference contains zero vectors")
    cosine = F.cosine_similarity(actual, expected, dim=-1)
    relative_rmse = (actual - expected).norm(dim=-1) / norms
    valid = (cosine >= PREFIX_CHECK_MIN_COSINE) & (relative_rmse <= PREFIX_CHECK_MAX_RELATIVE_RMSE)
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    rows = [dict(id=pair.id, support_template=style, cosine=float(cosine[i * len(styles) + j]),
                 relative_rmse=float(relative_rmse[i * len(styles) + j]))
            for i, pair in enumerate(pairs) for j, style in enumerate(styles)]
    return dict(passed=bool(valid.all()), split="historical_train_only", facts=len(pairs),
                feature_pairs=len(texts), support_templates=styles,
                writer_input_prefix=WRITER_INPUT_PREFIX,
                min_cosine_required=PREFIX_CHECK_MIN_COSINE,
                max_relative_rmse_allowed=PREFIX_CHECK_MAX_RELATIVE_RMSE,
                min_cosine=float(cosine.min()), max_relative_rmse=float(relative_rmse.max()),
                failed_feature_pairs=int((~valid).sum()),
                writer_text_sha256=digest(inputs),
                writer_prompt_token_ids_sha256=digest([backend.prompt_ids(text) for text in inputs]),
                per_feature=rows)


def atomic_torch(path, packet):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(packet, temporary)
    temporary.replace(path)


def make_evaluation_cases(packet, smoke=False):
    pairs = packet["train"][:4] if smoke else packet["confirmation"]
    if smoke:
        assignments = [dict(query_template_id="train_query_02", support_template_id="train_support_02",
                            paraphrase_template_id="train_support_03") for _ in pairs]
    else:
        assignments = packet["confirmation_view_assignments"]
    phases = {}
    for held_s in (False, True):
        for held_q in (False, True):
            phase = ("heldout" if held_s else "canonical") + "_support/" + ("heldout" if held_q else "canonical") + "_query"
            cases = []
            for pair, selected in zip(pairs, assignments):
                qid = selected["query_template_id"] if held_q else "train_query_00"
                sid = selected["support_template_id"] if held_s else "train_support_00"
                pid = selected["paraphrase_template_id"] if held_s else "train_support_01"
                cases.append(dict(id=pair.id, question=data.render_question(pair, qid),
                                  answer_a=pair.a.answer, answer_b=pair.b.answer,
                                  support_a=data.render_support(pair, sid),
                                  support_b=data.render_support(pair, sid, world="B"),
                                  support_p=data.render_support(pair, pid),
                                  query_template=qid, support_template=sid, paraphrase_template=pid))
            phases[phase] = dict(cases=cases, evaluate_unrelated=held_s == held_q)
    return phases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--original-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=PROTOCOL, complete=False, smoke=a.smoke,
                    writer_input_prefix=WRITER_INPUT_PREFIX, historical_prefix_feature_check=None,
                    started_at=datetime.now(timezone.utc).isoformat())
    json_write(a.output/"manifest.json", manifest)
    if json.loads((Path(a.model).parent/"manifest.json").read_text())["revision"] != REVISION:
        raise ValueError("Pinned model revision required")
    torch.set_num_threads(4)
    packet = data.datasets(train_size=32 if a.smoke else 4096)
    original = torch.load(a.original_cache, map_location="cpu", weights_only=True)
    if original["protocol"] != "generalization-v1-layer20" or original["model_revision"] != REVISION:
        raise ValueError("Original feature cache provenance mismatch")
    from vera_mem.augmentation_data import protocol_fingerprint as historical_fingerprint
    if original["data_fingerprint"] != historical_fingerprint():
        raise ValueError("Original rendering fingerprint mismatch")
    backend = ContextDistillationBackend(a.model, layer=20)
    token_map = {word: backend.tokenizer.encode(word, add_special_tokens=False) for word in data.MEMORY_WORDS}
    data.validate_tokenization(token_map)
    pairs = packet["train"]
    for i, pair in enumerate(pairs):
        old = original["examples"]["train"][i]
        if old["id"] != pair.id or old["answer"] != pair.a.answer:
            raise ValueError("Original feature cache example ordering mismatch")
    ids = packet["template_ids"]["train"]
    prefix_check = historical_prefix_feature_check(backend, packet, original)
    manifest["historical_prefix_feature_check"] = prefix_check
    json_write(a.output/"manifest.json", manifest)
    if not prefix_check["passed"]:
        raise ValueError("Historical writer prefix feature check failed; refusing to create caches")
    print(json.dumps(dict(preparing="historical_prefix_feature_check", **prefix_check)), flush=True)
    texts = [data.render_support(pair, template, world="B") for pair in pairs for template in ids["s"]]
    chunks = []
    for start in range(0, len(texts), 256):
        chunks.append(writer_features(backend, texts[start:start+256], batch_size=32))
        print(json.dumps(dict(preparing="train_alternative_supports", done=min(start+256, len(texts)), total=len(texts))), flush=True)
    alternative = torch.cat(chunks).reshape(len(pairs), 4, 9728)
    matched = 0
    for pair in pairs:
        q = data.render_question(pair, ids["q"][0])
        for sid in ids["s"]:
            lengths = [len(backend.prompt_ids(q, data.render_support(pair, sid, world=w))) for w in ("A", "B")]
            if lengths[0] != lengths[1]:
                raise ValueError("A/B teacher prompt lengths differ; protocol forbids this confound")
            matched += 1
    train = dict(protocol=PROTOCOL, split="train", smoke=a.smoke, model_revision=REVISION,
                 writer_input_prefix=WRITER_INPUT_PREFIX, historical_prefix_feature_check=prefix_check,
                 data_fingerprint=data.protocol_fingerprint(len(pairs)),
                 template_ids=ids, pairs=[asdict(pair) for pair in pairs],
                 q=original["features"]["train"]["q"][:len(pairs)].clone(),
                 s=torch.stack((original["features"]["train"]["s"][:len(pairs)], alternative), dim=1))
    atomic_torch(a.output/"train.pt", train)
    del original, train, chunks, alternative
    phases = make_evaluation_cases(packet, smoke=a.smoke)
    unique = list(dict.fromkeys(c[k] for v in phases.values() for c in v["cases"] for k in ("support_a", "support_b", "support_p")))
    features = writer_features(backend, unique, batch_size=32)
    lookup = dict(zip(unique, features))
    for phase in phases.values():
        phase["supports"] = torch.stack([torch.stack([lookup[c[k]] for k in ("support_a", "support_b", "support_p")]) for c in phase["cases"]])
        for c in phase["cases"]:
            if len(backend.prompt_ids(c["question"], c["support_a"])) != len(backend.prompt_ids(c["question"], c["support_b"])):
                raise ValueError("Confirmation A/B teacher prompt length mismatch")
    evaluation = dict(protocol=PROTOCOL, split="confirmation", smoke=a.smoke, model_revision=REVISION,
                      writer_input_prefix=WRITER_INPUT_PREFIX, historical_prefix_feature_check=prefix_check,
                      data_fingerprint=data.protocol_fingerprint(len(pairs)), phases=phases)
    atomic_torch(a.output/"confirmation.pt", evaluation)
    json_write(a.output/"data_manifest.json", data.protocol_manifest(len(pairs)))
    manifest.update(complete=True, train_size=len(pairs), evaluation_size=len(next(iter(phases.values()))["cases"]),
                    model_revision=REVISION, original_cache_sha256=sha256(a.original_cache),
                    train_cache_sha256=sha256(a.output/"train.pt"), confirmation_cache_sha256=sha256(a.output/"confirmation.pt"),
                    train_context_length_matched=matched, teacher_generations_performed=0,
                    finished_at=datetime.now(timezone.utc).isoformat())
    json_write(a.output/"manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
