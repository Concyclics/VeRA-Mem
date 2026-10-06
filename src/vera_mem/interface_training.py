"""One joint update's objectives for the staged memory-interface experiments.

The caller owns sampling, optimizer boundaries, train-only provenance and bank
construction. Each B/P bank must replace only its target record. Stable IDs
never enter the inference path; column indices supply offline address labels.
The shared gold branches always remain present. On sampling steps only FKL
switches to a common continuation drawn from a random A/B student world. This
is world-mixture sampled-prefix FKL, not per-world on-policy reverse KL.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F

from .context_distillation import distillation_loss
from .context_distillation_run import clear_bank, gold_tokens, sequence_mean, supervised_loss
from .counterfactual_run import address_loss, first_positions
from .counterfactual_losses import hidden_delta_loss, pair_behavior_loss, paraphrase_invariance_loss
from .factcentric_losses import style_block_retrieval_loss
from .interface_losses import normalized_behavior_loss


METHODS = ("base", "clip", "normalized", "hidden", "on_policy")


def activate_interface_bank(backend, module, key_features, value_features, *, detach=False):
    """Last-token key and selected writer-value features have separate inputs."""
    if (key_features.ndim != 3 or value_features.shape != key_features.shape
            or key_features.device != value_features.device):
        raise ValueError("Expected matching key/value feature banks [batch, records, D]")
    clear_bank(backend)
    backend.mode, backend.vector_vera = "vector_vera", module
    if detach:
        with torch.no_grad():
            backend.vector_keys = module.encode_key(key_features).detach()
            backend.vector_values = module.encode_value(value_features).detach()
    else:
        backend.vector_keys = module.encode_key(key_features)
        backend.vector_values = module.encode_value(value_features)
        if not backend.vector_keys.requires_grad or not backend.vector_values.requires_grad:
            raise AssertionError("Training requires differentiable key and value writers")


def _validate_batch(batch, method, on_policy):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not isinstance(on_policy, bool) or on_policy and method != "on_policy":
        raise ValueError("Only the on_policy arm may request sampled-prefix FKL")
    questions = batch["questions"]
    if isinstance(questions, str) or not questions or any(not isinstance(q, str) for q in questions):
        raise ValueError("questions must be a nonempty string batch")
    size = len(questions)
    for name in ("contexts", "answers"):
        worlds = batch[name]
        if (len(worlds) != 3 or any(isinstance(w, str) or len(w) != size for w in worlds)
                or any(not isinstance(v, str) for w in worlds for v in w)):
            raise ValueError(f"{name} must contain three matching A/B/P string batches")
    if batch["answers"][0] != batch["answers"][2]:
        raise ValueError("Paraphrase P must preserve the complete A answer")
    reference = batch["key_banks"][0]
    if reference.ndim != 3 or reference.shape[0] != size or any(s == 0 for s in reference.shape):
        raise ValueError("Banks must be nonempty [batch, records, features]")
    for name in ("key_banks", "value_banks"):
        banks = batch[name]
        if len(banks) != 3 or any(v.shape != reference.shape or v.device != reference.device for v in banks):
            raise ValueError(f"{name} must contain matching A/B/P feature banks")
    columns = torch.as_tensor(batch["columns"], device=reference.device)
    if (columns.shape != (size,) or columns.dtype not in (torch.int32, torch.int64)
            or bool(((columns < 0) | (columns >= reference.shape[1])).any())
            or len(columns.unique()) != size):
        raise ValueError("columns must contain unique valid target record positions")
    q = batch["q_views"]
    if q.ndim != 3 or q.shape[0] != reference.shape[1] or q.shape[2] != reference.shape[2]:
        raise ValueError("q_views must match bank record order and feature width")
    views = batch["key_views"]
    if (len(views) != 2 or views[0].ndim != 3 or views[0].shape[0] != q.shape[0]
            or views[0].shape[-1] != q.shape[-1] or views[1].shape != views[0].shape):
        raise ValueError("key_views must contain matching A/B [records, styles, D] features")
    weight = batch.get("key_consistency_weight", .05)
    if not isinstance(weight, Real) or isinstance(weight, bool) or not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("key_consistency_weight must be finite and nonnegative")
    return columns.long(), float(weight)


def _fkl(student, teacher):
    for name in ("counts", "batch_indices"):
        if not torch.equal(student[name], teacher[name]):
            raise AssertionError("Teacher/student prediction alignment mismatch")
    loss = distillation_loss(student["logits"], teacher["logits"], direction="forward", reduction="none")
    return sequence_mean(loss, student["batch_indices"], student["counts"])


def _input_cost(backend, questions, tokens, contexts=None):
    return sum(len(backend.prompt_ids(question, None if contexts is None else contexts[row])) + len(token)
               for row, (question, token) in enumerate(zip(questions, tokens)))


def interface_step(backend, module, batch, method, scale, generator, on_policy=False):
    """Return (differentiable total, JSON-safe metrics, exact trajectories).

    Common: mean A/B/P full-sequence FKL + .5 gold CE + .2 actual addressing
    on gold prediction positions + .2 full-style addressing + .05 AP first-logit
    JS + configured key-consistency weight (default .05). ``clip`` adds .1 old
    clipped effect; ``normalized`` adds .1 fixed-scale effect; ``hidden`` adds
    the latter and .1 hidden effect; ``on_policy`` adds only the normalized
    effect and optionally substitutes sampled-prefix FKL. No replay is run.

    The caller activates sampling on exactly one quarter of the on_policy arm's
    updates and records the extra teacher/student/sampling input costs. Other
    objectives remain on their identical gold branches, independent of sampling.
    Whole-answer gold CE includes EOS and weights sequences equally. A/B effects
    and AP consistency use each branch's prediction BEFORE its first answer token.
    """
    columns, key_weight = _validate_batch(batch, method, on_policy)
    questions, contexts, answers = batch["questions"], batch["contexts"], batch["answers"]
    key_banks, value_banks = batch["key_banks"], batch["value_banks"]
    golden = [gold_tokens(backend, world) for world in answers]
    if any(len(tokens) <= 1 for world in golden for tokens in world):
        raise ValueError("Every gold answer must contain a token before EOS")
    a_ids = torch.tensor([v[0] for v in golden[0]], device=backend.device)
    b_ids = torch.tensor([v[0] for v in golden[1]], device=backend.device)
    if bool((a_ids == b_ids).any()):
        raise ValueError("Counterfactual answers require different first-token IDs")

    # Disabling banks here is defense in depth; backend teacher=True additionally
    # disables adapters, detaches results, and restores transient hook state.
    clear_bank(backend)
    with torch.no_grad():
        teachers = [backend.forward_sequences(questions, tokens, contexts=context, teacher=True)
                    for tokens, context in zip(golden, contexts)]
    students, gold_fkl, gold_ce, actual_address, first_ce = [], [], [], [], []
    for keys, values, tokens, teacher in zip(key_banks, value_banks, golden, teachers):
        activate_interface_bank(backend, module, keys, values)
        branch = backend.forward_sequences(questions, tokens)
        students.append(branch)
        gold_fkl.append(_fkl(branch, teacher))
        gold_ce.append(supervised_loss(branch, tokens)[0])
        labels = torch.tensor([row[0] for row in tokens], device=backend.device)
        first_ce.append(F.cross_entropy(branch["logits"][first_positions(branch)], labels))
        actual_address.append(address_loss(backend, module, branch, columns))

    sl = [s["logits"][first_positions(s)] for s in students]
    sh = [s["hidden"][first_positions(s)] for s in students]
    tl = [t["logits"][first_positions(t)] for t in teachers]
    th = [t["hidden"][first_positions(t)] for t in teachers]
    clipped = pair_behavior_loss(sl[0], sl[1], tl[0], tl[1], a_ids, b_ids,
                                 target_clip=10., gold_weight=0.)
    normalized = normalized_behavior_loss(sl[0], sl[1], tl[0], tl[1], a_ids, b_ids,
                                          scale=scale, gold_weight=0.)
    behavior = clipped if method == "clip" else normalized
    hidden = hidden_delta_loss(sh[0], sh[1], th[0], th[1], min_teacher_norm=1e-3)
    invariance = paraphrase_invariance_loss(sl[0], sl[2], temperature=1., hidden_weight=0.)
    query_views = module.encode_query(batch["q_views"])
    key_views = [module.encode_key(view) for view in batch["key_views"]]
    style = torch.stack([style_block_retrieval_loss(query_views, keys, temperature=.1)["retrieval"]
                         for keys in key_views]).mean()
    # Target rows only: unchanged distractors must not dilute the intervention.
    key_consistency = (1. - F.cosine_similarity(key_views[0][columns], key_views[1][columns], dim=-1)).mean()

    selected_world, sampled = [], None
    sampling = dict(forward_calls=0, unpadded_input_tokens=0, padded_input_tokens=0)
    fkl, distill_tokens = gold_fkl, golden
    if on_policy:
        selected_world = torch.randint(2, (len(questions),), generator=generator, device=backend.device).tolist()
        chosen_keys = torch.stack([key_banks[world][row] for row, world in enumerate(selected_world)])
        chosen_values = torch.stack([value_banks[world][row] for row, world in enumerate(selected_world)])
        activate_interface_bank(backend, module, chosen_keys, chosen_values, detach=True)
        sampled = backend.sample_student(questions, max_new_tokens=8, temperature=1., generator=generator)
        sampling = dict(backend.last_sampling_cost)
        distill_tokens = [sampled, sampled, sampled]
        with torch.no_grad():
            sampled_teachers = [backend.forward_sequences(questions, sampled, contexts=context, teacher=True)
                                for context in contexts]
        fkl = []
        for keys, values, teacher in zip(key_banks, value_banks, sampled_teachers):
            activate_interface_bank(backend, module, keys, values)
            branch = backend.forward_sequences(questions, sampled)
            fkl.append(_fkl(branch, teacher))

    forward_kl = torch.stack(fkl).mean()
    full_ce = torch.stack(gold_ce).mean()
    address = torch.stack(actual_address).mean()
    main = forward_kl + .5 * full_ce
    total = main + .2 * address + .2 * style + .05 * invariance["loss"] + key_weight * key_consistency
    if method != "base":
        total = total + .1 * behavior["delta_loss"]
    if method == "hidden":
        total = total + .1 * hidden["loss"]

    student_cost = sum(_input_cost(backend, questions, tokens) for tokens in golden)
    teacher_cost = sum(_input_cost(backend, questions, tokens, context) for tokens, context in zip(golden, contexts))
    gold_count = sum(len(row) for world in golden for row in world)
    distill_count = sum(len(row) for world in distill_tokens for row in world)
    if on_policy:
        student_cost += sum(_input_cost(backend, questions, tokens) for tokens in distill_tokens)
        teacher_cost += sum(_input_cost(backend, questions, tokens, context)
                            for tokens, context in zip(distill_tokens, contexts))
    raw, target = normalized["teacher_delta_raw"], normalized["teacher_target_normalized"]
    legacy_clipped = normalized["legacy_clipped_mask"]
    valid = normalized["valid_mask"]
    metrics = dict(
        main_loss=float(main.detach()), total_loss=float(total.detach()),
        forward_kl=float(forward_kl.detach()), gold_forward_kl=float(torch.stack(gold_fkl).mean().detach()),
        full_sequence_ce=float(full_ce.detach()), first_token_ce=float(torch.stack(first_ce).mean().detach()),
        actual_address_loss=float(address.detach()), style_address_loss=float(style.detach()),
        key_consistency_loss=float(key_consistency.detach()), key_consistency_weight=key_weight,
        behavior_loss=float(behavior["delta_loss"].detach()), clipped_behavior_loss=float(clipped["delta_loss"].detach()),
        normalized_behavior_loss=float(normalized["delta_loss"].detach()), hidden_loss=float(hidden["loss"].detach()),
        paraphrase_loss=float(invariance["loss"].detach()), margin_scale=float(scale),
        behavior_valid_pairs=int(valid.sum()), behavior_clipped_pairs=int(behavior["clipped_mask"].sum()),
        legacy_behavior_clipped_pairs=int(legacy_clipped.sum()),
        legacy_behavior_clipped_valid_pairs=int((legacy_clipped & valid).sum()),
        teacher_behavior_delta_mean=float(raw.mean()), teacher_behavior_delta_std=float(raw.std(unbiased=False)),
        teacher_behavior_delta_min=float(raw.min()), teacher_behavior_delta_max=float(raw.max()),
        teacher_target_normalized_mean=float(target.mean()), teacher_target_normalized_std=float(target.std(unbiased=False)),
        teacher_target_normalized_min=float(target.min()), teacher_target_normalized_max=float(target.max()),
        student_behavior_delta_mean=float(normalized["student_delta"].mean()),
        teacher_first_token_joint_correct=int(((tl[0].argmax(-1) == a_ids) & (tl[1].argmax(-1) == b_ids)).sum()),
        student_first_token_joint_correct=int(((sl[0].argmax(-1) == a_ids) & (sl[1].argmax(-1) == b_ids)).sum()),
        hidden_valid_pairs=int(hidden["valid_count"]), teacher_delta_norm_mean=float(hidden["teacher_delta_norm"].mean()),
        student_delta_norm_mean=float(hidden["student_delta_norm"].mean()),
        student_input_tokens=student_cost, teacher_input_tokens=teacher_cost,
        target_tokens=gold_count + (distill_count if on_policy else 0),
        gold_target_tokens=gold_count, distillation_target_tokens=distill_count,
        student_forward_calls=3 + 3 * int(on_policy), teacher_forward_calls=3 + 3 * int(on_policy),
        sampled_tokens=sum(map(len, sampled)) if on_policy else 0,
        rollout_input_tokens=sampling["unpadded_input_tokens"], rollout_padded_input_tokens=sampling["padded_input_tokens"],
        rollout_forward_calls=sampling["forward_calls"], on_policy=on_policy,
        fkl_prefix_source="student_world_mixture" if on_policy else "gold", replay_target_tokens=0,
    )
    trajectories = [dict(question=question, contexts=[c[row] for c in contexts],
                         continuation_token_ids=[world[row] for world in distill_tokens],
                         gold_token_ids=[world[row] for world in golden],
                         sampled_world=selected_world[row] if on_policy else None,
                         fkl_prefix_source=metrics["fkl_prefix_source"])
                    for row, question in enumerate(questions)]
    return total, metrics, trajectories
