"""Train-only value-writer supervision against a fixed canonical old writer.

There are no backbone forwards or question-conditioned targets here. Each
training fact/world has one teacher value, encoded from its canonical support
view by the original checkpoint. All support rewrites of that fact/world fit
that same target. Both outputs already have the original rank-wise RMS
normalization: the loss does not replace their scale with unit-L2 vectors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .context_distillation_run import REVISION, append_jsonl, sha256
from .counterfactual_backend import BatchedStableVectorVeRA
from .interface_features import PREFIX
from .interface_variants import InterfaceVectorVeRA
from .run import json_write, tensor_digest


PROTOCOL = "memory-interface-v1"


def _validate_packet(packet, module):
    if (packet.get("protocol") != PROTOCOL or packet.get("model_revision") != REVISION
            or packet.get("split") != "train" or packet.get("task") != "classic"):
        raise ValueError("Writer fitting requires the pinned classic train packet")
    if packet.get("writer_prefix") != PREFIX:
        raise ValueError("Writer prefix differs from the historical feature protocol")
    rows, qids, sids = packet["rows"], packet["qids"], packet["sids"]
    if not rows or not qids or not sids or sids[0] != "train_support_00":
        raise ValueError("Nonempty train data with canonical support view zero required")
    if len(set(qids)) != len(qids) or len(set(sids)) != len(sids):
        raise ValueError("Training template identifiers must be unique")
    n, width = len(rows), module.in_features
    shapes = {"q": (n, len(qids), width), "last": (n, 2, len(sids), width),
              "pool": (n, 2, len(sids), width)}
    for key, shape in shapes.items():
        features = packet[key]
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != shape or not features.is_floating_point():
            raise ValueError(f"Invalid {key} training feature shape or dtype; expected {shape}")
        # Bound validation temporaries for the several-GB offline CPU cache.
        for chunk in features.detach().reshape(-1, width).split(1024):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"Nonfinite {key} training features")
    if len({row["id"] for row in rows}) != n or len({row["entity"] for row in rows}) != n:
        raise ValueError("Training facts and entities must be unique")
    for row in rows:
        if (len(row["questions"]) != len(qids) or len(row["supports"]) != 2
                or any(len(world) != len(sids) for world in row["supports"])):
            raise ValueError("Training text and feature view counts disagree")
    return n, len(sids)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def train_writer(backend, module, packet, args):
    """Fit Wv/optional residual MLP, save the specified last update, return audit.

    ``args.updates`` controls the fixed budget (default 2000 when absent);
    ``args.writer_batch_size`` defaults to 128. The experimental learning rate
    and global gradient clip are fixed at 1e-4 and 1 respectively. Sampling is
    uniform with replacement over independent fact, A/B world, and support-view
    indices, using a dedicated CPU generator seeded by ``args.seed``.

    Only ``backend.device`` is inspected. No backend state, memory bank, base
    parameter, or teacher target is updated. This is a fresh fitting stage, not
    an implicit optimizer continuation from the old checkpoint.
    """
    if not isinstance(module, InterfaceVectorVeRA):
        raise TypeError("Writer training requires InterfaceVectorVeRA")
    facts, views = _validate_packet(packet, module)
    updates = _positive_int(getattr(args, "updates", 2000), "updates")
    batch_size = _positive_int(getattr(args, "writer_batch_size", 128), "writer_batch_size")
    seed = args.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    device = module.Wv.weight.device
    if device != torch.device(backend.device):
        raise ValueError("Backend and writer must use the same device")
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("training.jsonl", "last.pt", "training_status.json")):
        raise FileExistsError("Writer training output already exists; use a new run directory")

    started = time.perf_counter()
    teacher_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = teacher_checkpoint["module"]
    if "interface_architecture" in state:
        raise ValueError("Canonical targets require the original pre-interface checkpoint")
    teacher_config = module.configuration()
    for name in ("writer_mode", "train_B", "value_mlp_hidden", "architecture_version"):
        teacher_config.pop(name)
    teacher = BatchedStableVectorVeRA(**teacher_config)
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).requires_grad_(False).eval()
    teacher_before = tensor_digest(teacher)
    if not teacher._statistics_ready:
        raise ValueError("Canonical teacher must have fitted historical statistics")
    checkpoint_hash = sha256(args.checkpoint)
    target_chunks = []
    # No graph survives target construction. World is never pooled away.
    with torch.no_grad():
        for canonical in packet["last"][:, :, 0].detach().reshape(-1, module.in_features).split(256):
            target_chunks.append(teacher.encode_value(canonical.to(device)).cpu())
    targets = torch.cat(target_chunks).reshape(facts, 2, module.rank)
    if not bool(torch.isfinite(targets).all()):
        raise FloatingPointError("Nonfinite canonical teacher targets")
    if module.writer_mode == "masked_mean":
        module.fit_value_statistics(packet["pool"].detach().reshape(-1, module.in_features))
    features = packet["pool" if module.writer_mode == "masked_mean" else "last"].detach()

    module.requires_grad_(False).train()
    module.zero_grad(set_to_none=True)
    module.Wv.requires_grad_(True)
    if module.value_mlp is not None:
        module.value_mlp.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in trainable]
    if not trainable or any(not (name.startswith("Wv.") or name.startswith("value_mlp.")) for name, _ in trainable):
        raise AssertionError("Only the value writer may receive optimizer updates")
    frozen = {name: value.detach().clone() for name, value in module.state_dict().items()
              if name not in {key for key, _ in trainable}}
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = hashlib.sha256()
    cfg = dict(seed=seed, updates=updates, batch_size=batch_size, learning_rate=1e-4,
               gradient_clip=1., loss="MSE of original RMS-normalized rank values",
               target="frozen old writer on same fact/world canonical support view 0",
               sampling="independent uniform fact/world/view with replacement",
               writer_mode=module.writer_mode, value_mlp_hidden=module.value_mlp_hidden,
               checkpoint_selection="fixed final update", split="train")
    train_started = time.perf_counter()
    for step in range(1, updates + 1):
        ids = torch.randint(facts, (batch_size,), generator=generator)
        worlds = torch.randint(2, (batch_size,), generator=generator)
        support_views = torch.randint(views, (batch_size,), generator=generator)
        exposure = dict(step=step, facts=ids.tolist(), worlds=worlds.tolist(), views=support_views.tolist())
        schedule.update(json.dumps(exposure, sort_keys=True, separators=(",", ":")).encode())
        inputs = features[ids, worlds, support_views].to(device)
        target = targets[ids, worlds].to(device)
        optimizer.zero_grad(set_to_none=True)
        value = module.encode_value(inputs)
        loss = F.mse_loss(value.float(), target.float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite writer fitting loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            diagnostics = dict(loss=float(loss), cosine=float(F.cosine_similarity(value.float(), target.float()).mean()),
                               value_rms=float(value.float().square().mean(-1).sqrt().mean()),
                               target_rms=float(target.float().square().mean(-1).sqrt().mean()),
                               gradient_norm=float(norm))
        row = dict(**exposure, **diagnostics, sample_schedule_sha256=schedule.hexdigest(),
                   elapsed_seconds=time.perf_counter() - started)
        append_jsonl(output / "training.jsonl", [row])
        if step == 1 or step % 100 == 0 or step == updates:
            print(json.dumps({key: row[key] for key in ("step", "loss", "cosine", "elapsed_seconds")}), flush=True)
    training_seconds = time.perf_counter() - train_started
    for name, value in module.state_dict().items():
        if name in frozen and not torch.equal(value, frozen[name]):
            raise AssertionError(f"Frozen writer/readout state changed: {name}")
    if tensor_digest(teacher) != teacher_before or any(p.grad is not None for p in teacher.parameters()):
        raise AssertionError("Canonical teacher changed or received gradients")
    result = dict(complete=True, step=updates, updates=updates, training_examples=updates * batch_size,
                  trainable_count=sum(p.numel() for p in parameters), trainable_names=[name for name, _ in trainable],
                  sample_schedule_sha256=schedule.hexdigest(), teacher_checkpoint_sha256=checkpoint_hash,
                  teacher_module_sha256=teacher_before, teacher_unchanged=True, frozen_state_unchanged=True,
                  targets_require_grad=targets.requires_grad, target_shape=list(targets.shape),
                  final_loss=diagnostics["loss"], final_cosine=diagnostics["cosine"],
                  training_seconds=training_seconds, elapsed_seconds=time.perf_counter() - started,
                  configuration=cfg)
    checkpoint = dict(protocol=PROTOCOL, architecture=module.configuration(), module=module.state_dict(),
                      optimizer=optimizer.state_dict(), step=updates, seed=seed, configuration=cfg,
                      sampler_rng_state=generator.get_state(), sample_schedule_sha256=schedule.hexdigest(),
                      teacher_checkpoint_sha256=checkpoint_hash)
    temporary = output / "last.pt.partial"
    torch.save(checkpoint, temporary)
    temporary.replace(output / "last.pt")
    module.zero_grad(set_to_none=True)
    module.requires_grad_(False).eval()
    json_write(output / "training_status.json", result)
    return result
