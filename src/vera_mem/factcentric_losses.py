"""Offline fact-identity objectives for the VeRA memory read/write interface.

Each batch row identifies a *fact/entity*, not an answer class. All styles of
that row are positive addresses, while every different fact is a negative,
including facts with the same answer word. No parser or entity identifier is
used by the deployed query encoder: identities only supervise offline losses.

The retrieval objective averages the log probability of every positive view.
It deliberately does not take logsumexp over positives, which could satisfy a
query using only its easiest, same-style positive and ignore the other styles.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _views(name: str, value: Tensor) -> None:
    if value.ndim != 3 or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must be nonempty [facts, views, dimensions]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _unit(value: Tensor) -> Tensor:
    # Float32 similarity/log_softmax avoids half-precision overflow at low
    # temperatures without detaching gradients or downcasting float64 tests.
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    return F.normalize(value, dim=-1, eps=1e-12)


def all_view_retrieval_loss(
    query: Tensor,
    key: Tensor,
    *,
    temperature: float = 0.1,
    entity_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    """Symmetric uniform-positive cross entropy over all query/key views.

    ``query[B,Vq,D]`` and ``key[B,Vs,D]`` share their fact ordering. Inputs
    normally come from L2-normalized encoders; defensive normalization makes
    this objective insensitive to projection magnitude. The default identities
    are ``arange(B)``. Optional integer ``entity_ids[B]`` marks repeated rows of
    the same fact as positives too, avoiding false negatives when resampling.

    For one query, the target distribution assigns equal probability to *all*
    keys with the same identity. Thus its irreducible loss is log(number of
    positive key views), and raw losses with different view counts should not
    be interpreted as directly comparable retrieval quality. Training batches
    should balance fact/view counts; repeated identities are supported but
    retain their sampling multiplicity. Different entities sharing the same
    answer must have different identities.

    Returns ``retrieval`` (the average of both directions), ``query_to_key``,
    and ``key_to_query``. This function never learns or changes a VDB.
    """
    _views("query", query)
    _views("key", key)
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("query and key must share fact count and dimensions")
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query and key must share device and dtype")
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    facts, query_views, dimensions = query.shape
    key_views = key.shape[1]
    if entity_ids is None:
        entity_ids = torch.arange(facts, device=query.device)
    elif (
        entity_ids.ndim != 1 or entity_ids.shape[0] != facts
        or entity_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("entity_ids must be an integer tensor with shape [facts]")
    entity_ids = entity_ids.to(device=query.device)
    query_ids = entity_ids.repeat_interleave(query_views)
    key_ids = entity_ids.repeat_interleave(key_views)
    positives = query_ids[:, None] == key_ids[None, :]
    logits = (
        _unit(query).reshape(-1, dimensions)
        @ _unit(key).reshape(-1, dimensions).T
    ) / float(temperature)
    target = positives.to(dtype=logits.dtype)
    q_to_k = -(F.log_softmax(logits, dim=1) * target).sum(1).div(target.sum(1)).mean()
    k_to_q = -(F.log_softmax(logits, dim=0) * target).sum(0).div(target.sum(0)).mean()
    return {"retrieval": (q_to_k + k_to_q) / 2, "query_to_key": q_to_k, "key_to_query": k_to_q}


def style_block_retrieval_loss(
    query: Tensor,
    key: Tensor,
    *,
    temperature: float = 0.1,
    entity_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    """Average symmetric fact retrieval over every query/key-style block.

    Inputs and identity semantics match :func:`all_view_retrieval_loss`.
    View index must identify the same style across all facts in a domain.
    For each pair of styles ``(a,b)``, classify each ``query[:,a]`` against
    ``key[:,b]`` and vice versa. Every candidate in one bank has a common
    style, so a style preference cannot distinguish positive from negative
    facts. The objective covers *all* style pairs equally without demanding
    calibration of logits between different support-style banks, unlike the
    flattened all-view objective. Repeated fact rows remain uniform positives.

    This loss cannot make a representation semantic by itself; unsupported
    features, absent training variation, and readout failures remain possible.
    """
    _views("query", query)
    _views("key", key)
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("query and key must share fact count and dimensions")
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query and key must share device and dtype")
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    facts = query.shape[0]
    if entity_ids is None:
        entity_ids = torch.arange(facts, device=query.device)
    elif (
        entity_ids.ndim != 1 or entity_ids.shape[0] != facts
        or entity_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("entity_ids must be an integer tensor with shape [facts]")
    entity_ids = entity_ids.to(device=query.device)
    # [query_style, key_style, query_fact, key_fact], not cross-style banks.
    logits = torch.einsum("aid,bjd->ijab", _unit(query), _unit(key)) / float(temperature)
    target = (entity_ids[:, None] == entity_ids[None, :]).to(dtype=logits.dtype)
    q_to_k = -(F.log_softmax(logits, dim=-1) * target).sum(-1).div(target.sum(-1)).mean()
    k_to_q = -(F.log_softmax(logits, dim=-2) * target).sum(-2).div(target.sum(-2)).mean()
    return {"retrieval": (q_to_k + k_to_q) / 2, "query_to_key": q_to_k, "key_to_query": k_to_q}


def canonical_anchor_loss(
    student: Tensor,
    anchor: Tensor,
    *,
    metric: str = "cosine",
) -> Tensor:
    """Align every student style to its detached canonical fact anchor.

    Shapes are ``student[B,V,D]`` and ``anchor[B,D]``. ``cosine`` is mean
    ``1-cos(student, anchor)``; ``normalized_mse`` is mean squared L2 distance
    between unit vectors (twice cosine distance for nonzero vectors). Anchor
    tensors are always detached, even if the caller forgot ``no_grad``.

    This alone does not prevent collapse: a useful value anchor must vary
    across facts and come from a frozen, independently trained writer. The
    readout language-model objective must also remain active when training
    values. A same-answer grouping is inappropriate for addressing anchors.
    """
    _views("student", student)
    if anchor.ndim != 2 or anchor.shape != (student.shape[0], student.shape[2]):
        raise ValueError("anchor must have shape [facts, dimensions] matching student")
    if not anchor.is_floating_point():
        raise ValueError("anchor must be floating point")
    if metric not in ("cosine", "normalized_mse"):
        raise ValueError("metric must be cosine or normalized_mse")
    normalized_student = _unit(student)
    normalized_anchor = _unit(anchor.detach().to(device=student.device, dtype=student.dtype))[:, None, :]
    if metric == "cosine":
        return (1 - (normalized_student * normalized_anchor).sum(-1)).mean()
    return (normalized_student - normalized_anchor).square().sum(-1).mean()


def factcentric_losses(
    query: Tensor,
    key: Tensor,
    value: Tensor | None = None,
    *,
    temperature: float = 0.1,
    entity_ids: Tensor | None = None,
    query_anchor: Tensor | None = None,
    key_anchor: Tensor | None = None,
    value_anchor: Tensor | None = None,
    query_anchor_weight: float = 1.0,
    key_anchor_weight: float = 1.0,
    value_anchor_weight: float = 1.0,
    anchor_metric: str = "cosine",
) -> dict[str, Tensor]:
    """Return explicit retrieval/anchor components and their weighted total.

    Query/key shapes follow :func:`all_view_retrieval_loss`; optional values
    are ``[B,Vs,rank]``. Anchor dimensions match the corresponding student.
    Missing anchor terms are scalar zeros. ``total`` contains no LM loss: the
    runner is responsible for adding its language-model/readout supervision.
    """
    result = all_view_retrieval_loss(query, key, temperature=temperature, entity_ids=entity_ids)
    if value is not None:
        _views("value", value)
        if value.shape[:2] != key.shape[:2] or value.device != key.device:
            raise ValueError("value must share key fact/view counts and device")
    total = result["retrieval"]
    for name, student, anchor, weight in (
        ("query_anchor", query, query_anchor, query_anchor_weight),
        ("key_anchor", key, key_anchor, key_anchor_weight),
        ("value_anchor", value, value_anchor, value_anchor_weight),
    ):
        if isinstance(weight, bool) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{name}_weight must be finite and nonnegative")
        if anchor is None:
            component = total.new_zeros(())
        else:
            if student is None:
                raise ValueError("value is required when value_anchor is supplied")
            component = canonical_anchor_loss(student, anchor, metric=anchor_metric)
        result[name] = component
        total = total + float(weight) * component
    result["total"] = total
    return result
