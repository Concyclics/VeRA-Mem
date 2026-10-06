"""Paired answer interventions and reserved expression families for VeRA memory.

An entity has two possible observed assignments A/B with different answers but
exactly the same question. Only one world is present in a bank at a time. A
third world P rephrases the target A observation without changing its answer.
This module constructs text/data only: it neither encodes a support nor writes
to a VDB. Changing the target observation can change both its learned key and
value; a value-only intervention requires the runner to explicitly keep its key.

Offline facts reuse the historical 4096 training and 64 development entities.
Fresh confirmation has 64 new entities, two reserved expression families and
fully crossed, label-balanced query/support style assignments. Constructing a
confirmation split does not authorize using it for training or model selection.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import random
from string import Formatter
from typing import Sequence

from . import augmentation_data as historical
from .data import Example, MEMORY_WORDS, synthetic_dataset


PROTOCOL_VERSION = "counterfactual-memory-pairs-v2"
MAX_TRAIN_SIZE = 4096
DEV_SIZE = 64
CONFIRMATION_SIZE = 64
CONFIRMATION_SEED = 37042
ALTERNATIVE_SEEDS = {"train": 31042, "dev": 32042, "confirmation": 33042}
CONFIRMATION_STYLE_SEED = 34042
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
# Verified by the pinned Qwen3-4B-Instruct-2507 tokenizer before this protocol.
# Counts exclude EOS. Paired A/B answers always have equal token lengths.
TOKEN_LENGTH_GROUPS = {
    1: ("apple", "river", "window", "basket", "cloud", "mirror", "rabbit", "silver", "orange", "forest"),
    2: ("candle", "garden", "tiger", "pencil", "lemon", "island"),
}
ANSWER_TOKEN_LENGTHS = {word: length for length, words in TOKEN_LENGTH_GROUPS.items() for word in words}


@dataclass(frozen=True)
class Template:
    id: str
    split: str
    kind: str
    family: str
    text: str


@dataclass(frozen=True)
class CounterfactualPair:
    original: Example
    alternative: Example

    def __post_init__(self):
        if not isinstance(self.original, Example) or not isinstance(self.alternative, Example):
            raise ValueError("Counterfactual worlds must both be Examples")
        a, b = self.original, self.alternative
        if a.id != b.id or a.metadata != b.metadata or not isinstance(a.metadata.get("entity"), str):
            raise ValueError("Counterfactual worlds must share ID and entity metadata")
        if a.question != b.question or a.paraphrase != b.paraphrase or a.choices != b.choices:
            raise ValueError("Counterfactual worlds must share answer-free query fields")
        if a.answer not in ANSWER_TOKEN_LENGTHS or b.answer not in ANSWER_TOKEN_LENGTHS or a.answer == b.answer:
            raise ValueError("Counterfactual worlds require two different vocabulary answers")
        if ANSWER_TOKEN_LENGTHS[a.answer] != ANSWER_TOKEN_LENGTHS[b.answer]:
            raise ValueError("Counterfactual answers must have the same token length")

    @property
    def id(self) -> str:
        return self.original.id

    @property
    def entity(self) -> str:
        return self.original.metadata["entity"]

    @property
    def a(self) -> Example:
        return self.original

    @property
    def b(self) -> Example:
        return self.alternative


_RESERVED_TEMPLATES = (
    Template("cf_confirmation_query_json", "confirmation", "query", "cf_json_request",
             "Read this JSON request and return only the assigned memory word.\n"
             '{{"entity":"{entity}","requested_field":"assigned memory word"}}'),
    Template("cf_confirmation_query_markdown", "confirmation", "query", "cf_markdown_request",
             "Answer the request in this Markdown table with only the assigned memory word.\n"
             "| Entity | Requested field |\n| --- | --- |\n| {entity} | assigned memory word |"),
    Template("cf_confirmation_support_json", "confirmation", "support", "cf_json_record",
             '{{"entity":"{entity}","assigned_memory_word":"{answer}"}}'),
    Template("cf_confirmation_support_markdown", "confirmation", "support", "cf_markdown_record",
             "| Entity | Assigned memory word |\n| --- | --- |\n| {entity} | {answer} |"),
)
_HISTORICAL_TEMPLATES = tuple(
    Template(**asdict(template)) for split in ("train", "dev", "test") for kind in ("query", "support")
    for template in historical.get_templates(split, kind)
)
_BY_ID = {template.id: template for template in _HISTORICAL_TEMPLATES+_RESERVED_TEMPLATES}


def get_templates(split: str, kind: str) -> tuple[Template, ...]:
    """Return train/dev templates or the two newly reserved confirmation ones."""
    if split not in {"train", "dev", "confirmation"} or kind not in {"query", "support"}:
        raise ValueError("Unknown counterfactual template split/kind")
    return tuple(template for template in _BY_ID.values() if template.split == split and template.kind == kind)


def template_ids() -> dict:
    """Cache-axis IDs; historical dev retains canonical plus its two old views.

    Confirmation axes contain only the two new families. Canonical confirmation
    questions/supports can be rendered using ``canonical_template_ids`` in the
    dataset packet; these are separate from the four balanced new-style cells.
    """
    result = {
        split: {domain: [template.id for template in get_templates(split, kind)]
                for domain, kind in (("q", "query"), ("s", "support"))}
        for split in ("train", "dev", "confirmation")
    }
    for domain in ("q", "s"):
        result["dev"][domain].insert(0, result["train"][domain][0])
    return result


def _template(identifier, kind):
    name = identifier if isinstance(identifier, str) else getattr(identifier, "id", None)
    if name not in _BY_ID:
        raise ValueError("Unknown counterfactual/historical template ID")
    template = _BY_ID[name]
    if not isinstance(identifier, str) and asdict(identifier) != asdict(template):
        raise ValueError("Template object differs from frozen protocol")
    if template.kind != kind:
        raise ValueError(f"Expected {kind} template")
    return template


def _example(item, world="A") -> Example:
    if world not in {"A", "B", "P"}:
        raise ValueError("World must be A, B or P")
    if isinstance(item, CounterfactualPair):
        return item.alternative if world == "B" else item.original
    if not isinstance(item, Example):
        raise ValueError("Expected Example or CounterfactualPair")
    return item


def _entity(example):
    entity = example.metadata.get("entity")
    if not isinstance(entity, str) or not entity:
        raise ValueError("Example has no entity metadata")
    return entity


def render_question(item, template_id, *, world="A") -> str:
    """Entity-only rendering: answer, stored question and support are ignored."""
    return _template(template_id, "query").text.format(entity=_entity(_example(item, world)))


def render_support(item, template_id, *, world="A") -> str:
    example = _example(item, world)
    return _template(template_id, "support").text.format(entity=_entity(example), answer=example.answer)


def validate_tokenization(answer_token_ids: dict[str, Sequence[int]]) -> None:
    """Optional runner preflight against the pinned tokenizer, before caching."""
    if set(answer_token_ids) != set(MEMORY_WORDS):
        raise ValueError("Tokenizer validation requires all 16 answer words")
    first = []
    for word in MEMORY_WORDS:
        tokens = answer_token_ids[word]
        if len(tokens) != ANSWER_TOKEN_LENGTHS[word] or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"Pinned answer token length changed: {word}")
        first.append(tokens[0])
    if len(first) != len(set(first)):
        raise ValueError("Answer first-token IDs must be distinct")


def _balanced_order(rows):
    groups = {word: [row for row in rows if row.answer == word] for word in MEMORY_WORDS}
    counts = {len(group) for group in groups.values()}
    if len(counts) != 1:
        raise ValueError("Every answer must have the same number of base facts")
    return [groups[word][index] for index in range(next(iter(counts))) for word in MEMORY_WORDS]


def _make_pairs(rows, seed):
    """Independent within-length-group derangements in each balanced block.

    Both answer marginals are exactly uniform. A new derangement is sampled for
    each 16-fact block, so alternative answers are not a global label mapping.
    Original row order is restored, permitting reuse of cached A/query inputs.
    """
    ordered = _balanced_order(rows)
    rng, alternatives = random.Random(seed), {}
    for start in range(0, len(ordered), len(MEMORY_WORDS)):
        block = {row.answer: row for row in ordered[start:start+len(MEMORY_WORDS)]}
        for words in TOKEN_LENGTH_GROUPS.values():
            permuted = list(words)
            while True:
                rng.shuffle(permuted)
                if all(a != b for a, b in zip(words, permuted)):
                    break
            for original_answer, alternative_answer in zip(words, permuted):
                alternatives[block[original_answer].id] = alternative_answer
    result = []
    for original in rows:
        alternative = replace(original, answer=alternatives[original.id], metadata=dict(original.metadata))
        alternative.support = render_support(alternative, "train_support_00")
        result.append(CounterfactualPair(original, alternative))
    return result


def _historical_entities(base):
    excluded = {row.metadata["entity"] for rows in base.values() for row in rows}
    # Also keep the older scaling-development/online/smoke entities untouched.
    for seed, stream_size, control_size in ((2042, 64, 0), (7042, 128, 64), (8042, 32, 16), (27042, 64, 0)):
        extra = synthetic_dataset(seed, stream_size, control_size)
        excluded.update(row.metadata["entity"] for rows in extra.values() for row in rows)
    return excluded


def confirmation_view_assignments(pairs, *, seed=CONFIRMATION_STYLE_SEED):
    """Assign four (query,support) cells once per A label AND once per B label.

    A/B labels form a degree-four bipartite multigraph. Four independently
    randomized perfect matchings assign the four style cells. This implements
    within-label randomized balancing on both sides, preventing an accidental
    answer/style association in either world. It never uses an entity index
    modulo style count. The same assignment is reused by A/B; P swaps only the
    support family. All world/view observations of an entity remain correlated.
    """
    if len(pairs) != CONFIRMATION_SIZE:
        raise ValueError("Confirmation style assignment requires exactly 64 paired facts")
    for world in ("A", "B"):
        if Counter(_example(pair, world).answer for pair in pairs) != Counter({word: 4 for word in MEMORY_WORDS}):
            raise ValueError("Confirmation requires four facts per A and B label")
    if len({pair.id for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate confirmation entity")
    rng, remaining, assigned = random.Random(seed), set(range(len(pairs))), {}
    cells = [(q, s) for q in range(2) for s in range(2)]
    rng.shuffle(cells)
    for query_view, support_view in cells:
        options = {word: [index for index in sorted(remaining) if pairs[index].original.answer == word]
                   for word in MEMORY_WORDS}
        for values in options.values():
            rng.shuffle(values)
        match = {}

        def augment(answer, seen):
            for index in options[answer]:
                destination = pairs[index].alternative.answer
                if destination in seen:
                    continue
                seen.add(destination)
                if destination not in match or augment(match[destination][0], seen):
                    match[destination] = (answer, index)
                    return True
            return False

        order = list(MEMORY_WORDS)
        rng.shuffle(order)
        if not all(augment(word, set()) for word in order) or len(match) != len(MEMORY_WORDS):
            raise RuntimeError("Balanced A/B style matching failed")
        for _, index in match.values():
            assigned[index] = (query_view, support_view)
            remaining.remove(index)
    if remaining:
        raise AssertionError("Some confirmation facts lack a view assignment")
    ids = template_ids()["confirmation"]
    return [dict(id=pair.id, query_view=assigned[index][0], support_view=assigned[index][1],
                 query_template_id=ids["q"][assigned[index][0]],
                 support_template_id=ids["s"][assigned[index][1]],
                 paraphrase_template_id=ids["s"][1-assigned[index][1]])
            for index, pair in enumerate(pairs)]


def datasets(train_size: int = MAX_TRAIN_SIZE) -> dict:
    """Return immutable-protocol pairs and explicit cache/view metadata.

    ``train_size`` selects a nested, balanced prefix (multiples of 16). Dev is
    retained solely for future diagnostics; the caller must explicitly decide
    whether to evaluate it. Confirmation is fixed at 64 fresh entities and must
    be excluded from every offline training/cache-selection path.
    """
    if type(train_size) is not int or not 16 <= train_size <= MAX_TRAIN_SIZE or train_size % 16:
        raise ValueError("train_size must be a multiple of 16 in [16, 4096]")
    base = historical.datasets()
    excluded = _historical_entities(base)
    fresh = _balanced_order(synthetic_dataset(CONFIRMATION_SEED, CONFIRMATION_SIZE, 0)["stream"])
    fresh = [replace(row, metadata={**row.metadata, "split": "confirmation", "protocol": PROTOCOL_VERSION,
                                    "never_written": False, "reserved_for_confirmation": True}) for row in fresh]
    if any(row.metadata["entity"] in excluded for row in fresh):
        raise RuntimeError("Fresh confirmation overlaps historical entities")
    if len({_entity(row) for row in fresh}) != CONFIRMATION_SIZE:
        raise RuntimeError("Fresh confirmation entities are not unique")
    result = {
        "train": _make_pairs(base["train"], ALTERNATIVE_SEEDS["train"])[:train_size],
        "dev": _make_pairs(base["dev"], ALTERNATIVE_SEEDS["dev"]),
        "confirmation": _make_pairs(fresh, ALTERNATIVE_SEEDS["confirmation"]),
        "template_ids": template_ids(),
        "canonical_template_ids": dict(q="train_query_00", s="train_support_00", paraphrase_s="train_support_01"),
    }
    result["confirmation_view_assignments"] = confirmation_view_assignments(result["confirmation"])
    return result


def _indices(pairs, target_index, indices):
    if type(target_index) is not int or not 0 <= target_index < len(pairs):
        raise ValueError("Invalid target_index")
    chosen = list(range(len(pairs))) if indices is None else list(indices)
    if (not chosen or any(type(index) is not int or not 0 <= index < len(pairs) for index in chosen)
            or len(chosen) != len(set(chosen)) or target_index not in chosen):
        raise ValueError("Bank indices must be unique valid indices including the target")
    if len({pairs[index].id for index in chosen}) != len(chosen):
        raise ValueError("A bank cannot contain multiple worlds/rows for one entity")
    return chosen


def _paraphrase_id(support_template_id):
    original = _template(support_template_id, "support")
    if original.split == "confirmation":
        alternatives = get_templates("confirmation", "support")
    else:
        alternatives = tuple(template for template in _HISTORICAL_TEMPLATES
                             if template.split == original.split and template.kind == "support")
    if original.id == "train_support_00":
        return "train_support_01"
    return next(template.id for template in alternatives if template.id != original.id)


def paired_bank(pairs: Sequence[CounterfactualPair], target_index: int, indices=None, *,
                query_template_id="train_query_00", support_template_id="train_support_00",
                paraphrase_template_id=None) -> dict:
    """Construct A/B/P banks differing in exactly one target observation.

    B changes its answer A→B while preserving target entity, question and support
    style. P changes only the target support style and retains A's answer. All
    non-target rows are identical in all three worlds. Returned Examples and
    metadata dictionaries are independent copies; callers cannot mutate inputs
    or another world through a shared row. Bank construction is offline data
    preparation, not an implicit reveal/write operation.
    """
    chosen = _indices(pairs, target_index, indices)
    original_style = _template(support_template_id, "support")
    paraphrase_template_id = paraphrase_template_id or _paraphrase_id(support_template_id)
    paraphrase_style = _template(paraphrase_template_id, "support")
    if paraphrase_style.split != original_style.split or paraphrase_style.id == original_style.id:
        raise ValueError("Paraphrase must use a different support style from the same split")
    _template(query_template_id, "query")
    result = {}
    for world in ("A", "B", "P"):
        bank = []
        for index in chosen:
            selected_world = "B" if index == target_index and world == "B" else "A"
            item = _example(pairs[index], selected_world)
            style = paraphrase_template_id if index == target_index and world == "P" else support_template_id
            bank.append(replace(item, question=render_question(item, query_template_id),
                                support=render_support(item, style), metadata=dict(item.metadata)))
        result[world] = bank
    position = chosen.index(target_index)
    result.update(query=result["A"][position].question, target_position=position,
                  target_id=pairs[target_index].id,
                  answers={world: result[world][position].answer for world in ("A", "B", "P")},
                  query_template_id=_template(query_template_id, "query").id,
                  support_template_id=original_style.id, paraphrase_template_id=paraphrase_style.id)
    return result


def hard_negative_indices(pairs: Sequence[CounterfactualPair], target_index: int, *, world="A", indices=None):
    """Offline candidates with the target answer but a different entity.

    Background facts retain their original A assignments in every paired bank.
    The target label changes only in B. These candidates must remain negative
    *addresses* even though copying their answer would appear correct. Never
    use this label-based sampler or its labels to construct inference queries.
    """
    chosen = _indices(pairs, target_index, indices)
    answer = _example(pairs[target_index], world).answer
    return [index for index in chosen if index != target_index
            and pairs[index].entity != pairs[target_index].entity and pairs[index].original.answer == answer]


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def protocol_manifest(train_size=MAX_TRAIN_SIZE):
    packet = datasets(train_size)
    historical_packet = historical.datasets()
    style_counts = {world: {word: {} for word in MEMORY_WORDS} for world in ("A", "B")}
    for pair, view in zip(packet["confirmation"], packet["confirmation_view_assignments"]):
        cell = view["query_template_id"]+"/"+view["support_template_id"]
        for world in ("A", "B"):
            table = style_counts[world][_example(pair, world).answer]
            table[cell] = table.get(cell, 0)+1
    return dict(
        protocol=PROTOCOL_VERSION, model_revision=MODEL_REVISION,
        historical_protocol_fingerprint=historical.protocol_fingerprint(),
        seeds=dict(train_entities=historical.TRAIN_SEED, dev_entities=historical.DEV_SEED,
                   confirmation_entities=CONFIRMATION_SEED, alternatives=dict(ALTERNATIVE_SEEDS),
                   confirmation_styles=CONFIRMATION_STYLE_SEED),
        counts={split: len(packet[split]) for split in ("train", "dev", "confirmation")},
        pair_sha256={split: _fingerprint([asdict(pair) for pair in packet[split]])
                     for split in ("train", "dev", "confirmation")},
        answer_token_lengths=dict(ANSWER_TOKEN_LENGTHS), answer_vocabulary=list(MEMORY_WORDS),
        tokenization_requirement="Pinned tokenizer; lengths exclude EOS; all 16 first-token IDs must be distinct",
        answer_marginals={split: {world: dict(Counter(_example(pair, world).answer for pair in packet[split]))
                                  for world in ("A", "B")}
                          for split in ("train", "dev", "confirmation")},
        answer_pair_counts={split: {word: dict(Counter(pair.alternative.answer for pair in packet[split]
                                                      if pair.original.answer == word)) for word in MEMORY_WORDS}
                            for split in ("train", "dev", "confirmation")},
        confirmation_answer_style_crosstab=style_counts,
        confirmation_assignment_sha256=_fingerprint(packet["confirmation_view_assignments"]),
        historical_entity_exclusion_sha256=_fingerprint(sorted(_historical_entities(historical_packet))),
        historical_exclusion_sources=["generalization train/dev/test/control", "scaling dev seed2042",
                                       "scaling online seed7042", "scaling smoke seed8042", "invalid counterfactual v1 seed27042"],
        template_ids=packet["template_ids"], canonical_template_ids=packet["canonical_template_ids"],
        templates=[asdict(template) for template in _HISTORICAL_TEMPLATES+_RESERVED_TEMPLATES],
        reserved_confirmation_families=["JSON object", "Markdown table"],
        cache_contract=dict(q="[N,Vq,D], shared by both answers; never derive queries from support/labels",
                            s="[N,2,Vs,D], worlds A/B; use template_ids to identify axes",
                            confirmation="Separate artifact; never select for offline training or model selection",
                            old_dev="Retained as data only; no automatic evaluation"),
        limitations=[
            "Finite 16-word vocabulary; equal-length pairs are a controlled intervention, not natural question answering.",
            "Only two reserved structure families and one fixed alternative per entity are provided.",
            "A/B/P and view records for the same entity are paired; uncertainty must resample entities, not records.",
            "Different-answer supports can change both keys and values; value-only interventions require explicit fixed-key handling.",
            "Same-answer wrong-entity retrieval is an addressing error even when answer exact match succeeds.",
        ],
    )


def protocol_fingerprint(train_size=MAX_TRAIN_SIZE):
    return _fingerprint(protocol_manifest(train_size))


def _validate_templates():
    if set(ANSWER_TOKEN_LENGTHS) != set(MEMORY_WORDS):
        raise RuntimeError("Token-length groups must partition the answer vocabulary")
    for template in _BY_ID.values():
        fields = {name for _, name, _, _ in Formatter().parse(template.text) if name is not None}
        if fields != ({"entity"} if template.kind == "query" else {"entity", "answer"}):
            raise RuntimeError("Template fields violate the no-query-label contract")
    old_text = "\n".join(template.text for template in _HISTORICAL_TEMPLATES)
    if '"requested_field"' in old_text or "| --- | --- |" in old_text:
        raise RuntimeError("Reserved structure families appeared in historical templates")


_validate_templates()
