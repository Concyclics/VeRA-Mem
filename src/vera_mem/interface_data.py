"""Multi-relation memory facts with held-out complete multiword answers.

This module defines text and split identity, not a retrieval shortcut. Student
questions expose only the named entity and requested relation. The stable
record ID, A/B world label, and answer are never query-template fields. Writers
must encode ``writer_text`` through the backbone and learn keys from those
features; metadata IDs are for bank bookkeeping and supervision only.

Four entities per group use each relation's two answers in a balanced 2:2
orientation. Both target answers occur in other entities' original A-bank
records, even when only the target switches to B. Each entity has four relations. Complete
answer strings are disjoint between train/dev/confirm, while component words
are shared: this tests compositional answer recall, not unseen token vocabulary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
from string import Formatter
from typing import Callable, Mapping, Sequence

from .data import Example, MEMORY_WORDS


PROTOCOL_VERSION = "interface-multirelation-v2"
SEEDS = {"train": 41042, "dev": 42042, "confirm": 43042}
MAX_COUNTS = {"train": 4096, "dev": 256, "confirm": 256}
MAX_NEW_TOKENS = 32
WRITER_PREFIX = "Remember this information: "
RELATIONS = ("arrival phrase", "departure phrase", "inspection phrase", "dispatch phrase")
GROUP_SIZE = 4 * len(RELATIONS)
ANSWER_INSTRUCTION = (
    "Return only the complete stored three-word answer, in the original word order. "
    "Do not include a label, quotes, JSON, a table, or an explanation."
)
# The first-word vocabulary has distinct first-token IDs under the previously
# pinned Qwen tokenizer. Still validate the actual tokenizer before a run.
FIRST_WORDS = tuple(MEMORY_WORDS)
TAIL_WORDS = (
    "amber", "birch", "cedar", "cobalt", "crimson", "crystal", "daisy", "delta",
    "dune", "elm", "feather", "fern", "flint", "frost", "granite", "harbor",
    "hazel", "heather", "ivory", "jade", "juniper", "lagoon", "lantern", "lilac",
    "lotus", "maple", "marble", "meadow", "mint", "moss", "opal", "orchid",
    "otter", "pebble", "pine", "plum", "quartz", "reed", "robin", "ruby",
    "sage", "shell", "sparrow", "spruce", "stone", "violet", "willow", "wren",
)


def stable_id(entity: str, relation: str) -> str:
    """Identity depends on the entity/relation pair, never its current answer."""
    payload = json.dumps([entity, relation], separators=(",", ":")).encode()
    return "interface-" + hashlib.sha256(payload).hexdigest()[:24]


def _world(world: str) -> str:
    if world not in {"A", "B", "P"}:
        raise ValueError("world must be A, B, or P")
    return world


@dataclass(frozen=True)
class InterfaceFact:
    id: str
    entity: str
    relation: str
    answer_a: str
    answer_b: str
    split: str

    def __post_init__(self):
        if not isinstance(self.entity, str) or re.fullmatch(r"I[0-9a-f]{16}", self.entity) is None:
            raise ValueError("entity must be an opaque I-prefixed identifier")
        if self.relation not in RELATIONS or self.split not in SEEDS:
            raise ValueError("Unknown relation or fact split")
        if self.id != stable_id(self.entity, self.relation):
            raise ValueError("Fact ID must depend only on entity and relation")
        for answer in (self.answer_a, self.answer_b):
            words = answer.split() if isinstance(answer, str) else []
            if (len(words) != 3 or answer != " ".join(words)
                    or words[0] not in FIRST_WORDS or any(w not in TAIL_WORDS for w in words[1:])):
                raise ValueError("Answers must be three words from the fixed component vocabularies")
        if self.answer_a.split()[0] == self.answer_b.split()[0]:
            raise ValueError("A/B first words must differ")

    @property
    def stable_id(self) -> str:
        return self.id

    def answer(self, world: str = "A") -> str:
        return self.answer_b if _world(world) == "B" else self.answer_a

    def to_example(self, world: str = "A", *, query_template="if_train_query_00",
                   support_template="if_train_support_00") -> Example:
        """Adapt to the existing Example interface without the old 16-label restriction."""
        return Example(
            id=self.id, question=render_question(self, query_template, world=world),
            answer=self.answer(world), support=render_support(self, support_template, world=world),
            paraphrase=render_question(self, "if_train_query_01", world=world),
            metadata=dict(entity=self.entity, relation=self.relation, split=self.split,
                          protocol=PROTOCOL_VERSION, never_written=False),
        )

    @property
    def a(self) -> Example:
        return self.to_example("A")

    @property
    def b(self) -> Example:
        return self.to_example("B")

    @property
    def original(self) -> Example:
        return self.a

    @property
    def alternative(self) -> Example:
        return self.b


@dataclass(frozen=True)
class Template:
    id: str
    split: str
    kind: str
    family: str
    text: str


def _query(identifier, split, family, text):
    return Template(identifier, split, "query", family, text + "\n\n" + ANSWER_INSTRUCTION)


_TEMPLATES = (
    _query("if_train_query_00", "train", "prose_wh", "What is the {relation} for entity {entity}?"),
    _query("if_train_query_01", "train", "prose_recall", "Recall the {relation} associated with {entity}."),
    _query("if_train_query_02", "train", "prose_lookup", "Look up entity {entity} and provide its {relation}."),
    _query("if_train_query_03", "train", "prose_entity_first", "For {entity}, which words form the {relation}?"),
    _query("if_train_query_04", "train", "prose_relation_first", "I need the {relation} belonging to entity {entity}."),
    _query("if_train_query_05", "train", "prose_prior_observation", "Using the stored observation, retrieve the {relation} of {entity}."),
    _query("if_train_query_06", "train", "prose_identify", "Please identify entity {entity}'s {relation}."),
    _query("if_train_query_07", "train", "prose_question_last", "Entity {entity} has several recorded phrases. What is its {relation}?"),
    Template("if_train_support_00", "train", "support", "prose_declaration", "The {relation} for entity {entity} is {answer}."),
    Template("if_train_support_01", "train", "support", "prose_assignment", "Entity {entity} has been assigned {answer} as its {relation}."),
    Template("if_train_support_02", "train", "support", "prose_value_first", "The words {answer} form the {relation} belonging to {entity}."),
    Template("if_train_support_03", "train", "support", "prose_observation", "Remember this association: for {entity}, use {answer} for the {relation}."),
    _query("if_dev_query_00", "dev", "prose_request", "Could you give me the saved {relation} of entity {entity}?"),
    _query("if_dev_query_01", "dev", "prose_reconstruction", "Reconstruct the exact {relation} recorded for {entity}."),
    Template("if_dev_support_00", "dev", "support", "prose_update", "A record states that {entity} uses the {relation} {answer}."),
    Template("if_dev_support_01", "dev", "support", "prose_report", "For the entity called {entity}, the recorded {relation} reads {answer}."),
    _query("if_confirm_query_json", "confirm", "json", 'Read the requested field in this JSON request:\n{{"entity":"{entity}","relation":"{relation}"}}'),
    _query("if_confirm_query_markdown", "confirm", "markdown", "Read the requested field in this table:\n| Entity | Requested relation |\n| --- | --- |\n| {entity} | {relation} |"),
    _query("if_confirm_query_yaml", "confirm", "yaml", "Read this YAML lookup request:\nentity: {entity}\nrequested_relation: {relation}"),
    _query("if_confirm_query_ini", "confirm", "ini", "Read this INI lookup request:\n[lookup]\nentity={entity}\nrelation={relation}"),
    Template("if_confirm_support_json", "confirm", "support", "json", '{{"entity":"{entity}","relation":"{relation}","value":"{answer}"}}'),
    Template("if_confirm_support_markdown", "confirm", "support", "markdown", "| Entity | Relation | Value |\n| --- | --- | --- |\n| {entity} | {relation} | {answer} |"),
    Template("if_confirm_support_yaml", "confirm", "support", "yaml", "entity: {entity}\nrelation: {relation}\nvalue: {answer}"),
    Template("if_confirm_support_ini", "confirm", "support", "ini", "[observation]\nentity={entity}\nrelation={relation}\nvalue={answer}"),
)
_BY_ID = {template.id: template for template in _TEMPLATES}


def get_templates(split: str, kind: str) -> tuple[Template, ...]:
    split = "confirm" if split == "confirmation" else split
    if split not in SEEDS or kind not in {"query", "support"}:
        raise ValueError("Unknown interface template split/kind")
    return tuple(t for t in _TEMPLATES if t.split == split and t.kind == kind)


def template_ids() -> dict:
    return {split: {axis: [t.id for t in get_templates(split, kind)]
                    for axis, kind in (("q", "query"), ("s", "support"))} for split in SEEDS}


def _template(identifier: str | Template, kind: str) -> Template:
    key = identifier.id if isinstance(identifier, Template) else identifier
    try:
        template = _BY_ID[key]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown interface template") from error
    if template.kind != kind or isinstance(identifier, Template) and identifier != template:
        raise ValueError("Template kind/content does not match the frozen protocol")
    return template


def _identity(item: InterfaceFact | Example) -> tuple[str, str]:
    if isinstance(item, InterfaceFact):
        return item.entity, item.relation
    if isinstance(item, Example):
        entity, relation = item.metadata.get("entity"), item.metadata.get("relation")
        if isinstance(entity, str) and entity and relation in RELATIONS:
            return entity, relation
    raise ValueError("Expected an InterfaceFact or Example with entity/relation metadata")


def render_question(item: InterfaceFact | Example, template_id: str | Template,
                    *, world: str = "A") -> str:
    """Render only entity/relation; no answer, stable record ID or world marker."""
    _world(world)
    entity, relation = _identity(item)
    return _template(template_id, "query").text.format(entity=entity, relation=relation)


def render_support(item: InterfaceFact | Example, template_id: str | Template,
                   *, world: str = "A") -> str:
    """Render a revealed observation. For Example, its own answer is authoritative."""
    _world(world)
    entity, relation = _identity(item)
    answer = item.answer(world) if isinstance(item, InterfaceFact) else item.answer
    return _template(template_id, "support").text.format(entity=entity, relation=relation, answer=answer)


def writer_text(item: InterfaceFact | Example, template_id: str | Template,
                *, world: str = "A") -> str:
    """Match the existing writer prefix exactly once; teacher receives raw support."""
    return WRITER_PREFIX + render_support(item, template_id, world=world)


def _pools() -> dict[str, list[InterfaceFact]]:
    used_entities, used_answers = set(), set()
    result = {}
    for split, maximum in MAX_COUNTS.items():
        rng, rows = random.Random(SEEDS[split]), []

        def entity():
            while True:
                value = "I" + f"{rng.getrandbits(64):016x}"
                if value not in used_entities:
                    used_entities.add(value)
                    return value

        def answer(exclude_first=None):
            while True:
                first = rng.choice(FIRST_WORDS)
                if first == exclude_first:
                    continue
                value = " ".join((first, rng.choice(TAIL_WORDS), rng.choice(TAIL_WORDS)))
                if value not in used_answers:
                    used_answers.add(value)
                    return value

        for _ in range(maximum // GROUP_SIZE):
            entities = [entity() for _ in range(4)]
            # A complete group includes four entities' four relations. This
            # makes every supported nested prefix retain its hard negatives.
            assignments = [(relation, answer()) for relation in RELATIONS]
            assignments = [(relation, a, answer(a.split()[0])) for relation, a in assignments]
            orientations = []
            for _relation in RELATIONS:
                orientation = [False, False, True, True]
                rng.shuffle(orientation)
                orientations.append(orientation)
            for index, name in enumerate(entities):
                for (relation, a, b), orientation in zip(assignments, orientations):
                    if orientation[index]:
                        a, b = b, a
                    rows.append(InterfaceFact(stable_id(name, relation), name, relation, a, b, split))
        result[split] = rows
    return result


def datasets(train_size: int = 4096, dev_size: int = 128, confirm_size: int = 128) -> dict[str, list[InterfaceFact]]:
    """Return nested complete fact groups with fixed seeds and disjoint answers.

    Counts are facts, not entities: four relations/entity and four entities/group.
    Require multiples of sixteen so neither same-entity relation negatives nor
    same-answer other-entity negatives disappear in a smaller training prefix.
    Confirmation views are reserved; generating their text is not permission to
    use them for fitting, model selection or teacher-prompt tuning.
    """
    counts = dict(train=train_size, dev=dev_size, confirm=confirm_size)
    for split, count in counts.items():
        if type(count) is not int or not GROUP_SIZE <= count <= MAX_COUNTS[split] or count % GROUP_SIZE:
            raise ValueError(f"{split} count must be a multiple of {GROUP_SIZE} in [{GROUP_SIZE}, {MAX_COUNTS[split]}]")
    result = {split: rows[:counts[split]] for split, rows in _pools().items()}
    validate_splits(result)
    return result


def hard_negative_ids(facts: Sequence[InterfaceFact], world: str = "A") -> dict[str, list[str]]:
    """Same answer AND relation, different entity; labels only for episode sampling."""
    _world(world)
    groups = defaultdict(list)
    for fact in facts:
        groups[(fact.relation, fact.answer(world))].append(fact)
    return {fact.id: [other.id for other in groups[(fact.relation, fact.answer(world))]
                      if other.entity != fact.entity] for fact in facts}


def original_bank_hard_negative_ids(facts: Sequence[InterfaceFact], world: str = "A") -> dict[str, list[str]]:
    """Match target A/B against other records actually present in the A bank.

    B evaluation updates only the target, so comparing other B alternatives
    would describe a different bank and cannot certify a real hard negative.
    """
    _world(world)
    groups = defaultdict(list)
    for fact in facts:
        groups[(fact.relation, fact.answer_a)].append(fact)
    return {fact.id: [other.id for other in groups[(fact.relation, fact.answer(world))]
                      if other.entity != fact.entity] for fact in facts}


def relation_negative_ids(facts: Sequence[InterfaceFact]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for fact in facts:
        groups[fact.entity].append(fact)
    return {fact.id: [other.id for other in groups[fact.entity] if other.relation != fact.relation]
            for fact in facts}


def validate_splits(data: Mapping[str, Sequence[InterfaceFact]]) -> None:
    if set(data) != set(SEEDS):
        raise ValueError("Expected train/dev/confirm fact splits")
    seen_entities, seen_answers, seen_ids = set(), set(), set()
    for split, rows in data.items():
        if not rows or any(fact.split != split for fact in rows):
            raise ValueError("Empty or mislabeled fact split")
        entities = {f.entity for f in rows}
        answers = {f.answer(w) for f in rows for w in ("A", "B")}
        identifiers = {f.id for f in rows}
        if len(identifiers) != len(rows) or identifiers & seen_ids:
            raise ValueError("Duplicate fact identity")
        if entities & seen_entities or answers & seen_answers:
            raise ValueError("Cross-split entity or complete-answer overlap")
        by_entity = defaultdict(set)
        for fact in rows:
            by_entity[fact.entity].add(fact.relation)
        if any(relations != set(RELATIONS) for relations in by_entity.values()):
            raise ValueError("Every entity must have all four relations")
        for world in ("A", "B"):
            if not all(original_bank_hard_negative_ids(rows, world).values()):
                raise ValueError("Missing original-bank same-answer different-entity hard negative")
        seen_entities.update(entities)
        seen_answers.update(answers)
        seen_ids.update(identifiers)


def validate_tokenization(facts: Sequence[InterfaceFact], encode: Callable[[str], Sequence[int]],
                          *, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Check the actual tokenizer, leaving room for EOS in the generation budget.

    Call with ``lambda text: tokenizer.encode(text, add_special_tokens=False)``.
    Three words are not assumed to be three tokens. A/B token lengths need not
    match; downstream batching/losses must support their actual lengths.
    """
    if not facts or type(max_new_tokens) is not int or max_new_tokens < 3:
        raise ValueError("Nonempty facts and a valid generation budget are required")
    cache = {}
    unequal_pairs = 0
    for fact in facts:
        for answer in (fact.answer_a, fact.answer_b):
            if answer not in cache:
                tokens = list(encode(answer))
                if (not 2 <= len(tokens) < max_new_tokens
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError(f"Answer violates multi-token/EOS-budget contract: {answer}")
                cache[answer] = tokens
        a, b = cache[fact.answer_a], cache[fact.answer_b]
        if a[0] == b[0]:
            raise ValueError("A/B answers must have different first-token IDs")
        unequal_pairs += len(a) != len(b)
    lengths = Counter(len(tokens) for tokens in cache.values())
    return dict(unique_answers=len(cache), min_answer_tokens=min(lengths), max_answer_tokens=max(lengths),
                token_length_counts=dict(sorted(lengths.items())), unequal_length_pairs=unequal_pairs,
                max_new_tokens=max_new_tokens, eos_budget_reserved=True)


def protocol_manifest(train_size=4096, dev_size=128, confirm_size=128) -> dict:
    data = datasets(train_size, dev_size, confirm_size)
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(
        protocol=PROTOCOL_VERSION, seeds=dict(SEEDS), max_counts=dict(MAX_COUNTS),
        counts={split: len(rows) for split, rows in data.items()},
        entity_counts={split: len({f.entity for f in rows}) for split, rows in data.items()},
        fact_sha256={split: digest([asdict(f) for f in rows]) for split, rows in data.items()},
        templates=[asdict(t) for t in _TEMPLATES], template_ids=template_ids(),
        relations=list(RELATIONS), answer_components=dict(first=list(FIRST_WORDS), tail=list(TAIL_WORDS)),
        complete_answers_disjoint=True, component_vocabulary_shared=True,
        writer_prefix=WRITER_PREFIX, max_new_tokens=MAX_NEW_TOKENS,
        canonical_template_ids=dict(q="if_train_query_00", s="if_train_support_00"),
        hard_negatives="same relation and complete answer, different entity; both A and B",
        uncertainty_unit="entity (all four relations and all views clustered)",
        limitations=[
            "Synthetic three-word values; not evidence of unrestricted natural-language knowledge memory.",
            "Expression families are authored stress tests; teacher behavior still requires independent acceptance.",
            "Complete answers are split-disjoint but component words and relation names are shared.",
            "A/B token lengths may differ; do not reuse the old equal-length 16-word validation contract.",
        ],
    )


def _validate_templates():
    if len(_BY_ID) != len(_TEMPLATES):
        raise RuntimeError("Duplicate interface template ID")
    train_dev_families = {t.family for t in _TEMPLATES if t.split != "confirm"}
    if train_dev_families & {t.family for t in get_templates("confirm", "query")}:
        raise RuntimeError("Confirmation structure family exposed in training/development")
    for template in _TEMPLATES:
        fields = {name for _, name, _, _ in Formatter().parse(template.text) if name is not None}
        expected = {"entity", "relation"} | ({"answer"} if template.kind == "support" else set())
        if fields != expected:
            raise RuntimeError(f"Forbidden/missing template fields in {template.id}: {fields}")
        if template.kind == "query" and not template.text.endswith(ANSWER_INSTRUCTION):
            raise RuntimeError("Every query must end with the complete-answer output instruction")


_validate_templates()
