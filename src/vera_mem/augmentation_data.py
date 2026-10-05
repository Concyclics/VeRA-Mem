"""Frozen fact and expression splits for template-generalization experiments.

Facts retain the finite 16-word vocabulary of ``synthetic_dataset``. Expression
families, rather than random rows from the same template, are held out. The
training facts exactly match the earlier scaling study's balanced ordering;
development and confirmation facts use new seeds. A query renderer never reads
an example's answer. Support rendering is only for a revealed observation: this
module does not write any feature or observation into a vector database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from string import Formatter
from typing import Literal

from .data import Example, MEMORY_WORDS, synthetic_dataset


PROTOCOL_VERSION = "generalization-templates-v1"
TRAIN_SEED = 1042
DEV_SEED = 12042
TEST_SEED = 17042
MAX_TRAIN_SIZE = 4096
DEV_SIZE = 64
MAX_TEST_SIZE = 128
CONTROL_SIZE = 32


@dataclass(frozen=True)
class Template:
    id: str
    split: Literal["train", "dev", "test"]
    kind: Literal["query", "support"]
    family: str
    text: str


# Deliberately keep training expressions in ordinary prose. Development checks
# use request/record cards and leading output constraints. Confirmation adds
# XML, CSV, and dialogue formats, each absent from train and development. These
# are explicit stress-test families, not a representative natural-language
# distribution. The historical failed paraphrase is exposed in training (01),
# so success on that wording must never be described as held-out generalization.
_TEMPLATES = (
    Template("train_query_00", "train", "query", "train_canonical_wh",
             "What is the assigned memory word for entity {entity}? Reply with only the word."),
    Template("train_query_01", "train", "query", "train_recall_imperative",
             "Recall the memory word associated with {entity}. Return only that word."),
    Template("train_query_02", "train", "query", "train_lookup_imperative",
             "Look up entity {entity} in memory and provide its assigned word. Respond with one word only."),
    Template("train_query_03", "train", "query", "train_entity_first",
             "Entity {entity} has an assigned memory word. What is it? Give only that single word."),
    Template("train_query_04", "train", "query", "train_identify_request",
             "Please identify the memory word assigned to {entity}. Your entire answer must be that one word."),
    Template("train_query_05", "train", "query", "train_association_question",
             "Which memory word goes with entity {entity}? Answer using only the word itself."),
    Template("train_query_06", "train", "query", "train_prior_observation",
             "Based on the stored observation about {entity}, return its assigned memory word. Output exactly one word."),
    Template("train_query_07", "train", "query", "train_possessive_question",
             "What is entity {entity}'s memory word? Respond with the assigned word and nothing else."),
    Template("train_support_00", "train", "support", "train_canonical_declaration",
             "The assigned memory word for entity {entity} is {answer}."),
    Template("train_support_01", "train", "support", "train_entity_assignment",
             "Entity {entity} has been assigned the memory word {answer}."),
    Template("train_support_02", "train", "support", "train_observation_declaration",
             "Store this association: entity {entity} uses {answer} as its memory word."),
    Template("train_support_03", "train", "support", "train_value_first_declaration",
             "The memory word {answer} belongs to entity {entity}."),
    Template("dev_query_00", "dev", "query", "dev_request_card",
             "Memory lookup request\nEntity: {entity}\nRequested field: assigned memory word\nReturn one word only."),
    Template("dev_query_01", "dev", "query", "dev_leading_constraint",
             "Only one word is permitted in your reply. Retrieve the assigned memory word of entity {entity}."),
    Template("dev_support_00", "dev", "support", "dev_record_card",
             "Memory record\nEntity: {entity}\nAssigned word: {answer}"),
    Template("dev_support_01", "dev", "support", "dev_leading_constraint_record",
             "Remember the following assignment exactly: for {entity}, the memory word to retain is {answer}."),
    Template("test_query_00", "test", "query", "test_xml_request",
             "Read this request and output only the requested memory word, without tags.\n"
             "<request><entity>{entity}</entity><field>assigned memory word</field></request>"),
    Template("test_query_01", "test", "query", "test_csv_completion",
             "Complete the missing memory_word field in this CSV record. Return only its single-word value, not the CSV row.\n"
             "entity,memory_word\n{entity},?"),
    Template("test_query_02", "test", "query", "test_dialogue_completion",
             "Complete the assistant's response in the exchange below with only the assigned memory word.\n"
             "User: Tell me the memory word linked to entity {entity}.\nAssistant:"),
    Template("test_support_00", "test", "support", "test_xml_record",
             "<memory><entity>{entity}</entity><assigned_word>{answer}</assigned_word></memory>"),
    Template("test_support_01", "test", "support", "test_csv_record",
             "entity,memory_word\n{entity},{answer}"),
    Template("test_support_02", "test", "support", "test_dialogue_record",
             "User: What memory word is assigned to entity {entity}?\nAssistant: {answer}"),
)
_BY_ID = {template.id: template for template in _TEMPLATES}


def get_templates(split: str, kind: str) -> tuple[Template, ...]:
    """Return ordered immutable templates; canonical wording is train index 0."""
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown expression split {split!r}")
    if kind not in {"query", "support"}:
        raise ValueError(f"Unknown template kind {kind!r}")
    return tuple(template for template in _TEMPLATES
                 if template.split == split and template.kind == kind)


def _template(template_id: str | Template, kind: str) -> Template:
    identifier = template_id.id if isinstance(template_id, Template) else template_id
    try:
        template = _BY_ID[identifier]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown template ID {identifier!r}") from error
    if isinstance(template_id, Template) and template_id != template:
        raise ValueError("Template object does not match the frozen protocol")
    if template.kind != kind:
        raise ValueError(f"Template {identifier!r} is {template.kind}, not {kind}")
    return template


def _entity(example: Example) -> str:
    entity = example.metadata.get("entity")
    if not isinstance(entity, str) or not entity:
        raise ValueError("Example has no entity metadata")
    return entity


def render_question(example: Example, template_id: str | Template) -> str:
    """Construct a query using entity identity only, never hidden label data."""
    return _template(template_id, "query").text.format(entity=_entity(example))


def render_support(example: Example, template_id: str | Template) -> str:
    """Construct an observation; callers control the causal reveal boundary."""
    return _template(template_id, "support").text.format(
        entity=_entity(example), answer=example.answer)


def _balanced_prefix(rows: list[Example], count: int) -> list[Example]:
    groups = {word: [example for example in rows if example.answer == word]
              for word in MEMORY_WORDS}
    ordered = [groups[word][index]
               for index in range(len(rows) // len(MEMORY_WORDS))
               for word in MEMORY_WORDS]
    return ordered[:count]


def _count(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def datasets(train_size: int = MAX_TRAIN_SIZE,
             eval_size: int = MAX_TEST_SIZE) -> dict[str, list[Example]]:
    """Return balanced nested facts with disjoint entities and fresh eval seeds.

    Changing a sample count only truncates a fixed balanced pool. Control labels
    are random unobserved associations, not abstention labels. Their support
    strings exist for auditing but must never be supplied to the online writer.
    """
    _count("train_size", train_size, MAX_TRAIN_SIZE)
    _count("eval_size", eval_size, MAX_TEST_SIZE)
    train_pool = synthetic_dataset(TRAIN_SEED, MAX_TRAIN_SIZE, 0)["stream"]
    evaluation = synthetic_dataset(TEST_SEED, MAX_TEST_SIZE, CONTROL_SIZE)
    data = {
        "train": _balanced_prefix(train_pool, train_size),
        "dev": synthetic_dataset(DEV_SEED, DEV_SIZE, 0)["stream"],
        "test": _balanced_prefix(evaluation["stream"], eval_size),
        "control": evaluation["control"],
    }
    data = {split: [replace(example, metadata={
        **example.metadata, "split": split,
        "protocol": PROTOCOL_VERSION, "never_written": split == "control",
    }) for example in rows] for split, rows in data.items()}
    identities = [example.metadata["entity"] for rows in data.values() for example in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Entity overlap between generalization fact splits")
    return data


def _fingerprint(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def protocol_manifest(train_size: int = MAX_TRAIN_SIZE,
                      eval_size: int = MAX_TEST_SIZE) -> dict:
    """Return source-independent protocol metadata, templates, and fact hashes."""
    data = datasets(train_size, eval_size)
    return {
        "protocol": PROTOCOL_VERSION,
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED, "test_and_control": TEST_SEED},
        "pool_sizes": {"train": MAX_TRAIN_SIZE, "dev": DEV_SIZE,
                       "test": MAX_TEST_SIZE, "control": CONTROL_SIZE},
        "counts": {split: len(rows) for split, rows in data.items()},
        "fact_sha256": {split: _fingerprint([asdict(example) for example in rows])
                        for split, rows in data.items()},
        "templates": [asdict(template) for template in _TEMPLATES],
        "answer_vocabulary": list(MEMORY_WORDS),
        "historical_paraphrase_is_training_template": "train_query_01",
        "confirmation_families": ["xml", "csv", "dialogue"],
        "limitations": [
            "Finite 16-word output vocabulary is shared across entity splits.",
            "Held-out expression families are authored stress tests, not a natural-language benchmark.",
            "Control accuracy measures guessing on never-written random facts, not abstention.",
            "Views of the same fact are correlated; uncertainty must resample facts, not views.",
        ],
    }


def protocol_fingerprint(train_size: int = MAX_TRAIN_SIZE,
                         eval_size: int = MAX_TEST_SIZE) -> str:
    return _fingerprint(protocol_manifest(train_size, eval_size))


def _validate_templates() -> None:
    # Catch protocol errors at import, before expensive feature preparation.
    if len(_BY_ID) != len(_TEMPLATES):
        raise RuntimeError("Duplicate template ID")
    families = [template.family for template in _TEMPLATES]
    if len(families) != len(set(families)):
        raise RuntimeError("Template family IDs must be unique")
    for template in _TEMPLATES:
        fields = {name for _, name, _, _ in Formatter().parse(template.text)
                  if name is not None}
        required = {"entity"} if template.kind == "query" else {"entity", "answer"}
        if fields != required:
            raise RuntimeError(f"Invalid placeholders in {template.id}: {fields}")


_validate_templates()
