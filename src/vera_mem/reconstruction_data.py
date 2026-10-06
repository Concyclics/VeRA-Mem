"""Small content-reconstruction data with frozen-writer generalization controls.

The model sees only rendered observations and entity-only questions. IDs,
worlds, donor IDs and answer strings are bookkeeping/target fields, never
extra model inputs. ``training_records`` is the only training projection:
canonical A/B observations and questions, with no C, D, SWAP or heldout text.

C/D keep A's first two words and introduce distinct unseen third-word combinations.
SWAP assigns another entity's A to the target, while that donor remains in the
original A bank. Evaluation must overwrite ONE target record at a time with
shared model parameters frozen. A full-bank permutation is a different test.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import random
import re
from typing import Callable, Sequence

from .data import MEMORY_WORDS


PROTOCOL = "reconstruction-write-read-v1"
SEED = 101042
SPLITS = ("train", "dev", "confirm")
DEFAULT_COUNTS = dict(train=16, dev=32, confirm=64)
WORLDS = ("A", "B", "C", "D", "SWAP")
RELATION = "stored content"
MAX_NEW_TOKENS = 32
WRITER_PREFIX = "Remember this information: "
FIRST_WORDS = tuple(MEMORY_WORDS)
SECOND_WORDS = (
    "amber", "birch", "cobalt", "crimson", "crystal", "delta", "dune", "fern",
    "flint", "frost", "granite", "hazel", "ivory", "jade", "lagoon", "lilac",
)
THIRD_WORDS = (
    "badger", "beaver", "buffalo", "camel", "cougar", "dolphin", "donkey", "eagle",
    "falcon", "gecko", "goose", "heron", "lizard", "monkey", "panda", "salmon",
)
INSTRUCTION = (
    "Return only the stored three words in their original order. "
    "Do not add a label, quotes, or an explanation."
)
CONDITIONS = {"known_ab": "B", "fresh_c": "C", "fresh_d": "D", "swap_a": "SWAP"}
_FIELDS = dict(A="a", B="b", C="c", D="d", SWAP="swap_a")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def stable_id(entity: str) -> str:
    return "reconstruction-" + digest([entity, RELATION])[:24]


def _check_entity(entity):
    if not isinstance(entity, str) or re.fullmatch(r"R[0-9a-f]{16}", entity) is None:
        raise ValueError("Expected an opaque reconstruction entity")


def render_question(entity: str, *, split="train", heldout=False) -> str:
    """Only entity and fixed task text enter the question, never a world label."""
    _check_entity(entity)
    if split not in SPLITS:
        raise ValueError("Unknown split")
    if not heldout:
        stem = f"Reconstruct the stored content for entity {entity}."
    elif split == "train":
        stem = f"Which three words were stored for {entity}? Reproduce them exactly."
    elif split == "dev":
        stem = f"Retrieve entity {entity}'s content and reproduce its three words."
    else:
        stem = f"RECONSTRUCTION REQUEST\nEntity to recall: {entity}\nRequested output: its stored content"
    return stem + "\n\n" + INSTRUCTION


def render_support(entity: str, answer: str, *, split="train", heldout=False) -> str:
    _check_entity(entity)
    _validate_answer(answer)
    if split not in SPLITS:
        raise ValueError("Unknown split")
    if not heldout:
        return f"Entity {entity} stores the content: {answer}."
    if split == "train":
        return f"The content assigned to {entity} is {answer}."
    if split == "dev":
        return f"For entity {entity}, the recorded content reads {answer}."
    return f"CONTENT RECORD\nOwner: {entity}\nStored words: {answer}\nEND RECORD"


def _validate_answer(answer):
    words = answer.split() if isinstance(answer, str) else []
    if (len(words) != 3 or answer != " ".join(words)
            or any(word not in vocabulary for word, vocabulary in
                   zip(words, (FIRST_WORDS, SECOND_WORDS, THIRD_WORDS)))):
        raise ValueError("Answer must contain three words from the position vocabularies")


def payload_span(support: str, answer: str) -> tuple[int, int]:
    """Character offsets of the unique observed three-word payload, not the ID.

    Pooling/token-offset conversion belongs to the runner. This helper never
    inserts target text: it requires the payload to already exist in support.
    """
    _validate_answer(answer)
    matches = list(re.finditer(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", support))
    if len(matches) != 1:
        raise ValueError("Support must contain exactly one complete observed payload")
    return matches[0].span()


def _balanced_pairs(rng, count, used, reserved):
    """Every 16 targets expose every component word once in each A/B world."""
    result = []
    for _ in range(count // 16):
        for _attempt in range(10000):
            first = rng.sample(list(FIRST_WORDS), 16)
            offset = rng.randrange(1, 16)
            a_middle, b_middle = [rng.sample(list(SECOND_WORDS), 16) for _ in range(2)]
            a_last, b_last = [rng.sample(list(THIRD_WORDS), 16) for _ in range(2)]
            pairs = [(" ".join((first[i], a_middle[i], a_last[i])),
                      " ".join((first[(i + offset) % 16], b_middle[i], b_last[i])))
                     for i in range(16)]
            answers = [answer for pair in pairs for answer in pair]
            occupied = Counter(" ".join(a.split()[:2]) for a in used)
            occupied.update(" ".join(a.split()[:2]) for a in answers)
            pending = Counter(reserved)
            pending.update({prefix: 2 * count for prefix, count in
                            Counter(" ".join(a.split()[:2]) for a, _ in pairs).items()})
            # Reserve two unseen tails per A prefix plus one spare: the C/D
            # tails must also avoid this entity's B tail, which may lie outside
            # its A prefix. This prevents valid larger budgets exhausting D.
            capacity_ok = all(occupied[prefix] + pending[prefix] <= len(THIRD_WORDS) - 1
                              for prefix in occupied.keys() | pending.keys())
            if len(set(answers)) == 32 and not set(answers) & used and capacity_ok:
                result.extend(pairs)
                used.update(answers)
                reserved.clear(); reserved.update(pending)
                break
        else:
            raise ValueError("Unable to construct a disjoint balanced answer block")
    return result


def datasets(train_size=16, dev_size=32, confirm_size=64, *, seed=SEED):
    """Return raw records, NOT a training-ready projection.

    Counts/seed are part of the sealed dataset definition. Changing a count
    defines a new dataset; nested-prefix stability is deliberately not claimed.
    No process-global random state is used. Train/dev/confirm share words but
    have disjoint entities and complete A/B/C/D answers. C is for development
    mechanisms; D is separately sealed for the final frozen-write check.
    """
    counts = dict(train=train_size, dev=dev_size, confirm=confirm_size)
    if type(seed) is not int or any(type(n) is not int or n < 16 or n > 128 or n % 16
                                    for n in counts.values()):
        raise ValueError("Counts must be multiples of 16 in [16, 128]; seed must be an integer")
    used_answers, result, reserved = set(), {}, Counter()
    for split in SPLITS:
        rng = random.Random(int(digest([seed, split, "AB"])[:16], 16))
        pairs = _balanced_pairs(rng, counts[split], used_answers, reserved)
        result[split] = []
        for index, (a, b) in enumerate(pairs):
            entity = "R" + digest([PROTOCOL, seed, split, index])[:16]
            result[split].append(dict(id=stable_id(entity), entity=entity,
                relation=RELATION, split=split, a=a, b=b))
    # All A/B labels are fixed before any C is chosen. Thus even a future
    # split's A/B cannot accidentally equal a frozen-write C target.
    for split in SPLITS:
        rng = random.Random(int(digest([seed, split, "controls"])[:16], 16))
        rows = result[split]
        indices = rng.sample(list(range(len(rows))), len(rows))
        donors = {}
        for left, right in zip(indices[::2], indices[1::2]):
            donors[left], donors[right] = right, left
        for index, row in enumerate(rows):
            first, middle, last = row["a"].split()
            candidates = [" ".join((first, middle, third)) for third in THIRD_WORDS
                          if third not in {last, row["b"].split()[2]}
                          and " ".join((first, middle, third)) not in used_answers]
            if not candidates:
                raise ValueError("No unused third-word combination remains")
            row["c"] = rng.choice(candidates)
            used_answers.add(row["c"])
            candidates = [answer for answer in candidates if answer != row["c"]]
            if not candidates:
                raise ValueError("No separate sealed D third-word combination remains")
            row["d"] = rng.choice(candidates)
            used_answers.add(row["d"])
            donor = rows[donors[index]]
            row["swap_a"] = donor["a"]
            row["worlds"] = list(WORLDS)
            row["questions"] = [render_question(row["entity"], split=split, heldout=h)
                                for h in (False, True)]
            row["supports"] = [[render_support(row["entity"], row[_FIELDS[w]],
                                split=split, heldout=h) for h in (False, True)] for w in WORLDS]
            row["provenance"] = dict(protocol=PROTOCOL, seed=seed,
                swap_donor_id=donor["id"], swap_donor_entity=donor["entity"],
                c_definition="development: A first two words plus a new third-word combination",
                d_definition="sealed confirmation: a different unseen third-word combination",
                train_allowed_worlds=["A", "B"], train_allowed_views=[0],
                bank_intervention="one target overwritten against unchanged original A bank")
    validate_records(result)
    return result


def training_records(rows):
    """Fresh dictionaries exposing exactly canonical A/B, excluding all probes.

    Fit feature statistics, teacher qualification for training, schedules and
    losses on this projection only. The runner must balance A/B for each ID.
    The source digest binds the full sealed record but reveals no probe text.
    """
    output = []
    for row in rows:
        if row["split"] != "train" or row.get("worlds") != list(WORLDS):
            raise ValueError("Training requires full raw training records")
        output.append(dict(id=row["id"], entity=row["entity"], relation=RELATION,
            split="train", a=row["a"], b=row["b"], worlds=["A", "B"],
            questions=[row["questions"][0]],
            supports=[[row["supports"][0][0]], [row["supports"][1][0]]],
            provenance=dict(protocol=PROTOCOL, source_split="train",
                projection="canonical_AB_only", source_record_sha256=digest(row))))
    validate_training_records(output)
    return output


def validate_training_records(rows):
    """Strict whitelist for the only rows allowed to fit shared parameters.

    Deliberately rejects token/cache/evaluation fields as well: annotate a
    separate packet, not the sealed text rows. An opaque source hash is allowed
    for lineage, but no probe string, donor metadata or heldout view is allowed.
    """
    expected = {"id", "entity", "relation", "split", "a", "b", "worlds",
                "questions", "supports", "provenance"}
    provenance = {"protocol", "source_split", "projection", "source_record_sha256"}
    if not rows or len(rows) % 16:
        raise ValueError("Training projection requires complete blocks of 16")
    identities, entities, answers = set(), set(), set()
    for row in rows:
        if set(row) != expected or set(row.get("provenance", {})) != provenance:
            raise ValueError("Training projection contains forbidden or missing fields")
        _check_entity(row["entity"])
        p = row["provenance"]
        if (row["split"] != "train" or row["relation"] != RELATION
                or row["worlds"] != ["A", "B"] or row["id"] != stable_id(row["entity"])
                or p["protocol"] != PROTOCOL or p["source_split"] != "train"
                or p["projection"] != "canonical_AB_only"
                or not isinstance(p["source_record_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", p["source_record_sha256"]) is None):
            raise ValueError("Training projection identity/world/provenance contract changed")
        for field in ("a", "b"):
            _validate_answer(row[field])
        if row["a"].split()[0] == row["b"].split()[0]:
            raise ValueError("Training A/B answers must differ in the first word")
        if (row["id"] in identities or row["entity"] in entities
                or row["a"] in answers or row["b"] in answers):
            raise ValueError("Duplicate training identity or answer")
        if (row["questions"] != [render_question(row["entity"])]
                or row["supports"] != [[render_support(row["entity"], row[field])]
                                        for field in ("a", "b")]):
            raise ValueError("Training projection must contain canonical A/B text only")
        identities.add(row["id"]); entities.add(row["entity"])
        answers.update((row["a"], row["b"]))
    for field in ("a", "b"):
        for position, vocabulary in enumerate((FIRST_WORDS, SECOND_WORDS, THIRD_WORDS)):
            if Counter(row[field].split()[position] for row in rows) != Counter(
                    {word: len(rows) // 16 for word in vocabulary}):
                raise ValueError("Training A/B component exposure is not balanced")
    return dict(records=len(rows), worlds=["A", "B"], views=1,
                canonical_only=True, probe_fields_absent=True,
                balanced_component_exposure=True, records_sha256=digest(rows))


def evaluation_records(rows):
    """Copy full records: A/B/C/D/SWAP world axis, canonical/heldout view axis."""
    return copy.deepcopy(list(rows))


def evaluation_pairs(rows, condition="known_ab"):
    """Adapt one control to a two-world evaluator without changing its A bank.

    ``b`` is the intervention target (original B, fresh C/D or donor A). Source
    identity/world and donor lineage remain explicit in CPU-only provenance.
    Fresh C/D/SWAP success must be reported separately from train A/B fitting.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown reconstruction evaluation condition")
    target = CONDITIONS[condition]
    output = []
    for row in rows:
        if row.get("worlds") != list(WORLDS):
            raise ValueError("Evaluation pairs require full raw records")
        output.append(dict(id=row["id"], entity=row["entity"], relation=RELATION,
            split=row["split"], a=row["a"], b=row[_FIELDS[target]], worlds=["A", target],
            questions=copy.deepcopy(row["questions"]),
            supports=copy.deepcopy([row["supports"][0], row["supports"][WORLDS.index(target)]]),
            provenance=dict(row["provenance"], condition=condition,
                source_worlds=["A", target], source_record_sha256=digest(row))))
    return output


def validate_records(splits):
    if set(splits) != set(SPLITS):
        raise ValueError("All three splits are required")
    all_entities, all_ids, all_answers = set(), set(), set()
    training_words = [{word for row in splits["train"] for w in ("a", "b")
                       for word in [row[w].split()[position]]} for position in range(3)]
    for split in SPLITS:
        rows = splits[split]
        if not rows or len(rows) % 16:
            raise ValueError("Each split requires complete blocks of 16")
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate record ID")
        entities = {row["entity"] for row in rows}
        answers = [row[w] for row in rows for w in ("a", "b", "c", "d")]
        if (len(entities) != len(rows) or entities & all_entities
                or set(by_id) & all_ids or len(set(answers)) != len(answers)
                or set(answers) & all_answers):
            raise ValueError("Entity, ID or complete A/B/C/D answer overlap")
        for row in rows:
            entity = row["entity"]
            _check_entity(entity)
            if (row["id"] != stable_id(entity) or row["split"] != split
                    or row["relation"] != RELATION or row["worlds"] != list(WORLDS)):
                raise ValueError("Record identity/world/split contract changed")
            for world in WORLDS:
                _validate_answer(row[_FIELDS[world]])
            a, b, c, d = (row[w].split() for w in ("a", "b", "c", "d"))
            if (a[0] == b[0] or a[:2] != c[:2] or a[:2] != d[:2]
                    or c[2] in {a[2], b[2]} or d[2] in {a[2], b[2], c[2]}):
                raise ValueError("A/B distinction or fresh C/D third-word contract changed")
            if any(word not in training_words[position] for label in (a, b, c, d)
                   for position, word in enumerate(label)):
                raise ValueError("A probe contains a component word absent from training A/B")
            donor = by_id.get(row["provenance"]["swap_donor_id"])
            if (donor is None or donor["id"] == row["id"] or row["swap_a"] != donor["a"]
                    or row["provenance"]["swap_donor_entity"] != donor["entity"]
                    or donor["provenance"]["swap_donor_id"] != row["id"]):
                raise ValueError("SWAP must be an in-split, nonidentity, reciprocal A donor")
            expected_questions = [render_question(entity, split=split, heldout=h) for h in (False, True)]
            if row["questions"] != expected_questions:
                raise ValueError("Question must contain entity-only canonical/heldout task text")
            for question in row["questions"]:
                if row["id"] in question or any(re.search(r"\b" + re.escape(word) + r"\b", question, re.I)
                    for label in (a, b, c, d, row["swap_a"].split()) for word in label):
                    raise ValueError("Question exposes an answer component or bookkeeping ID")
            expected_supports = [[render_support(entity, row[_FIELDS[w]], split=split, heldout=h)
                                  for h in (False, True)] for w in WORLDS]
            if row["supports"] != expected_supports:
                raise ValueError("Support must exactly bind the target entity to its world answer")
            for world, supports in zip(WORLDS, row["supports"]):
                for support in supports:
                    payload_span(support, row[_FIELDS[world]])
        all_entities.update(entities); all_ids.update(by_id); all_answers.update(answers)
    return dict(complete_answers_disjoint=True, entity_ids_disjoint=True,
                component_vocabulary_seen_in_training=True, question_answer_component_leaks=0,
                single_target_swap_has_original_bank_donor=True)


def validate_tokenization(rows, encode: Callable[[str], Sequence[int]], *, max_new_tokens=MAX_NEW_TOKENS):
    """Measure training A/B only; refuse C/D/heldout evaluation records here.

    Identical first tokens for A/C are intentional. Do not enforce the former
    first-token classification contract or silently replace any probe labels.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 2:
        raise ValueError("Invalid generation token budget")
    lengths, unequal = [], 0
    for row in rows:
        if (row.get("split") != "train" or row.get("worlds") != ["A", "B"]
                or len(row.get("questions", [])) != 1
                or [len(x) for x in row.get("supports", [])] != [1, 1]):
            raise ValueError("Tokenization preflight requires canonical A/B training projection")
        pair = []
        for field in ("a", "b"):
            tokens = list(encode(row[field]))
            if len(tokens) < 3 or any(type(x) is not int or x < 0 for x in tokens):
                raise ValueError("Three-word answers require valid multi-token sequences")
            if len(tokens) + 1 > max_new_tokens:
                raise ValueError("Answer plus EOS exceeds generation budget")
            lengths.append(len(tokens)); pair.append(len(tokens))
        unequal += pair[0] != pair[1]
    if not lengths:
        raise ValueError("No training records")
    return dict(answers=len(lengths), min_answer_tokens=min(lengths), max_answer_tokens=max(lengths),
                unequal_length_pairs=unequal, max_new_tokens=max_new_tokens, eos_reserved=True,
                scope="canonical training A/B only; no probe-dependent repair")


def protocol_manifest(splits=None):
    splits = datasets() if splits is None else splits
    checks = validate_records(splits)
    return dict(protocol=PROTOCOL, seed=next(iter(splits["train"]))["provenance"]["seed"],
        counts={split: len(rows) for split, rows in splits.items()},
        record_sha256={split: digest(rows) for split, rows in splits.items()},
        training_projection_sha256=digest(training_records(splits["train"])),
        worlds=list(WORLDS), conditions=dict(CONDITIONS), max_new_tokens=MAX_NEW_TOKENS,
        component_vocabularies=[list(x) for x in (FIRST_WORDS, SECOND_WORDS, THIRD_WORDS)],
        component_counts={split: [dict(Counter(row[field].split()[position]
            for row in rows for field in ("a", "b"))) for position in range(3)]
            for split, rows in splits.items()}, checks=checks,
        training_scope="A/B canonical only; balance both worlds for every target",
        evaluation_scope="Known entity A/B fit, development C, sealed D and donor A swaps; new entities and formats reported separately",
        uncertainty_unit="entity; reciprocal SWAP partners form a paired dependence cluster",
        limitations=["Synthetic random content, one relation and small banks; not natural-text generalization.",
            "C/D share A's first two words intentionally; report third-word and full-answer generation separately.",
            "Never use C/D/SWAP or heldout text to fit statistics, teacher prompts, schedules or losses.",
            "Teacher checks on heldout conditions report feasibility after model selection; do not tune on them.",
            "A/B fit alone cannot establish frozen-write generalization; compare real, empty and shuffled banks."])
