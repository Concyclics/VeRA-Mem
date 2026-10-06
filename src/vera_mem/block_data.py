"""Fresh, historically disjoint data for low-rank block-operator experiments.

The canonical train table and 16-fact factorial evaluation interface match QKV.
All identities and complete payloads exclude both reconstruction (101042) and
QKV (121042), using deterministic source datasets without accessing predictions.
Templates and component words are reused controls, not novel research formats.

The episode generator is deliberately reused verbatim from qkv_data. Its
protocol remains qkv-binding-episodes-v1, separately from this data protocol;
this changes dataset identity without quietly changing exposure/assignment rules.
"""
from __future__ import annotations

from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
import random
import re

from . import reconstruction_data as previous
from . import qkv_data as qkv


PROTOCOL = "block-outer-binding-data-v1"
EPISODE_PROTOCOL = qkv.PROTOCOL
SEED = 221042
SCHEDULE_SEED = 81042
TRAIN_ENTITIES = 64
TRAIN_PAYLOADS = 128
BATCH_SIZE = 8
BANK_SIZE = 16
EVAL_SIZE = 16
UPDATES = 2048
REGIMES = ("static", "rebind")
WORLDS = ("A", "B", "C", "D", "SWAP")
FIELDS = ("a", "b", "c", "d", "swap_a")
RELATION = previous.RELATION
WRITER_PREFIX = previous.WRITER_PREFIX
FIRST_WORDS, SECOND_WORDS, THIRD_WORDS = previous.FIRST_WORDS, previous.SECOND_WORDS, previous.THIRD_WORDS
VOCABULARIES = (FIRST_WORDS, SECOND_WORDS, THIRD_WORDS)
payload_span = previous.payload_span
render_question = previous.render_question
render_support = previous.render_support


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rng(seed, *parts):
    return random.Random(int(digest([PROTOCOL, seed, *parts])[:16], 16))


def _id(entity):
    return "block-" + digest([PROTOCOL, entity, RELATION])[:24]


def _entity(seed, split, index):
    return "R" + digest([PROTOCOL, seed, split, index])[:16]


@lru_cache(maxsize=1)
def _history():
    # Only deterministic data are consulted. No manifests, predictions or scores.
    recon = previous.datasets(seed=101042)
    former = qkv.dataset(seed=121042)
    re = frozenset(r["entity"] for rows in recon.values() for r in rows)
    ra = frozenset(r[f] for rows in recon.values() for r in rows for f in ("a", "b", "c", "d"))
    qe = frozenset(former["train"]["entities"]) | frozenset(
        r["entity"] for split in ("known", "dev", "confirm") for r in former[split])
    qa = frozenset(former["train"]["payloads"]) | frozenset(
        r[f] for split in ("known", "dev", "confirm") for r in former[split] for f in FIELDS)
    sources = ((previous.PROTOCOL, 101042, len(re), len(ra), digest(recon)),
               (qkv.PROTOCOL, 121042, len(qe), len(qa), digest(former)))
    return re | qe, ra | qa, sources


def _historical():
    return _history()[:2]


def history_manifest():
    """Return fresh metadata; callers cannot mutate cached exclusion sets."""
    entities, answers, sources = _history()
    return dict(sources=[dict(protocol=p, seed=s, entity_count=e, complete_answer_count=a,
                              dataset_sha256=h) for p, s, e, a, h in sources],
                excluded_entity_count=len(entities), excluded_complete_answer_count=len(answers),
                excluded_entities_sha256=digest(sorted(entities)),
                excluded_answers_sha256=digest(sorted(answers)))


def _payloads(seed, excluded):
    """Eight balanced 16-record blocks, with unique complete combinations."""
    rng = _rng(seed, "training-payloads")
    used = set(excluded); values = []
    for _ in range(TRAIN_PAYLOADS // 16):
        for _attempt in range(10000):
            columns = [rng.sample(list(v), len(v)) for v in VOCABULARIES]
            block = [" ".join(words) for words in zip(*columns)]
            if len(set(block)) == 16 and not set(block) & used:
                values.extend(block); used.update(block); break
        else:
            raise ValueError("Could not build a historically disjoint balanced payload block")
    # Randomize pair allocation without changing exact per-position counts.
    rng.shuffle(values)
    return values


def _train_row(entity, a, b):
    return dict(id=_id(entity), entity=entity, relation=RELATION, split="train", a=a, b=b,
        worlds=["A", "B"], questions=[render_question(entity)],
        supports=[[render_support(entity, x)] for x in (a, b)],
        provenance=dict(protocol=PROTOCOL, scope="canonical_static_reference_AB_only"))


def _eval_rows(seed, split, entities, answers):
    template_split = "train" if split == "known" else split
    rows = []
    for index, (entity, (a, b, c, d)) in enumerate(zip(entities, answers)):
        donor = index ^ 1
        swap = answers[donor][0]
        rows.append(dict(id=_id(entity), entity=entity, relation=RELATION, split=split,
            a=a, b=b, c=c, d=d, swap_a=swap, worlds=list(WORLDS),
            questions=[render_question(entity, split=template_split, heldout=h) for h in (False, True)],
            supports=[[render_support(entity, x, split=template_split, heldout=h) for h in (False, True)]
                      for x in (a, b, c, d, swap)],
            provenance=dict(protocol=PROTOCOL, seed=seed, paired_payload_index=index,
                edited_word_index=index % 3, swap_donor_id=_id(entities[donor]),
                swap_donor_entity=entities[donor], template_scope="historical_template_reused_not_new_to_study",
                content_scope="A/B are trained payloads; only known entities are trained; fixed known bindings need not receive target supervision under rebind; C/D are unseen complete combinations",
                bank_intervention="only target group replaced; all other records remain A")))
    return rows


def dataset(seed=SEED):
    """Return train-only table plus three paired 16-fact evaluation packets.

    ``train`` is the complete whitelist of training text/indices. It contains
    no heldout templates, C/D/SWAP labels or evaluation entity identifiers.
    Cross-product writer caches must encode the actual entity+payload text;
    swapping contextual hidden states from another entity is not a rebind.
    """
    _integer(seed, "seed")
    if seed in (101042, 121042):
        raise ValueError("Use a new data seed, not reconstruction 101042 or QKV 121042")
    old_entities, old_answers = _historical()
    entities = [_entity(seed, "train", i) for i in range(TRAIN_ENTITIES)]
    values = _payloads(seed, old_answers)
    train = dict(protocol=PROTOCOL, seed=seed, entities=entities, payloads=values,
                 rows=[_train_row(e, values[2*i], values[2*i+1]) for i, e in enumerate(entities)])
    used = set(values) | set(old_answers); rng = _rng(seed, "novel-eval-combinations")
    answers = []
    for i in range(EVAL_SIZE):
        a, b = values[2*i:2*i+2]; words = a.split(); position = i % 3
        candidates = []
        for token in VOCABULARIES[position]:
            altered = list(words); altered[position] = token; candidate = " ".join(altered)
            if candidate not in used and token != words[position]:
                candidates.append(candidate)
        if len(candidates) < 2:
            raise ValueError("Could not reserve distinct novel C/D one-word edits")
        c, d = rng.sample(candidates, 2); used.update((c, d)); answers.append((a, b, c, d))
    result = dict(protocol=PROTOCOL, seed=seed, train=train,
        known=_eval_rows(seed, "known", entities[:EVAL_SIZE], answers),
        dev=_eval_rows(seed, "dev", [_entity(seed, "dev", i) for i in range(EVAL_SIZE)], answers),
        confirm=_eval_rows(seed, "confirm", [_entity(seed, "confirm", i) for i in range(EVAL_SIZE)], answers),
        provenance=dict(history=history_manifest(), paired_eval_payloads=True,
            eval_bank_size=BANK_SIZE, episode_protocol=EPISODE_PROTOCOL,
            episode_generator="vera_mem.qkv_data.episode; unchanged target/background/bank/payload schedule",
            novelty_scope="All identities and complete payloads exclude reconstruction101042 and QKV121042; C/D exclude new training payloads",
            repeated_content_scope="known/dev/confirm deliberately share A/B/C/D by row to isolate entity effects",
            template_scope="same historical templates; heldout only relative to this round's training"))
    validate_dataset(result)
    return result


def validate_training_data(train):
    """Reject extra probe text/metadata in the training artifact whitelist."""
    if not isinstance(train, dict) or set(train) != {"protocol", "seed", "entities", "payloads", "rows"}:
        raise ValueError("Training table has missing or forbidden fields")
    if train["protocol"] != PROTOCOL:
        raise ValueError("Training protocol differs")
    _integer(train["seed"], "seed")
    if train["seed"] in (101042, 121042):
        raise ValueError("Historical data seed is forbidden")
    entities, values, rows = train["entities"], train["payloads"], train["rows"]
    if (not isinstance(entities, list) or len(entities) != TRAIN_ENTITIES
            or any(not isinstance(e, str) or re.fullmatch(r"R[0-9a-f]{16}", e) is None for e in entities)
            or len(set(entities)) != TRAIN_ENTITIES):
        raise ValueError("Invalid training entities")
    if (not isinstance(values, list) or len(values) != TRAIN_PAYLOADS
            or any(not isinstance(v, str) for v in values) or len(set(values)) != TRAIN_PAYLOADS):
        raise ValueError("Invalid training payloads")
    for value in values:
        previous._validate_answer(value)
    for pos, vocab in enumerate(VOCABULARIES):
        if Counter(v.split()[pos] for v in values) != Counter({w: 8 for w in vocab}):
            raise ValueError("Training component exposure is not balanced")
    old_entities, old_answers = _historical()
    if set(entities) & old_entities or set(values) & old_answers:
        raise ValueError("Historical entity or complete-answer overlap")
    if entities != [_entity(train["seed"], "train", i) for i in range(TRAIN_ENTITIES)]:
        raise ValueError("Training identity provenance differs")
    expected = [_train_row(e, values[2*i], values[2*i+1]) for i, e in enumerate(entities)]
    if rows != expected:
        raise ValueError("Training rows must be exact canonical static A/B references without probe fields")
    return dict(protocol=PROTOCOL, entities=TRAIN_ENTITIES, payloads=TRAIN_PAYLOADS,
                canonical_only=True, probe_fields_absent=True, train_sha256=digest(train))


def validate_dataset(value):
    if not isinstance(value, dict) or set(value) != {"protocol", "seed", "train", "known", "dev", "confirm", "provenance"}:
        raise ValueError("Dataset schema differs")
    if value["protocol"] != PROTOCOL or value["train"].get("seed") != value["seed"]:
        raise ValueError("Dataset protocol/seed differs")
    proof = validate_training_data(value["train"])
    train = value["train"]; train_values = set(train["payloads"])
    old_entities, old_answers = _historical(); new_values = set(); split_entities = {}
    for split in ("known", "dev", "confirm"):
        rows = value[split]
        if not isinstance(rows, list) or len(rows) != EVAL_SIZE:
            raise ValueError("Evaluation banks must all have 16 facts")
        entities = [r["entity"] for r in rows]
        expected_entities = train["entities"][:EVAL_SIZE] if split == "known" else [_entity(value["seed"], split, i) for i in range(EVAL_SIZE)]
        if entities != expected_entities or set(entities) & old_entities:
            raise ValueError("Evaluation identity provenance or historical overlap")
        split_entities[split] = set(entities)
        for i, row in enumerate(rows):
            if row["a"] != train["payloads"][2*i] or row["b"] != train["payloads"][2*i+1]:
                raise ValueError("Paired evaluation A/B contents differ from the training reference")
            for field in ("c", "d"):
                answer = row[field]; previous._validate_answer(answer)
                if answer in train_values or answer in old_answers:
                    raise ValueError("C/D must be historically and training unseen complete combinations")
                changed = [j for j, (a, b) in enumerate(zip(row["a"].split(), answer.split())) if a != b]
                if changed != [i % 3]:
                    raise ValueError("C/D must edit only the predeclared balanced word position")
                if split == "known":
                    if answer in new_values:
                        raise ValueError("C/D complete combinations must be distinct across targets/worlds")
                    new_values.add(answer)
            if split != "known" and any(row[f] != value["known"][i][f] for f in FIELDS):
                raise ValueError("Factorial evaluation payload pairing differs")
        answers = [tuple(r[f] for f in ("a", "b", "c", "d")) for r in rows]
        if rows != _eval_rows(value["seed"], split, entities, answers):
            raise ValueError("Evaluation question/support/SWAP/provenance differs")
    if split_entities["dev"] & set(train["entities"]) or split_entities["confirm"] & set(train["entities"]) or split_entities["dev"] & split_entities["confirm"]:
        raise ValueError("New evaluation entities overlap training/each other")
    provenance = value["provenance"]
    if (not isinstance(provenance, dict) or provenance.get("history") != history_manifest()
            or provenance.get("episode_protocol") != EPISODE_PROTOCOL
            or provenance.get("paired_eval_payloads") is not True
            or type(provenance.get("eval_bank_size")) is not int
            or provenance["eval_bank_size"] != BANK_SIZE):
        raise ValueError("Historical exclusion or episode provenance differs")
    return dict(proof, eval_facts_per_bank=EVAL_SIZE, novel_complete_payloads=len(new_values),
        edit_position_counts=dict(sorted(Counter(i % 3 for i in range(EVAL_SIZE)).items())),
        new_entity_splits_disjoint=True, eval_payloads_intentionally_paired=True,
        historical_exclusion_count=len(old_answers), historical_entity_exclusion_count=len(old_entities),
        episode_protocol=EPISODE_PROTOCOL, dataset_sha256=digest(value))


def training_data(value):
    """Copy the strict train table; never carry evaluation text into a cache."""
    validate_training_data(value["train"])
    return copy.deepcopy(value["train"])


# Keep the exact reviewed schedule, including its independent RNG streams and
# its historical protocol string. Data text/IDs are supplied by this module.
episode = qkv.episode
episodes = qkv.episodes
validate_episode = qkv.validate_episode
