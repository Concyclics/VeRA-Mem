"""Fresh article-cloze and synthetic records for the cold-start experiment.

No model, tensor, or network dependency. World A is a real Wikipedia excerpt;
world B is an explicitly counterfactual single-span edit, not a factual claim.
The second view is reserved for evaluation, even when present in train JSONL.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from . import interface_data as historical

PROTOCOL = "coldstart-article-cloze-v1"
SPLITS = ("train", "dev", "confirm")
SYNTHETIC_SEEDS = dict(train=71042, dev=72042, confirm=73042)
WIKI_SPLIT_SEED = 74042
INSTRUCTION = "Return only the three words, in their original order and capitalization. Do not add a label or explanation."
STOP_FIRST = frozenset("the a an and or of in to for with is was were are as by at on it its this that from be has had have not which who his her their he she they we you into also one two three".split())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def overlapping_occurrences(text, needle):
    if not isinstance(needle,str) or not needle:raise ValueError("A nonempty literal span is required")
    return [match.start() for match in re.finditer("(?="+re.escape(needle)+")",text)]


def replace_target_after_anchor(text, anchor, old, new):
    """Replace the uniquely addressed span, even if old overlaps elsewhere.

    The anchor must occur exactly once (including overlapping occurrences).
    Its next whitespace-delimited span must match old exactly. Never use the
    first occurrence of old: that occurrence can overlap and destroy anchor.
    """
    if not all(isinstance(x,str) and x for x in (text,anchor,old,new)):
        raise ValueError("Text, anchor and target strings must be nonempty")
    positions=overlapping_occurrences(text,anchor)
    if len(positions)!=1:raise ValueError("Anchor must occur exactly once")
    end=positions[0]+len(anchor);space=re.match(r"\s+",text[end:])
    if space is None:raise ValueError("Anchor must be followed by whitespace and target")
    start=end+space.end();stop=start+len(old)
    if text[start:stop]!=old or (stop<len(text) and not text[stop].isspace()):
        raise ValueError("The span immediately after anchor does not match old exactly")
    return text[:start]+new+text[stop:]


def _question(note, anchor, heldout=False):
    if heldout:
        stem = f"BEGIN NOTE LOOKUP\nNote identifier: {note}\nLocate this exact anchor: {anchor}\nRequested field: the next three words in that note\nEND NOTE LOOKUP"
    else:
        stem = f"In stored note {note}, what three words immediately follow the exact anchor '{anchor}'?"
    return stem + "\n\n" + INSTRUCTION


def _support(note, passage, heldout=False):
    return (f"BEGIN ARCHIVED NOTE {note}\n{passage}\nEND ARCHIVED NOTE" if heldout
            else f"Note {note}.\n{passage}")


def extract_candidate(article):
    """Take one <=120-whitespace-word source passage, with a unique plain span."""
    words = str(article.get("text", "")).split()[:120]
    if len(words) < 35:
        return None
    passage = " ".join(words)
    normalized_passage = normalize(passage)
    candidates = []
    anchor_length = 5 + int(digest(str(article["id"]))[:8], 16) % 4
    for index in range(max(anchor_length, 12), len(words) - 3):
        answer = " ".join(words[index:index + 3])
        anchor = " ".join(words[index-anchor_length:index])
        if not all(re.fullmatch(r"[A-Za-z]{2,18}", w) for w in words[index:index+3]):
            continue
        if words[index].casefold() in STOP_FIRST:
            continue
        if len(overlapping_occurrences(normalized_passage,normalize(answer))) != 1 or len(overlapping_occurrences(passage,anchor)) != 1:
            continue
        if normalize(answer) in normalize(anchor):
            continue
        candidates.append((index, answer, anchor))
    if not candidates:
        return None
    index, answer, anchor = candidates[int(digest([str(article["id"]), "span"])[:16], 16) % len(candidates)]
    return dict(passage=passage, a=answer, anchor=anchor, answer_word_index=index,
                anchor_words=anchor_length, passage_words=len(words))


def prepare_wikipedia(articles, counts=None):
    """Cluster exact identity/title/body/leading-passage duplicates before split.

    Only exact normalized-text matching is claimed. This does NOT detect all
    paraphrases, redirects, overlapping partial texts, or semantic duplicates.
    """
    counts = counts or dict(train=32768, dev=64, confirm=128)
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 2 for n in counts.values()):
        raise ValueError("All three split counts must be integers >= 2")
    rows, parents, indices = [], [], {}
    stats = Counter()

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for article in articles:
        stats["source_rows"] += 1
        title, body, identity = normalize(article.get("title", "")), normalize(article.get("text", "")), str(article.get("id", ""))
        if not title or not body or not identity:
            stats["empty_identity_title_or_body"] += 1
            continue
        candidate = extract_candidate(article)
        index = len(rows); parents.append(index)
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        entry = dict(page_id=identity, title=str(article["title"]), url=str(article.get("url", "")),
                     normalized_title_sha256=digest(title), normalized_body_sha256=body_hash,
                     candidate=candidate, source_shard=article.get("source_shard"), source_row=article.get("source_row"))
        rows.append(entry)
        keys = [("page_id", identity), ("title", title), ("body", body_hash)]
        if candidate:
            keys.append(("passage", digest(normalize(candidate["passage"]))))
        for key in keys:
            if key in indices:
                left, right = find(index), find(indices[key])
                stats["matching_"+key[0]+"_edges"] += 1
                if left != right:
                    parents[max(left,right)] = min(left,right)
            else:
                indices[key] = index
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    pools = defaultdict(list)
    stats["duplicate_cluster_rows_removed"] = len(rows)-len(groups)
    for members in groups.values():
        cluster_ids = sorted({rows[i]["page_id"] for i in members})
        cluster_id = digest(["wiki-cluster", cluster_ids])
        bucket = int(digest([WIKI_SPLIT_SEED, cluster_id])[:16], 16) % 100
        split = "dev" if bucket == 0 else "confirm" if bucket == 1 else "train"
        available = [rows[i] for i in members if rows[i]["candidate"]]
        if not available:
            stats["clusters_without_eligible_span"] += 1
            continue
        item = min(available, key=lambda x: (x["page_id"], x["normalized_body_sha256"]))
        item = dict(item, cluster_id=cluster_id, cluster_page_ids=cluster_ids)
        pools[split].append(item)
    chosen, used_answers = {}, set()
    for split in SPLITS:
        ordered = sorted(pools[split], key=lambda x: digest([WIKI_SPLIT_SEED, "select", x["cluster_id"]]))
        stats[split+"_eligible_clusters"] = len(ordered)
        selected, local_answers = [], set()
        for item in ordered:
            answer = normalize(item["candidate"]["a"])
            if answer in used_answers:
                stats[split+"_cross_split_answer_removed"] += 1
                continue
            if answer in local_answers:
                stats[split+"_duplicate_answer_removed"] += 1
                continue
            if len(selected) >= counts[split]:
                stats[split+"_eligible_beyond_quota"] += 1
                continue
            selected.append(item); local_answers.add(answer)
        if len(selected) != counts[split]:
            raise ValueError(f"Insufficient {split} candidates: {len(selected)} of {counts[split]}; counts={dict(stats)}")
        chosen[split] = selected; used_answers |= local_answers
    result = {}
    for split, items in chosen.items():
        records = []
        for i, item in enumerate(items):
            c = item["candidate"]
            note = "W" + digest(["coldstart-note", item["page_id"]])[:16]
            questions = [_question(note, c["anchor"], h) for h in (False,True)]
            donor = None
            for offset in range(1, len(items)):
                d = items[(i+offset)%len(items)]
                answer = d["candidate"]["a"]
                if (answer.split()[0].casefold() != c["a"].split()[0].casefold()
                    and normalize(answer) not in normalize(c["passage"])
                    and all(normalize(answer) not in normalize(q) for q in questions)):
                    donor = d; break
            if donor is None:
                raise ValueError("Could not assign different-article B answer")
            b = donor["candidate"]["a"]
            altered = replace_target_after_anchor(c["passage"],c["anchor"],c["a"],b)
            provenance = {k:v for k,v in item.items() if k != "candidate"}
            provenance.update(anchor=c["anchor"], answer_word_index=c["answer_word_index"], passage_words=c["passage_words"],
                anchor_words=c["anchor_words"], original_passage_sha256=digest(c["passage"]),
                b_donor_page_id=donor["page_id"], b_donor_cluster_id=donor["cluster_id"],
                world_a="source Wikipedia text",world_b="counterfactual single-span replacement",
                deduplication="exact normalized page ID/title/body/leading-passage clusters; no general near-duplicate claim")
            records.append(dict(id="coldwiki-"+digest(note)[:24], entity=note, relation="three-word continuation", split=split,
                a=c["a"],b=b, questions=questions, supports=[[_support(note,p,h) for h in (False,True)] for p in (c["passage"],altered)],
                provenance=provenance))
        result[split] = records
    validate_records(result)
    stats.update({split+"_selected":len(rows) for split,rows in result.items()})
    return result, dict(stats)


def matched_records(rows):
    """Same target tokens and questions; only the observed support is shortened."""
    result = copy.deepcopy(rows)
    for row in result:
        anchor = row["provenance"]["anchor"]
        row["supports"] = [[_support(row["entity"],anchor+" "+answer,h) for h in (False,True)] for answer in (row["a"],row["b"])]
        row["provenance"]["matched_support"] = "Only observation text differs; same labels, questions and note identifiers"
    return result


def synthetic_records(counts=None, extra_excluded_answers=()):
    """New seeds and explicit historical exclusions, without mutating old pools."""
    counts = counts or dict(train=4096,dev=64,confirm=128)
    if set(counts) != set(SPLITS) or any(type(v) is not int or v < 16 or v % 16 for v in counts.values()):
        raise ValueError("Synthetic counts must be positive complete groups of 16")
    old = historical._pools()
    excluded_entities = {f.entity for rows in old.values() for f in rows}
    excluded_answers = {normalize(f.answer(w)) for rows in old.values() for f in rows for w in ("A","B")}
    excluded_answers.update(map(normalize,extra_excluded_answers))
    result = {}; stats = Counter()
    for split in SPLITS:
        rng = random.Random(SYNTHETIC_SEEDS[split]); facts = []
        def entity():
            while True:
                value="I"+f"{rng.getrandbits(64):016x}"
                if value not in excluded_entities:
                    excluded_entities.add(value);return value
                stats["entity_resamples"]+=1
        def answer(exclude=None):
            while True:
                first=rng.choice(historical.FIRST_WORDS)
                value=" ".join((first,rng.choice(historical.TAIL_WORDS),rng.choice(historical.TAIL_WORDS)))
                if first != exclude and normalize(value) not in excluded_answers:
                    excluded_answers.add(normalize(value));return value
                stats["answer_resamples"]+=1
        for group in range(counts[split]//16):
            entities=[entity() for _ in range(4)]
            for relation in historical.RELATIONS:
                a=answer();b=answer(a.split()[0]);orientation=[False,False,True,True];rng.shuffle(orientation)
                for name,reverse in zip(entities,orientation):
                    facts.append((historical.InterfaceFact(historical.stable_id(name,relation),name,relation,b if reverse else a,a if reverse else b,split),group))
        rows=[]
        # Keep whole construction groups nested while not relying on entity order.
        for f,group in facts:
            questions=[historical.render_question(f,"if_train_query_00"),
                f"BEGIN ASSOCIATION LOOKUP\nEntity identifier: {f.entity}\nRequested association: {f.relation}\nEND ASSOCIATION LOOKUP\n\n{INSTRUCTION}"]
            supports=[[historical.render_support(f,"if_train_support_00",world=w),
                f"BEGIN ASSOCIATION RECORD\nEntity identifier: {f.entity}\nAssociation name: {f.relation}\nRecorded words: {f.answer(w)}\nEND ASSOCIATION RECORD"] for w in ("A","B")]
            rows.append(dict(id=f.id,entity=f.entity,relation=f.relation,split=split,a=f.answer_a,b=f.answer_b,
                questions=questions,supports=supports,provenance=dict(seed=SYNTHETIC_SEEDS[split],group_id=f"coldsynthetic-{split}-{group:05d}",
                    task="fresh synthetic four-entity/four-relation associations",historical_full_answers_excluded=True)))
        result[split]=rows
    validate_records(result)
    for split, rows in result.items():
        original_bank=defaultdict(set)
        for row in rows:original_bank[(row["relation"],row["a"])].add(row["entity"])
        for row in rows:
            for answer in (row["a"],row["b"]):
                assert original_bank[(row["relation"],answer)]-{row["entity"]}
    return result,dict(stats)


def validate_records(splits):
    if set(splits)!=set(SPLITS):raise ValueError("Missing split")
    seen_ids,set_entities,seen_answers=set(),set(),set()
    for split,rows in splits.items():
        ids={r["id"] for r in rows};entities={r["entity"] for r in rows};answers={normalize(r[k]) for r in rows for k in ("a","b")}
        if len(ids)!=len(rows) or ids&seen_ids or entities&set_entities or answers&seen_answers:
            raise ValueError("Duplicate identity/entity or cross-split complete-answer leakage")
        for row in rows:
            if row["split"]!=split or len(row["questions"])!=2 or any(len(v)!=2 for v in row["supports"]):raise ValueError("Record schema mismatch")
            if len(row["supports"])!=2 or any(len(row[k].split())!=3 for k in ("a","b")):raise ValueError("Three-word/world contract")
            if row["a"].split()[0].casefold()==row["b"].split()[0].casefold():raise ValueError("A/B first words collide")
            for q in row["questions"]:
                if any(normalize(row[k]) in normalize(q) for k in ("a","b")) or row["id"] in q:raise ValueError("Question exposes answer or stable record ID")
            for s in range(2):
                anchor=row.get("provenance",{}).get("anchor")
                expected=(replace_target_after_anchor(row["supports"][0][s],anchor,row["a"],row["b"]) if anchor
                          else row["supports"][0][s].replace(row["a"],row["b"],1))
                if (not anchor and len(overlapping_occurrences(row["supports"][0][s],row["a"]))!=1) or row["supports"][1][s]!=expected:
                    raise ValueError("Support update is not exactly one target-span replacement")
        seen_ids|=ids;set_entities|=entities;seen_answers|=answers


def load_records(directory, domain, split):
    if domain not in {"wikipedia","matched","synthetic"} or split not in SPLITS:raise ValueError("Unknown domain/split")
    return [json.loads(line) for line in (Path(directory)/domain/(split+".jsonl")).read_text().splitlines() if line]
