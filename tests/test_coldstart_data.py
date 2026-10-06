"""Data leakage, counterfactual-edit and matched-label contracts."""
import copy
import random
from collections import defaultdict

import pytest

from vera_mem import coldstart_data as data
from vera_mem import interface_data as old


def letters(value):
    result=""
    while True:
        result=chr(97+value%26)+result
        value=value//26-1
        if value<0:return result


def article(i):
    return dict(id=str(i),title="Article "+letters(i),url="https://example.org/"+str(i),
        text=" ".join("word"+letters(i)+"z"+letters(j) for j in range(65)))


@pytest.fixture(scope="module")
def prepared():
    articles=[article(i) for i in range(1000)]
    # A title match and a body match form one transitive 3-page cluster.
    articles.extend([dict(article(1100),title=articles[0]["title"]),
                     dict(article(1200),text=article(1100)["text"])])
    return data.prepare_wikipedia(articles,dict(train=64,dev=4,confirm=4))


def test_article_split_dedup_and_answers_are_isolated(prepared):
    splits,stats=prepared
    assert stats["source_rows"]==1002
    assert stats["duplicate_cluster_rows_removed"]==2
    assert stats["matching_title_edges"]>=1 and stats["matching_body_edges"]>=1
    seen_pages=set();seen_clusters=set();seen_answers=set()
    for split,rows in splits.items():
        assert len(rows)==(64 if split=="train" else 4)
        pages={r["provenance"]["page_id"] for r in rows}
        clusters={r["provenance"]["cluster_id"] for r in rows}
        answers={data.normalize(r[k]) for r in rows for k in ("a","b")}
        assert not pages&seen_pages and not clusters&seen_clusters and not answers&seen_answers
        seen_pages|=pages;seen_clusters|=clusters;seen_answers|=answers
    data.validate_records(splits)


def test_real_a_bank_has_b_donor_and_single_edit_without_query_answer(prepared):
    for rows in prepared[0].values():
        by_page={r["provenance"]["page_id"]:r for r in rows}
        for row in rows:
            source=row["provenance"]
            donor=by_page[source["b_donor_page_id"]]
            assert donor["id"]!=row["id"] and donor["a"]==row["b"]
            assert row["a"].split()[0].casefold()!=row["b"].split()[0].casefold()
            assert 5<=source["anchor_words"]<=8 and source["passage_words"]<=120
            assert len(source["anchor"].split())==source["anchor_words"]
            for q in row["questions"]:
                assert source["anchor"] in q and row["entity"] in q
                assert row["id"] not in q
                assert all(data.normalize(row[k]) not in data.normalize(q) for k in ("a","b"))
            for view in range(2):
                a,b=row["supports"][0][view],row["supports"][1][view]
                assert a.count(row["a"])==1 and b==a.replace(row["a"],row["b"],1)


def test_matched_keeps_exact_labels_and_prompts_but_changes_only_support(prepared):
    for rows in prepared[0].values():
        matched=data.matched_records(rows)
        for original,new in zip(rows,matched):
            for key in ("id","entity","relation","a","b","questions","split"):
                assert original[key]==new[key]
            assert original["supports"]!=new["supports"]
            assert len(new["supports"][0][0].split())<len(original["supports"][0][0].split())
            assert "matched_support" not in original["provenance"]


def test_validation_rejects_leakage_and_multi_span_edits(prepared):
    rows=copy.deepcopy(prepared[0]);r=rows["dev"][0]
    r["questions"][0]+=" "+r["a"]
    with pytest.raises(ValueError,match="exposes answer"):data.validate_records(rows)
    rows=copy.deepcopy(prepared[0]);rows["dev"][0]["supports"][1][0]+=" extra"
    with pytest.raises(ValueError,match="one target-span"):data.validate_records(rows)
    rows=copy.deepcopy(prepared[0]);rows["dev"][0]["entity"]=rows["train"][0]["entity"]
    with pytest.raises(ValueError,match="cross-split"):data.validate_records(rows)


def test_fresh_synthetic_retains_actual_bank_negatives_and_excludes_history():
    random.seed(7);state=random.getstate()
    splits,stats=data.synthetic_records(dict(train=64,dev=16,confirm=16))
    assert random.getstate()==state
    old_rows=[f for rows in old._pools().values() for f in rows]
    old_answers={data.normalize(f.answer(w)) for f in old_rows for w in ("A","B")}
    old_entities={f.entity for f in old_rows}
    for split,rows in splits.items():
        assert len({r["provenance"]["group_id"] for r in rows})==len(rows)//16
        groups=defaultdict(list)
        for row in rows:
            assert row["entity"] not in old_entities
            assert all(data.normalize(row[k]) not in old_answers for k in ("a","b"))
            groups[row["provenance"]["group_id"]].append(row)
            for key,number in (("a",1),("b",2)):
                assert sum(r["entity"]!=row["entity"] and r["relation"]==row["relation"] and r["a"]==row[key] for r in rows)==number
            assert sum(r["entity"]==row["entity"] and r["relation"]!=row["relation"] for r in rows)==3
        assert all(len(group)==16 and len({r['entity'] for r in group})==4 for group in groups.values())
    assert splits==data.synthetic_records(dict(train=64,dev=16,confirm=16))[0]


def test_candidate_requires_real_three_word_unique_span_and_plain_word_limit():
    assert data.extract_candidate(dict(id="empty",text="short text")) is None
    assert data.extract_candidate(dict(id="numeric",text=" ".join(str(i) for i in range(120)))) is None
    candidate=data.extract_candidate(article(100))
    assert candidate and len(candidate["a"].split())==3 and candidate["passage"].count(candidate["a"])==1
    assert candidate["anchor"]+" "+candidate["a"] in candidate["passage"]


def test_target_replacement_preserves_real_overlapping_anchor_and_tail():
    anchor="References Sources 1916 Ship launches"
    old,new="Ship launches Ship","Marble silver canyon"
    source="Preface. "+anchor+" Ship launches Ship launches"
    # str.count misses overlapping occurrences; the old first-replace edited
    # the suffix of the question's anchor instead of its requested continuation.
    assert source.count(old)==1 and len(data.overlapping_occurrences(source,old))==2
    assert anchor not in source.replace(old,new,1)
    actual=data.replace_target_after_anchor(source,anchor,old,new)
    assert actual=="Preface. "+anchor+" Marble silver canyon launches"
    assert data.overlapping_occurrences(actual,anchor)==[len("Preface. ")]


@pytest.mark.parametrize("gap",[" ","\t","\n \t","\u00a0"])
@pytest.mark.parametrize("tail",["","\nTRAILER"])
def test_target_replacement_preserves_whitespace_framing_and_end_boundary(gap,tail):
    anchor="Unique [marker]+?"
    old,new="alpha beta gamma","delta epsilon zeta"
    source="HEADER\n"+anchor+gap+old+tail
    assert data.replace_target_after_anchor(source,anchor,old,new)=="HEADER\n"+anchor+gap+new+tail


@pytest.mark.parametrize("source,anchor,old,message",[
    ("other alpha beta gamma","missing","alpha beta gamma","exactly once"),
    ("MARK alpha beta gamma MARK","MARK","alpha beta gamma","exactly once"),
    ("aaaaa alpha beta gamma","aaa","alpha beta gamma","exactly once"),
    ("MARKalpha beta gamma","MARK","alpha beta gamma","whitespace"),
    ("MARK alpha beta gammas","MARK","alpha beta gamma","match old exactly"),
    ("MARK alpha beta gamma!","MARK","alpha beta gamma","match old exactly"),
    ("MARK alpha beta","MARK","alpha beta gamma","match old exactly"),
    ("MARK Alpha beta gamma","MARK","alpha beta gamma","match old exactly"),
    ("MARK wrong alpha beta gamma","MARK","alpha beta gamma","match old exactly"),
])
def test_target_replacement_rejects_ambiguous_anchor_or_nonexact_word_span(source,anchor,old,message):
    with pytest.raises(ValueError,match=message):
        data.replace_target_after_anchor(source,anchor,old,"delta epsilon zeta")


def test_overlapping_occurrences_are_literal_and_do_not_skip_shared_suffix():
    assert data.overlapping_occurrences("a.a.a","a.a")==[0,2]
    assert data.overlapping_occurrences("x [a]+ [a]+ y","[a]+")==[2,7]
    with pytest.raises(ValueError,match="nonempty"):
        data.overlapping_occurrences("text","")


def test_candidate_rejects_answers_repeated_only_at_overlapping_offsets():
    # Numeric surroundings disqualify every other 3-word candidate. Both
    # alternating spans occur twice with overlap but only once via str.count.
    words=[str(i) for i in range(12)]+["References","Sources","1916", "Ship","launches","Ship","launches","Ship","launches"]
    words += [str(100+i) for i in range(20)]
    passage=" ".join(words)
    assert passage.count("Ship launches Ship")==1
    assert len(data.overlapping_occurrences(data.normalize(passage),"ship launches ship"))==2
    assert data.extract_candidate(dict(id="overlap-only",text=passage)) is None


def overlap_record():
    anchor="References Sources 1916 Ship launches"
    a,b="Ship launches Ship","Marble silver canyon"
    passage="Preface "+anchor+" Ship launches Ship launches"
    changed="Preface "+anchor+" Marble silver canyon launches"
    note="Woverlap"
    return dict(id="stable-record-overlap",entity=note,relation="three-word continuation",split="train",a=a,b=b,
                questions=[data._question(note,anchor,h) for h in (False,True)],
                supports=[[data._support(note,p,h) for h in (False,True)] for p in (passage,changed)],
                provenance=dict(anchor=anchor))


def test_validation_accepts_addressed_overlap_and_matched_views_but_rejects_v1_edit():
    row=overlap_record()
    splits=dict(train=[row],dev=[],confirm=[])
    data.validate_records(splits)
    data.validate_records(dict(train=data.matched_records([row]),dev=[],confirm=[]))
    for view in range(2):
        broken=copy.deepcopy(splits)
        target=broken["train"][0]
        target["supports"][1][view]=target["supports"][0][view].replace(target["a"],target["b"],1)
        with pytest.raises(ValueError,match="exactly one target-span"):
            data.validate_records(broken)


def test_validation_rejects_repeated_anchor_and_unaddressed_overlapping_targets():
    row=overlap_record()
    row["supports"][0][0]+=" "+row["provenance"]["anchor"]
    with pytest.raises(ValueError,match="Anchor must occur exactly once"):
        data.validate_records(dict(train=[row],dev=[],confirm=[]))
    row=overlap_record()
    row["provenance"]={}
    row["questions"]=["What is the requested field?","Return the requested field."]
    row["supports"]=[["Ship launches Ship launches Ship"]*2,["Marble silver canyon launches Ship"]*2]
    assert row["supports"][0][0].count(row["a"])==1
    with pytest.raises(ValueError,match="exactly one target-span"):
        data.validate_records(dict(train=[row],dev=[],confirm=[]))
