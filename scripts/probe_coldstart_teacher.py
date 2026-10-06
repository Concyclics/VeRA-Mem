"""Teacher-only feasibility probe on fixed, disjoint Wikipedia training subsets.

python scripts/probe_coldstart_teacher.py --model MODEL --cache corrected/train.pt --run-dir NEW_DIR
First 16 eligible cache records are calibration; another 64 are sampled once with
a fixed seed from the remaining eligible train records. Eligibility excludes any
complete token-bounded normalized answer already present in the question, with
all exclusions recorded before inference. All three prompts are fixed before either
subset is scored. No dev/confirm examples, adapter, VDB, optimizer or automatic
prompt selection is used. Gold annotation is extra label-localization supervision.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from vera_mem.coldstart_data import replace_target_after_anchor
from vera_mem.coldstart_teacher import STRATEGIES,make_teacher_context,make_teacher_question,question_exposes_answer,strategy_metadata
from vera_mem.metrics import normalize_answer

PROTOCOL = "coldstart-teacher-feasibility-v1"
CACHE_PROTOCOL = "dictionary-coldstart-v1"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VALIDATION_SEED = 95043
CALIBRATION_SIZE,VALIDATION_SIZE,MAX_NEW_TOKENS = 16,64,32
SELECTION_POLICY = "train-only-label-disjoint-v2"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(2**20),b""):h.update(part)
    return h.hexdigest()


def json_write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def select_rows(packet):
    if (packet.get("protocol")!=CACHE_PROTOCOL or packet.get("model_revision")!=REVISION
            or packet.get("split")!="train" or packet.get("task")!="wikipedia"):
        raise ValueError("Probe accepts only the pinned corrected Wikipedia training cache")
    rows=packet["rows"]
    if len(rows)<CALIBRATION_SIZE+VALIDATION_SIZE or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Need at least 80 unique training records")
    # Validate all sources and annotate deterministic, pre-inference exclusions.
    # Malformed anchors remain fatal: never silently accept an unrepaired cache.
    eligible,excluded,substring_only,gold_failures = [],[],[],[]
    for index,row in enumerate(rows):
        if row.get("split")!="train" or len(row["questions"])!=1 or len(row["supports"])!=2 or any(len(s)!=1 for s in row["supports"]):
            raise ValueError("Only canonical training views may enter this probe")
        anchor=row.get("provenance",{}).get("anchor")
        if not anchor or anchor not in row["questions"][0]:raise ValueError("Missing literal question anchor")
        if normalize_answer(row["a"])==normalize_answer(row["b"]):raise ValueError("A/B answers must differ")
        leaked=[]
        normalized_question=normalize_answer(row["questions"][0])
        for world,label in enumerate(("a","b")):
            answer,context=row[label],row["supports"][world][0]
            if len(answer.split())!=3:raise ValueError("Answers must have exactly three whitespace words")
            if replace_target_after_anchor(context,anchor,answer,answer)!=context:
                raise ValueError("Counterfactual target is not immediately after its preserved anchor")
            normalized_answer=normalize_answer(answer)
            if question_exposes_answer(row["questions"][0],answer):
                leaked.append(dict(world=label.upper(),answer=answer,normalized_answer=normalized_answer,
                    reason="complete_token_bounded_normalized_answer_in_question"))
            elif normalized_answer in normalized_question:
                substring_only.append(dict(index=index,id=row["id"],world=label.upper(),answer=answer,
                    reason="partial_word_substring_only_not_a_complete_answer_phrase"))
            # Required before a possible gold-supervised full training run, not
            # merely on the 80 probe targets. Preserve any unsupported sources.
            try:make_teacher_context(row,context,answer,"gold_annotated")
            except ValueError as error:
                gold_failures.append(dict(index=index,id=row["id"],world=label.upper(),reason=str(error)))
        if leaked:
            excluded.append(dict(index=index,id=row["id"],question=row["questions"][0],
                provenance=row.get("provenance",{}),reasons=leaked))
        else:eligible.append(index)
    if len(eligible)<CALIBRATION_SIZE+VALIDATION_SIZE:
        raise ValueError("Fewer than 80 eligible train records after explicit question-label leakage exclusions: "+str(len(excluded)))
    indices=dict(calibration=eligible[:CALIBRATION_SIZE],
        validation=sorted(random.Random(VALIDATION_SEED).sample(eligible[CALIBRATION_SIZE:],VALIDATION_SIZE)))
    return dict(protocol=PROTOCOL,source_split="train",calibration_size=CALIBRATION_SIZE,
        validation_size=VALIDATION_SIZE,validation_seed=VALIDATION_SEED,source_records=len(rows),
        selection_policy=SELECTION_POLICY,eligible_records=len(eligible),excluded_count=len(excluded),excluded_records=excluded,
        exclusion_reason_counts={"complete_token_bounded_normalized_answer_in_question":len(excluded)},
        harmless_substring_count=len(substring_only),harmless_substring_matches=substring_only,
        normalization="Unicode NFKC + casefold + punctuation-to-spaces + whitespace collapse; articles preserved; match the complete answer at token boundaries.",
        teacher_context_validation=dict(strategy="gold_annotated",checked_contexts=2*len(rows),
            passed=not gold_failures,failed_contexts=len(gold_failures),failures=gold_failures,
            note="All training A/B contexts checked before inference, including records excluded only from feasibility sampling. Literal ampersands are preserved; no XML parser or label rewriting is used."),
        canonical_views_only=True,all_source_records_anchor_checked=True,
        selected_indices=indices,selected_ids={split:[rows[i]["id"] for i in ix] for split,ix in indices.items()},
        row_order_sha256=hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest(),
        selection_note="First 16 eligible train rows and a fixed, disjoint sample of 64 remaining eligible train rows. Only pre-inference question-label leakage excludes rows; no predictions, dev or confirm affect selection. Original training caches/schedules are not modified. Training validation is not held-out document generalization.")


def _model_hash(model):
    import torch
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        value=p.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _assert_teacher(backend):
    if backend.mode!="none" or backend.model.training or any(p.requires_grad or p.grad is not None for p in backend.model.parameters()):
        raise RuntimeError("Probe requires a frozen eval-mode teacher with no gradients")
    for name in ("vector_vera","vector_keys","vector_values","vector_store","vector_override","oracle_values"):
        if getattr(backend,name,None) is not None:raise RuntimeError("Teacher probe must not attach memory: "+name)


def _summarize(events,selection):
    groups=defaultdict(dict);costs=defaultdict(float)
    for e in events:
        key=(e["subset"],e["strategy"],e["id"])
        if e["world"] in groups[key]:raise RuntimeError("Repeated probe world")
        groups[key][e["world"]]=e
        for k in ("generation_tokens","generation_seconds","generation_forward_calls","generation_input_positions",
                  "prompt_tokens","answer_scoring_tokens","scoring_seconds","scoring_forward_calls","scoring_input_positions"):
            costs[k]+=e[k]
    result={}
    for subset in ("calibration","validation"):
        result[subset]={}
        for strategy in STRATEGIES:
            pairs=[groups[(subset,strategy,identity)] for identity in selection["selected_ids"][subset]]
            if any(set(pair)!={"A","B"} for pair in pairs):raise RuntimeError("Missing probe A/B pair")
            flat=[pair[w] for pair in pairs for w in ("A","B")]
            both=sum(pair["A"]["em"] and pair["B"]["em"] for pair in pairs)
            result[subset][strategy]=dict(**strategy_metadata(strategy),count=len(pairs),predictions=len(flat),
                a_correct=sum(pair["A"]["em"] for pair in pairs),b_correct=sum(pair["B"]["em"] for pair in pairs),
                both_correct=both,paired_em=both/len(pairs),teacher_pair_reaches_90_percent=both/len(pairs)>=.9,
                exact_literal_pair_correct=sum(pair["A"]["literal_em"] and pair["B"]["literal_em"] for pair in pairs),
                paired_containment=sum(pair["A"]["answer_containment"] and pair["B"]["answer_containment"] for pair in pairs)/len(pairs),
                answer_token_nll=sum(e["answer_nll_sum"] for e in flat)/sum(e["answer_scoring_tokens"] for e in flat),
                budget_hits=sum(e["budget_hit"] for e in flat),
                three_word_outputs=sum(e["prediction_words"]==3 for e in flat))
    costs["generation_calls"]=len(events);costs["scoring_calls"]=len(events)
    costs["total_backbone_forward_calls"]=costs["generation_forward_calls"]+costs["scoring_forward_calls"]
    costs["total_backbone_input_positions"]=costs["generation_input_positions"]+costs["scoring_input_positions"]
    return result,dict(costs)


def run_probe(backend,packet,output,selection=None):
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("predictions.jsonl","metrics.json","selection.json")):
        raise FileExistsError("Refusing to overwrite teacher probe")
    selection=select_rows(packet) if selection is None else selection
    if selection!=select_rows(packet):raise ValueError("Selection differs from the fixed train-only rule")
    if not selection["teacher_context_validation"]["passed"]:
        raise ValueError("Gold annotation cannot represent all training contexts; inspect selection.teacher_context_validation")
    _assert_teacher(backend)
    json_write(output/"selection.json",selection)
    before=_model_hash(backend.model);started=time.perf_counter()
    current={"kind":None,"calls":0,"positions":0}
    def count_forward(_model,args,kwargs):
        _assert_teacher(backend)
        if torch.is_grad_enabled() or current["kind"] is None:raise RuntimeError("Unexpected trainable/untracked forward")
        ids=kwargs.get("input_ids",args[0] if args else None)
        if ids is None or ids.ndim!=2 or ids.shape[0]!=1:raise RuntimeError("Expected one unpadded teacher sequence")
        current["calls"]+=1;current["positions"]+=ids.numel()
    hook=backend.model.register_forward_pre_hook(count_forward,with_kwargs=True)
    events=[]
    def synchronize():
        first=next(backend.model.parameters(),None)
        if first is not None and first.is_cuda:torch.cuda.synchronize(first.device)
    try:
        with torch.no_grad(),(output/"predictions.jsonl").open("x",buffering=1) as raw:
            for subset in ("calibration","validation"):
                for strategy in STRATEGIES:
                    for index in selection["selected_indices"][subset]:
                        row=packet["rows"][index];question=make_teacher_question(row,row["questions"][0],strategy)
                        for world,wi,label in (("A",0,"a"),("B",1,"b")):
                            answer,source=row[label],row["supports"][wi][0]
                            context=make_teacher_context(row,source,answer,strategy)
                            prompt_tokens=len(backend.prompt_ids(question,context=context))
                            current.update(kind="generation",calls=0,positions=0)
                            prediction,tokens,seconds=backend.generate(question,context=context,max_new_tokens=MAX_NEW_TOKENS)
                            generation_calls,generation_positions=current["calls"],current["positions"]
                            if generation_calls<1:raise RuntimeError("Teacher generation made no recorded model forward")
                            current.update(kind="scoring",calls=0,positions=0)
                            synchronize();score_started=time.perf_counter()
                            score=backend.score(question,answer,context=context)
                            synchronize();scoring_seconds=time.perf_counter()-score_started
                            if current["calls"]<1 or score.tokens<1:raise RuntimeError("Teacher scoring made no recorded model forward")
                            normalized,expected=normalize_answer(prediction),normalize_answer(answer)
                            event=dict(subset=subset,source_index=index,id=row["id"],strategy=strategy,world=world,
                                question=question,source_context=source,teacher_context=context,answer=answer,prediction=prediction,
                                **{k:v for k,v in strategy_metadata(strategy).items() if k!="strategy"},
                                em=int(normalized==expected),literal_em=int(prediction.strip()==answer),
                                answer_containment=int((" "+expected+" ") in (" "+normalized+" ")),
                                prediction_words=len(prediction.split()),budget_hit=tokens>=MAX_NEW_TOKENS,
                                generation_tokens=tokens,generation_seconds=seconds,prompt_tokens=prompt_tokens,
                                generation_forward_calls=generation_calls,generation_input_positions=generation_positions,
                                answer_nll_sum=float(score.nll_sum),answer_scoring_tokens=score.tokens,scoring_seconds=scoring_seconds,
                                scoring_forward_calls=current["calls"],scoring_input_positions=current["positions"])
                            current["kind"]=None
                            raw.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+"\n");events.append(event)
                    print(json.dumps(dict(subset=subset,strategy=strategy,completed_pairs=len(selection["selected_indices"][subset]))),flush=True)
    finally:
        hook.remove()
    _assert_teacher(backend);after=_model_hash(backend.model)
    if before!=after:raise RuntimeError("Teacher parameters changed")
    summaries,costs=_summarize(events,selection)
    result=dict(protocol=PROTOCOL,complete=True,source_split="train",max_new_tokens=MAX_NEW_TOKENS,
        teacher_only=True,shared_question=True,strategies=list(STRATEGIES),subsets=summaries,costs=costs,
        elapsed_seconds=time.perf_counter()-started,backbone_hash_before=before,backbone_hash_after=after,
        backbone_unchanged=True,gradient_steps=0,selection=selection,
        interpretation="Gold-annotated accuracy tests explicit label-localization supervision, not raw-context extraction. These are seen training-distribution feasibility checks; no dev/confirm selection or document-generalization claim.",
        cost_scope="Actual top-level model forward hooks count batch-1 input positions including cached generation and full scoring inputs. Generation time is backend-measured; scoring time synchronized. Costs are shared-device observations, not isolated latency or FLOPs.",
        artifacts={name:sha256(output/name) for name in ("predictions.jsonl","selection.json")})
    json_write(output/"metrics.json",result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args(argv)
    if args.run_dir.exists():raise FileExistsError("Use a fresh teacher probe directory")
    import torch
    from vera_mem.counterfactual_backend import CounterfactualBackend
    torch.set_num_threads(4);torch.manual_seed(VALIDATION_SEED)
    if json.loads((args.model.parent/"manifest.json").read_text()).get("revision")!=REVISION:
        raise ValueError("Teacher model revision differs")
    cache_hash=sha256(args.cache)
    packet=torch.load(args.cache,map_location="cpu",weights_only=True,mmap=True)
    selection=select_rows(packet)
    args.run_dir.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).resolve().parents[1]/"src/vera_mem/coldstart_teacher.py",
        Path(__file__).resolve().parents[1]/"src/vera_mem/coldstart_data.py"]
    manifest=dict(protocol=PROTOCOL,stage="teacher_probe",complete=False,started_at=datetime.now(timezone.utc).isoformat(),
        configuration={k:str(v) for k,v in vars(args).items()},cache_sha256=cache_hash,model_revision=REVISION,
        source_sha256={path.name:sha256(path) for path in sources},selection=selection,
        strategies={strategy:strategy_metadata(strategy) for strategy in STRATEGIES},no_dev_or_confirm=True)
    json_write(args.run_dir/"manifest.json",manifest)
    try:
        if not selection["teacher_context_validation"]["passed"]:
            raise ValueError("Gold annotation failed full-training preflight; counts/reasons retained in manifest.selection.teacher_context_validation")
        loaded=time.perf_counter();backend=CounterfactualBackend(str(args.model),layer=20)
        manifest["model_load_seconds"]=time.perf_counter()-loaded
        result=run_probe(backend,packet,args.run_dir,selection)
        if sha256(args.cache)!=cache_hash:raise RuntimeError("Input cache changed during teacher probe")
        if any(sha256(path)!=manifest["source_sha256"][path.name] for path in sources):raise RuntimeError("Probe source changed during execution")
        manifest.update(complete=True,finished_at=datetime.now(timezone.utc).isoformat(),result=result,backbone_unchanged=True)
        json_write(args.run_dir/"manifest.json",manifest)
        print(json.dumps(result["subsets"],ensure_ascii=False),flush=True)
    except BaseException as error:
        manifest.update(error=repr(error),failed_at=datetime.now(timezone.utc).isoformat())
        json_write(args.run_dir/"manifest.json",manifest);raise


if __name__=="__main__":main()
