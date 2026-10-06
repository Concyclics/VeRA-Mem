"""Frozen CPU evaluation of independent foundation and episodic memory banks.

Only the offline-learned foundation prototypes are compressed. Every episodic
record and its counterfactual update follow the original interface evaluator.
Foundation routing statistics count actual CPU searches, including prompt,
repeated-prefix and answer-scoring positions; they are not first-token recall.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch

from .dictionary_vera import merge_bank
from .interface_eval import evaluate_interface
from .run import json_write, tensor_digest

PROTOCOL = 'coldstart-cpu-banks-v1'
MODES = ('full', 'off', 'random256', 'cluster256')
COMPRESSION_SEED = 92043


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_base_bank(module, mode='full', merge_count=None):
    """Read only checkpoint prototypes; never use eval documents or labels."""
    if mode not in MODES:
        raise ValueError(f'base_mode must be one of {MODES}')
    if merge_count is not None and (type(merge_count) is not int or merge_count < 1):
        raise ValueError('merge_count must be a positive integer')
    if mode in ('full', 'off') and merge_count is not None:
        raise ValueError('merge_count applies only to random256/cluster256')
    keys, values = module.base_keys.detach().cpu().float().clone(), module.base_values.detach().cpu().float().clone()
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values) or not len(keys):
        raise ValueError('Foundation parameters must be nonempty matching key/value matrices')
    if not bool(torch.isfinite(keys).all() and torch.isfinite(values).all()) or bool((keys.norm(dim=-1)==0).any()):
        raise ValueError('Foundation keys must be finite/nonzero and values finite')
    original_count = len(keys)
    result = dict(mode=mode, source_records=original_count, compression_seed=COMPRESSION_SEED,
                  original_keys=keys.clone(), original_values=values.clone(),
                  values_rms_recalibrated=False, count_bias_applied=False,
                  provenance='checkpoint foundation parameters; no evaluation features/text/answers used')
    if mode == 'full':
        selected = list(range(original_count))
    elif mode == 'off':
        selected = []
    else:
        count = merge_count if merge_count is not None else 256
        if count > original_count:
            raise ValueError('Compression cannot increase the foundation bank size')
        if mode == 'cluster256':
            merged = merge_bank(keys, values, count, split='train', seed=COMPRESSION_SEED)
            # split identifies learned training prototypes, not the evaluated packet.
            result.update(keys=merged['keys'], values=merged['values'],
                ids=[f'cluster-{i}' for i in range(count)], merge=merged,
                members=[(merged['assignments']==i).nonzero().flatten().tolist() for i in range(count)],
                active_records=count, source_indices=None, merge_count=count,
                exact_function_preserving=False)
            return result
        generator = torch.Generator(device='cpu').manual_seed(COMPRESSION_SEED)
        selected = sorted(torch.randperm(original_count, generator=generator)[:count].tolist())
    result.update(keys=keys[selected], values=values[selected], ids=[f'base-{i}' for i in selected],
                  members=[[i] for i in selected], active_records=len(selected), source_indices=selected,
                  merge_count=len(selected) if mode=='random256' else None,
                  exact_function_preserving=mode=='full')
    return result


class _Usage:
    def __init__(self, size):
        self.top1 = torch.zeros(size, dtype=torch.long)
        self.topk = torch.zeros(size, dtype=torch.long)
        self.mass = torch.zeros(size, dtype=torch.float64)
        self.search_calls = self.query_positions = 0

    def add(self, info):
        indices, weights = info['indices'].detach().cpu(), info['weights'].detach().cpu().double()
        if indices.shape != weights.shape or indices.ndim < 1:
            raise RuntimeError('Invalid foundation retrieval trace')
        self.search_calls += 1
        self.query_positions += math.prod(indices.shape[:-1])
        if not indices.shape[-1]:
            return
        flat = indices.reshape(-1, indices.shape[-1])
        if bool(((flat < 0) | (flat >= len(self.top1))).any()) or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
            raise RuntimeError('Invalid foundation index/weight')
        self.top1 += torch.bincount(flat[:,0], minlength=len(self.top1))
        self.topk += torch.bincount(flat.flatten(), minlength=len(self.topk))
        self.mass.index_add_(0, flat.flatten(), weights.flatten())

    def snapshot(self):
        total = float(self.mass.sum())
        probabilities = self.mass[self.mass>0]/total if total else self.mass[:0]
        entropy = float(-(probabilities*probabilities.log()).sum()) if total else 0.
        active = (self.topk>0).nonzero().flatten().tolist()
        return dict(search_calls=self.search_calls, query_positions=self.query_positions,
            records=len(self.top1), top1_used=int((self.top1>0).sum()), topk_used=len(active),
            top1_coverage=float((self.top1>0).float().mean()) if len(self.top1) else None,
            topk_coverage=float((self.topk>0).float().mean()) if len(self.topk) else None,
            weight_mass=total, weight_entropy=entropy, effective_slots=math.exp(entropy) if total else 0.,
            sparse_counts=[dict(index=i,top1=int(self.top1[i]),topk=int(self.topk[i]),weight_mass=float(self.mass[i])) for i in active])


@contextmanager
def _observe_reads(backend, store, output):
    """Delegate the original calls and record summable per-call usage only."""
    total, current = _Usage(len(store)), None
    missing = object()
    saved = {name:backend.__dict__.get(name,missing) for name in ('generate','score')}
    originals = {name:getattr(backend,name) for name in saved}
    original_search = store.search
    search_saved = store.__dict__.get('search',missing)
    generation_index, call_index = -1, 0
    before = store.hash()
    handle = Path(output).open('x',buffering=1)

    def search(*args, **kwargs):
        info = original_search(*args,**kwargs)
        total.add(info)
        if current is None:
            raise RuntimeError('Foundation CPU read outside generation/scoring')
        current.add(info)
        return info

    def wrapper(kind):
        def call(*args,**kwargs):
            nonlocal current,generation_index,call_index
            if current is not None:
                raise RuntimeError('Nested backend generation/scoring unsupported')
            if kind=='generate':
                generation_index += 1
            teacher = getattr(backend,'mode',None)=='none'
            current = _Usage(len(store))
            try:
                result = originals[kind](*args,**kwargs)
                stats = current.snapshot()
                if teacher and stats['search_calls']:
                    raise RuntimeError('Teacher unexpectedly read foundation memory')
                if store.hash()!=before:
                    raise RuntimeError('Foundation CPU bank mutated during evaluation')
                handle.write(json.dumps(dict(call_index=call_index,prediction_index=generation_index,
                    kind='generation' if kind=='generate' else 'answer_scoring',teacher=teacher,
                    bank_hash=before,**stats),allow_nan=False)+'\n')
                call_index += 1
                return result
            finally:
                current = None
        return call

    try:
        store.search = search
        backend.generate, backend.score = wrapper('generate'),wrapper('score')
        yield total
    finally:
        handle.close()
        if search_saved is missing:
            del store.search
        else:
            store.search = search_saved
        for name,value in saved.items():
            if value is missing:
                delattr(backend,name)
            else:
                setattr(backend,name,value)


def evaluate_coldstart(backend, module, packet, output, include_teacher=False,
                       max_cases=64, base_mode='full', merge_count=None):
    """Frozen top-k CPU reads from two banks; restore caller state on failure.

    ``empty`` in the inherited protocol disables the episodic bank only. The
    foundation contributes only with positive alpha and a nonempty bank.
    ``off + empty`` or ``alpha=0 + empty`` has no memory contribution. CPU base
    reads are retained at alpha=0 for consistent paths and cost auditing.
    Teacher always bypasses both banks.
    """
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name in ('predictions.jsonl','metrics.json','foundation.pt','foundation_usage.jsonl'):
        if (output/name).exists():
            raise FileExistsError('Refusing to overwrite coldstart evaluation: '+str(output/name))
    selected = prepare_base_bank(module,base_mode,merge_count)
    before = tensor_digest(module)
    flags = {name:p.requires_grad for name,p in module.named_parameters()}
    training, routing = module.training, module.routing_mode
    previous_cpu = getattr(module,'base_cpu_override',None)
    original_cpu_hash = previous_cpu.hash() if previous_cpu is not None else None
    stats, result, active_before, active_after = None,None,None,None
    try:
        module.requires_grad_(False).eval()
        module.set_routing_mode('sparse')
        with module.use_cpu_base(selected['keys'],selected['values'],ids=selected['ids']) as store:
            active_before = store.hash()
            torch.save(dict(protocol=PROTOCOL,**selected,store=store.snapshot(),bank_hash=active_before,
                alpha_base=module.alpha_base,base_top_k=module.base_top_k,
                original_shared_weights=before),output/'foundation.pt')
            with _observe_reads(backend,store,output/'foundation_usage.jsonl') as usage:
                result = evaluate_interface(backend,module,packet,output,
                    include_teacher=include_teacher,max_new_tokens=32,max_cases=max_cases)
                stats = usage.snapshot()
            active_after = store.hash()
            if active_after != active_before:
                raise RuntimeError('Foundation bank changed')
    finally:
        module.set_routing_mode(routing)
        for name,p in module.named_parameters():
            p.requires_grad_(flags[name])
        module.train(training)
        if tensor_digest(module)!=before:
            raise RuntimeError('Coldstart evaluation changed original module parameters/buffers')
        if getattr(module,'base_cpu_override',None) is not previous_cpu:
            raise RuntimeError('Previous CPU foundation context was not restored')
        if previous_cpu is not None and previous_cpu.hash()!=original_cpu_hash:
            raise RuntimeError('Previous foundation bank was mutated')
    effective = module.alpha_base > 0 and selected['active_records'] > 0
    effect_status = ('weight_zero' if module.alpha_base == 0 else
                     'bank_empty' if not selected['active_records'] else 'weighted_branch_enabled')
    weighted_top_k = min(module.base_top_k,selected['active_records']) if effective else 0
    result['coldstart'] = dict(protocol=PROTOCOL,base_mode=base_mode,
        source_records=selected['source_records'],active_records=selected['active_records'],
        merge_count=selected['merge_count'],compression_seed=COMPRESSION_SEED,
        foundation_snapshot='foundation.pt',foundation_snapshot_sha256=_sha(output/'foundation.pt'),
        foundation_usage='foundation_usage.jsonl',foundation_usage_sha256=_sha(output/'foundation_usage.jsonl'),
        bank_hash_before=active_before,bank_hash_after=active_after,
        original_shared_weights_before=before,original_shared_weights_after=tensor_digest(module),
        original_state_before=dict(training=training,routing_mode=routing,requires_grad=flags,cpu_bank_hash=original_cpu_hash),
        original_state_after=dict(training=module.training,routing_mode=module.routing_mode,
            requires_grad={n:p.requires_grad for n,p in module.named_parameters()},cpu_bank_hash=original_cpu_hash),
        evaluated_routing='sparse',foundation_cpu=True,episodic_cpu=True,
        alpha_base=module.alpha_base,base_top_k=module.base_top_k,episodic_top_k=module.top_k,
        foundation_effective=effective,foundation_effect_status=effect_status,
        foundation_weighted_top_k=weighted_top_k,
        maximum_weighted_records=min(module.top_k,len(packet['rows']))+weighted_top_k,
        foundation_effect_note='Effective means positive configured alpha and a nonempty bank, not verified behavioral usefulness. Maximum weighted records applies to real/shuffled/canonical-key reads with nonempty episodic memory; teacher bypasses both banks.',
        zero_weight_search_cost_included=True,
        values_rms_recalibrated=False,count_bias_applied=False,
        dynamic_records_compressed=False,usage=stats,
        usage_scope='All foundation search positions during generation and answer scoring, including prompt/repeated prefixes and any padding; not first-answer recall or unique corpus tokens. At alpha_base=0 these are audit-only routes with zero foundation output weight; search costs remain included.',
        empty_control='Episodic-only empty; foundation stays in selected mode and contributes only if alpha_base>0 and nonempty. off+empty or alpha_base=0+empty has no memory contribution.',
        compression_note='Cluster value arithmetic means retain magnitude; random is a fixed subset. Compression changes capacity/top-k multiplicity/softmax mass and is not function-preserving.')
    json_write(output/'metrics.json',result)
    return result
