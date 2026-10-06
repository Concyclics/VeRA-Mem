"""Frozen writer features with a verified content-only pooling mask."""
from __future__ import annotations
import torch

PREFIX = "Remember this information: "


def content_prompt(tokenizer, support):
    if not isinstance(support,str) or not support:
        raise ValueError("Nonempty support required")
    messages=[{"role":"system","content":"You are a helpful assistant. Follow the requested answer format."},
              {"role":"user","content":PREFIX+support}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    start=rendered.index(PREFIX)+len(PREFIX); end=start+len(support)
    if rendered[start:end] != support: raise AssertionError("Content offset mismatch")
    enc=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    # Boundary-crossing tokens are excluded, never silently include a wrapper.
    mask=[int(a>=start and b<=end and b>a) for a,b in enc["offset_mapping"]]
    if not any(mask): raise ValueError("Empty content mask")
    return enc["input_ids"],mask


@torch.no_grad()
def support_features(backend,supports,batch_size=32):
    last,pooled=[],[]
    for start in range(0,len(supports),batch_size):
        chunk=supports[start:start+batch_size]
        encoded=[content_prompt(backend.tokenizer,t) for t in chunk]
        sequences=[x[0] for x in encoded]
        if any(len(x)>backend.max_input_tokens for x in sequences): raise ValueError("No silent truncation")
        if any(x!=backend.prompt_ids(PREFIX+t) for x,t in zip(sequences,chunk)):
            raise AssertionError("Writer prompt diverges from historical backend")
        ids,attention=backend._right_pad(sequences)
        positions=attention.sum(-1)-1; rows=torch.arange(len(chunk),device=backend.device)
        content=torch.zeros_like(attention,dtype=torch.bool)
        for row,(_,mask) in enumerate(encoded): content[row,:len(mask)]=torch.tensor(mask,device=backend.device,dtype=torch.bool)
        captured=[]
        def hook(_module,args):
            x=args[0].float()
            captured.append((x[rows,positions].cpu(),
                             (x*content.unsqueeze(-1)).sum(1).div(content.sum(1,keepdim=True)).cpu()))
        h=backend.target.register_forward_pre_hook(hook)
        try:
            with backend.disabled(): backend.model.model(input_ids=ids,attention_mask=attention,use_cache=False)
        finally: h.remove()
        if len(captured)!=1: raise RuntimeError("Unexpected hook count")
        last.append(captured[0][0]); pooled.append(captured[0][1])
    return torch.cat(last),torch.cat(pooled)
