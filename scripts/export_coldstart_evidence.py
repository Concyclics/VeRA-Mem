"""Export aggregate audit evidence, preserving hashes of fuller local originals.

Raw predictions, model weights and per-slot arrays remain in the backed workspace.
This is a presentation projection of completed audits, not another inference run.
"""
import argparse
import hashlib
import json
from pathlib import Path

OMIT={'sparse_counts','initial_norm','final_norm','delta_l2','relative_delta_l2',
      'cosine_similarity_to_parent','unit_direction_delta_l2'}

def compact(value):
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items()
        if not (k in OMIT and isinstance(v,list))}
    if isinstance(value,list):
        if len(value)>256 and all(isinstance(x,(int,float,bool)) for x in value):
            return dict(omitted_per_slot_array=True,entries=len(value))
        return [compact(v) for v in value]
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    original=a.input.read_bytes();source=json.loads(original)
    payload=compact(source)
    payload['public_export']=dict(source_path=str(a.input.resolve()),
        source_sha256=hashlib.sha256(original).hexdigest(),
        omitted_named_fields=sorted(OMIT),large_numeric_arrays_summarized=True,
        note='Aggregate projection; complete arrays and raw evidence remain in the checksum-backed workspace.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(a.output),bytes=a.output.stat().st_size)))

if __name__=='__main__':main()
