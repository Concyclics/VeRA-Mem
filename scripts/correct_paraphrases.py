"""Re-evaluate saved baseline checkpoints after fixing the paraphrase flag.

No optimization or writes to the original run. Results are a named supplement.
"""
import argparse
from pathlib import Path
import json
import time
import torch
from vera_mem.backend import QwenBackend
from vera_mem.memory import ExactMemory, LexicalMemory, LoRABank, FrozenRouter
from vera_mem.metrics import load_examples
from vera_mem.run import Experiment, aggregate, json_write, digest_file

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--wait-for-suite", type=Path)
a = p.parse_args()
if a.wait_for_suite:
    deadline = time.monotonic() + 1800
    while True:
        state = json.loads(a.wait_for_suite.read_text())
        if state.get("complete"):
            break
        if any(job.get("exit_code", 0) for job in state.get("jobs", [])):
            raise RuntimeError("Prerequisite experiment failed")
        if time.monotonic() > deadline:
            raise TimeoutError("Prerequisite experiment did not complete")
        time.sleep(10)
cfg = json.loads((a.run / "config.json").read_text())
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(cfg["cpu_threads"])
b = QwenBackend(a.model, cfg["layer"], cfg["max_input_tokens"])
stream = load_examples(a.run / "stream.jsonl")
priming = load_examples(a.run / "priming.jsonl")
observed = {x.id:x for x in priming+stream}
experiment = Experiment(b, cfg, a.output, "synthetic")
results = []
for method in cfg["methods"]:
    (a.output / method).mkdir()
    bank, router = None, None
    lexical = LexicalMemory()
    if method == "text_tfidf":
        for i, ex in enumerate(priming+stream):
            lexical.write(ex.id, ex.support, i)
    if method.startswith("lora"):
        slots = int(method.split("_")[0].removeprefix("lora"))
        rank = int(method.split("_r")[1])
        saved = torch.load(a.run / method / "adapter.pt", map_location="cuda", weights_only=True)
        bank = LoRABank(b.target.in_features, b.target.out_features, rank, slots, seed=cfg["seed"]).to("cuda")
        bank.load_state_dict(saved["bank"])
        bank.requires_grad_(False)
        if slots > 1:
            router = FrozenRouter(slots, cfg["seed"])
            router.load_state_dict(saved["router"])
    rows = experiment.evaluate(method, stream, "paraphrase_corrected", bank, router,
                               ExactMemory(), lexical, observed, paraphrase=True)
    result = {"method": method, "paraphrase": aggregate(rows), "correction": "original baseline evaluator omitted paraphrase=True; same saved checkpoint, true alternate-template queries, no optimization"}
    if bank is not None:
        result["checkpoint_sha256"] = digest_file(a.run / method / "adapter.pt")
    json_write(a.output / method / "metrics.json", result)
    results.append(result)
    print(json.dumps(result), flush=True)
json_write(a.output / "summary.json", results)
json_write(a.output / "manifest.json", {"complete": True, "source_stream_sha256":digest_file(a.run/'stream.jsonl'), "script_sha256":digest_file(__file__), "optimization_steps":0})
