import argparse
from pathlib import Path
from vera_mem.data import prepare_medmcqa

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--n-train", type=int, default=64)
p.add_argument("--n-control", type=int, default=16)
a = p.parse_args()
data = prepare_medmcqa(a.output, a.seed, a.n_train, a.n_control)
print({key: len(value) for key, value in data.items()})
